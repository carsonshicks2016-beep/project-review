// Minimal differentiable MLP: forward, backward, Adam.
//
// Hand-rolled rather than pulling in TensorFlow.js: the networks here are a few thousand
// parameters, the project has no build step, and the rollout workers are classic workers
// that cannot import modules anyway (the code gets inlined there). A dependency that
// large would also fight three.js for the WebGL context.
//
// Layout: all parameters live in one flat Float32Array so they can be shipped to workers
// as a transferable and fed to Adam without reshaping.

export function layerOffsets(layerSizes) {
  const offsets = [];
  let o = 0;
  for (let i = 0; i < layerSizes.length - 1; i++) {
    const inDim = layerSizes[i];
    const outDim = layerSizes[i + 1];
    offsets.push({ w: o, b: o + inDim * outDim, inDim, outDim });
    o += inDim * outDim + outDim;
  }
  return { offsets, total: o };
}

export class MLP {
  // Hidden layers use tanh; the output layer is linear so it can represent action means
  // and state values without being squashed.
  constructor(layerSizes) {
    this.layerSizes = layerSizes;
    const { offsets, total } = layerOffsets(layerSizes);
    this.offsets = offsets;
    this.numParams = total;
    this.params = new Float32Array(total);
    this.initOrthogonalish();
  }

  // Scaled uniform init per fan-in. PPO implementations usually use orthogonal init with
  // a small final-layer gain so the initial policy is near-deterministic and low-variance;
  // approximating that with a 0.01 gain on the output layer matters more than the exact
  // distribution at this size.
  initOrthogonalish() {
    for (let l = 0; l < this.offsets.length; l++) {
      const { w, b, inDim, outDim } = this.offsets[l];
      const isLast = l === this.offsets.length - 1;
      const scale = isLast ? 0.01 : Math.sqrt(2 / inDim);
      for (let i = 0; i < inDim * outDim; i++) {
        this.params[w + i] = (Math.random() * 2 - 1) * scale;
      }
      for (let i = 0; i < outDim; i++) this.params[b + i] = 0;
    }
  }

  makeCache() {
    // acts[0] is the input; acts[l+1] is the output of layer l.
    return this.layerSizes.map(n => new Float32Array(n));
  }

  // Writes activations into `acts` and returns the output layer.
  forward(x, acts, params = this.params) {
    acts[0].set(x);
    for (let l = 0; l < this.offsets.length; l++) {
      const { w, b, inDim, outDim } = this.offsets[l];
      const src = acts[l];
      const dst = acts[l + 1];
      const isLast = l === this.offsets.length - 1;
      for (let j = 0; j < outDim; j++) {
        let sum = params[b + j];
        const row = w + j * inDim;
        for (let k = 0; k < inDim; k++) sum += src[k] * params[row + k];
        dst[j] = isLast ? sum : Math.tanh(sum);
      }
    }
    return acts[acts.length - 1];
  }

  // Accumulates dL/dparams into `grads` given dL/dOutput. Requires `acts` from a matching
  // forward pass. Hidden activations are tanh outputs, so the local derivative is 1 - a^2
  // and no pre-activation needs storing.
  backward(acts, dOut, grads, params = this.params) {
    const L = this.offsets.length;
    let delta = dOut;

    for (let l = L - 1; l >= 0; l--) {
      const { w, b, inDim, outDim } = this.offsets[l];
      const src = acts[l];

      for (let j = 0; j < outDim; j++) {
        const d = delta[j];
        if (d === 0) continue;
        grads[b + j] += d;
        const row = w + j * inDim;
        for (let k = 0; k < inDim; k++) grads[row + k] += d * src[k];
      }

      if (l > 0) {
        const prev = new Float32Array(inDim);
        for (let k = 0; k < inDim; k++) {
          let sum = 0;
          for (let j = 0; j < outDim; j++) sum += delta[j] * params[w + j * inDim + k];
          // src is the tanh output of layer l-1
          prev[k] = sum * (1 - src[k] * src[k]);
        }
        delta = prev;
      }
    }
  }
}

export class Adam {
  constructor(numParams, { lr = 3e-4, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.t = 0;
    this.m = new Float32Array(numParams);
    this.v = new Float32Array(numParams);
  }

  // Clips the global gradient norm before stepping, which is what keeps PPO updates from
  // occasionally blowing up on an outlier batch.
  step(params, grads, maxGradNorm = 0.5) {
    this.t++;

    if (maxGradNorm > 0) {
      let sq = 0;
      for (let i = 0; i < grads.length; i++) sq += grads[i] * grads[i];
      const norm = Math.sqrt(sq);
      if (norm > maxGradNorm) {
        const s = maxGradNorm / (norm + 1e-6);
        for (let i = 0; i < grads.length; i++) grads[i] *= s;
      }
    }

    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let i = 0; i < params.length; i++) {
      const g = grads[i];
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * g;
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * g * g;
      const mh = this.m[i] / bc1;
      const vh = this.v[i] / bc2;
      params[i] -= this.lr * mh / (Math.sqrt(vh) + this.eps);
    }
  }
}

// Running mean/variance for observation normalization. PPO is markedly less stable
// without it when inputs have wildly different scales, which these do (metres vs radians).
export class RunningNorm {
  constructor(dim) {
    this.dim = dim;
    this.mean = new Float32Array(dim);
    this.var = new Float32Array(dim).fill(1);
    this.count = 1e-4;
  }

  update(batch, n) {
    // Chan et al. parallel variance: merges a batch's moments without a second pass.
    const bMean = new Float64Array(this.dim);
    const bVar = new Float64Array(this.dim);
    for (let i = 0; i < n; i++) {
      const off = i * this.dim;
      for (let d = 0; d < this.dim; d++) bMean[d] += batch[off + d];
    }
    for (let d = 0; d < this.dim; d++) bMean[d] /= n;
    for (let i = 0; i < n; i++) {
      const off = i * this.dim;
      for (let d = 0; d < this.dim; d++) {
        const diff = batch[off + d] - bMean[d];
        bVar[d] += diff * diff;
      }
    }
    for (let d = 0; d < this.dim; d++) bVar[d] /= n;

    const totalCount = this.count + n;
    for (let d = 0; d < this.dim; d++) {
      const delta = bMean[d] - this.mean[d];
      const mA = this.var[d] * this.count;
      const mB = bVar[d] * n;
      const M2 = mA + mB + delta * delta * this.count * n / totalCount;
      this.mean[d] += delta * n / totalCount;
      this.var[d] = M2 / totalCount;
    }
    this.count = totalCount;
  }

  normalizeInto(x, out) {
    for (let d = 0; d < this.dim; d++) {
      out[d] = Math.max(-10, Math.min(10, (x[d] - this.mean[d]) / Math.sqrt(this.var[d] + 1e-8)));
    }
    return out;
  }

  serialize() {
    return { mean: Array.from(this.mean), var: Array.from(this.var), count: this.count };
  }
}
