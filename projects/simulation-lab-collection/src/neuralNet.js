// Network topology. Input width comes from src/observation.js so the net and the
// observation builder cannot disagree.
//
// 22 inputs: relative current gate (3), relative next gate (3), local velocity (3),
// angular velocity (3), heading alignment (1), height (1), and 8 wall-distance sensors.
//
// Note: an earlier experiment that widened this to 22 by adding gate NORMALS dropped the
// GA from 12/16 to 3/16 -- extra search dimensions cost more than redundant information is
// worth. The sensors are different: without them the policy cannot perceive the house at
// all, so the information is load-bearing rather than redundant.
import { OBS_DIM } from './observation.js';

export const LAYER_SIZES = [OBS_DIM, 16, 12, 4];

export class NeuralNet {
  constructor(layerSizes = LAYER_SIZES) {
    this.layerSizes = layerSizes;
    this.totalWeights = NeuralNet.getWeightCount(layerSizes);
  }

  static getWeightCount(layerSizes) {
    let count = 0;
    for (let i = 0; i < layerSizes.length - 1; i++) {
      count += layerSizes[i] * layerSizes[i + 1] + layerSizes[i + 1];
    }
    return count;
  }

  forward(inputs, weights) {
    let currentInputs = new Float32Array(inputs);
    let weightOffset = 0;

    for (let i = 0; i < this.layerSizes.length - 1; i++) {
      const inDim = this.layerSizes[i];
      const outDim = this.layerSizes[i + 1];
      const nextInputs = new Float32Array(outDim);

      for (let j = 0; j < outDim; j++) {
        let sum = 0;
        for (let k = 0; k < inDim; k++) {
          sum += currentInputs[k] * weights[weightOffset + j * inDim + k];
        }
        sum += weights[weightOffset + inDim * outDim + j]; // bias
        nextInputs[j] = Math.tanh(sum);
      }
      
      weightOffset += inDim * outDim + outDim;
      currentInputs = nextInputs;
    }

    return {
      thrust: (currentInputs[0] + 1) * 0.5,
      pitch: currentInputs[1],
      roll: currentInputs[2],
      yaw: currentInputs[3]
    };
  }
}
