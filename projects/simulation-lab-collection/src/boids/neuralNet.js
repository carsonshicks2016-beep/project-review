/**
 * ApexFlock - High-Performance Co-Evolutionary Neural Network
 * Features:
 * - Layer-by-layer forward pass with activation caching for real-time HUD inspection
 * - Multi-activation support (LeakyReLU, Tanh, Sigmoid)
 * - Genetic operators: Gaussian mutation, weight flipping, crossover
 * - Speciation genetic distance metric
 * - JSON serialization / deserialization
 */

export class NeuralNetwork {
  constructor(topology) {
    // topology is an array of layer sizes, e.g. [22, 16, 12, 5]
    this.topology = [...topology];
    this.numLayers = topology.length;
    
    // Weights and biases
    this.weights = []; // Array of Float32Array [layerIndex][fromNode * toLayerSize + toNode]
    this.biases = [];  // Array of Float32Array [layerIndex][toNode]
    
    // Cached activations for visualization & inspection: array of Float32Array per layer
    this.activations = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.activations.push(new Float32Array(topology[i]));
    }

    this._initWeights();
  }

  _initWeights() {
    this.weights = [];
    this.biases = [];
    for (let l = 0; l < this.numLayers - 1; l++) {
      const fanIn = this.topology[l];
      const fanOut = this.topology[l + 1];
      
      // Xavier / He initialization
      const scale = Math.sqrt(2.0 / (fanIn + fanOut));
      const w = new Float32Array(fanIn * fanOut);
      for (let i = 0; i < w.length; i++) {
        w[i] = (Math.random() * 2 - 1) * scale;
      }
      this.weights.push(w);

      const b = new Float32Array(fanOut);
      for (let i = 0; i < b.length; i++) {
        b[i] = (Math.random() * 2 - 1) * 0.1;
      }
      this.biases.push(b);
    }
  }

  /**
   * Forward pass: computes activations through all layers.
   * Caches activations so the UI can visualize neural firing in real time.
   * @param {Array|Float32Array} inputs 
   * @returns {Float32Array} output layer activations
   */
  forward(inputs) {
    const inputLayer = this.activations[0];
    const nIn = Math.min(inputs.length, inputLayer.length);
    for (let i = 0; i < nIn; i++) {
      inputLayer[i] = inputs[i];
    }

    for (let l = 0; l < this.numLayers - 1; l++) {
      const inNodes = this.topology[l];
      const outNodes = this.topology[l + 1];
      const currentActs = this.activations[l];
      const nextActs = this.activations[l + 1];
      const w = this.weights[l];
      const b = this.biases[l];
      const isLastLayer = (l === this.numLayers - 2);

      for (let j = 0; j < outNodes; j++) {
        let sum = b[j];
        const offset = j * inNodes;
        for (let i = 0; i < inNodes; i++) {
          sum += currentActs[i] * w[offset + i];
        }

        if (isLastLayer) {
          // Steering (pitch, yaw, roll) use Tanh (-1 to 1)
          // Thrust / Burst and Signals use Sigmoid (0 to 1)
          if (j < 3) {
            nextActs[j] = Math.tanh(sum);
          } else {
            nextActs[j] = 1.0 / (1.0 + Math.exp(-Math.max(-10, Math.min(10, sum))));
          }
        } else {
          // LeakyReLU for hidden layers
          nextActs[j] = sum > 0 ? sum : sum * 0.05;
        }
      }
    }

    return this.activations[this.numLayers - 1];
  }

  /**
   * Clones this neural network into a new instance with identical weights.
   */
  clone() {
    const copy = new NeuralNetwork(this.topology);
    for (let l = 0; l < this.numLayers - 1; l++) {
      copy.weights[l].set(this.weights[l]);
      copy.biases[l].set(this.biases[l]);
    }
    return copy;
  }

  /**
   * Applies genetic mutation with adaptive strength.
   * @param {number} rate - Probability of mutating each weight (e.g. 0.08)
   * @param {number} power - Gaussian standard deviation for weight change (e.g. 0.3)
   */
  mutate(rate = 0.08, power = 0.3) {
    for (let l = 0; l < this.numLayers - 1; l++) {
      const w = this.weights[l];
      for (let i = 0; i < w.length; i++) {
        if (Math.random() < rate) {
          const u1 = Math.max(1e-7, Math.random());
          const u2 = Math.random();
          const gaussian = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
          
          if (Math.random() < 0.05) {
            w[i] = -w[i];
          } else if (Math.random() < 0.03) {
            w[i] = (Math.random() * 2 - 1) * 2.0;
          } else {
            w[i] += gaussian * power;
          }
          w[i] = Math.max(-5.0, Math.min(5.0, w[i]));
        }
      }

      const b = this.biases[l];
      for (let i = 0; i < b.length; i++) {
        if (Math.random() < rate) {
          const u1 = Math.max(1e-7, Math.random());
          const u2 = Math.random();
          const gaussian = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
          b[i] += gaussian * (power * 0.5);
          b[i] = Math.max(-3.0, Math.min(3.0, b[i]));
        }
      }
    }
  }

  /**
   * Performs uniform crossover with another parent network.
   * @param {NeuralNetwork} partner 
   * @returns {NeuralNetwork} child network
   */
  crossover(partner) {
    const child = this.clone();
    for (let l = 0; l < this.numLayers - 1; l++) {
      const childW = child.weights[l];
      const partW = partner.weights[l];
      for (let i = 0; i < childW.length; i++) {
        if (Math.random() < 0.5) {
          childW[i] = partW[i];
        }
      }
      const childB = child.biases[l];
      const partB = partner.biases[l];
      for (let i = 0; i < childB.length; i++) {
        if (Math.random() < 0.5) {
          childB[i] = partB[i];
        }
      }
    }
    return child;
  }

  /**
   * Computes genetic distance between two neural networks for speciation.
   * @param {NeuralNetwork} other 
   * @returns {number} normalized genetic distance
   */
  geneticDistance(other) {
    let diffSum = 0;
    let totalCount = 0;
    for (let l = 0; l < this.numLayers - 1; l++) {
      const w1 = this.weights[l];
      const w2 = other.weights[l];
      for (let i = 0; i < w1.length; i++) {
        diffSum += Math.abs(w1[i] - w2[i]);
        totalCount++;
      }
    }
    return totalCount > 0 ? diffSum / totalCount : 0;
  }

  /**
   * Serializes neural network to plain JSON object.
   */
  toJSON() {
    return {
      topology: this.topology,
      weights: this.weights.map(w => Array.from(w)),
      biases: this.biases.map(b => Array.from(b))
    };
  }

  /**
   * Deserializes neural network from JSON object.
   */
  static fromJSON(data) {
    const net = new NeuralNetwork(data.topology);
    for (let l = 0; l < net.numLayers - 1; l++) {
      net.weights[l].set(data.weights[l]);
      net.biases[l].set(data.biases[l]);
    }
    return net;
  }
}
