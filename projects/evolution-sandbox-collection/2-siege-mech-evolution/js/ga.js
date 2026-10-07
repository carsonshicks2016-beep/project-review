/**
 * ga.js — Genetic Algorithm for Mech Evolution
 * Siege Mech Evolution
 *
 * MechDNA  – Encodes physical traits (limb lengths, muscle params, masses)
 *            and a small neural network for turret aiming.
 * MechGA   – Selection, crossover, mutation, and generational evolution.
 */

/* ================================================================== */
/*  Template gene counts                                              */
/* ================================================================== */

/**
 * Return the number of physical genes needed for a given mech template.
 * This must stay in sync with MechBody._build* methods.
 *
 * Layout per template:
 *   limbLengths  (bones + muscles)  – one float each
 *   muscleAmps   (per muscle)
 *   muscleFreqs  (per muscle)
 *   musclePhases (per muscle)
 *   nodeMasses   (per node)
 */
function _geneLayout(template) {
  switch (template) {
    case 'biped':
      return { limbCount: 31, muscleCount: 11, nodeCount: 17 };
    case 'tripod':
      return { limbCount: 13, muscleCount: 3, nodeCount: 8 };
    case 'quadruped':
      return { limbCount: 24, muscleCount: 8, nodeCount: 12 };
    default:
      return { limbCount: 31, muscleCount: 11, nodeCount: 17 };
  }
}

function _physicalGeneCount(template) {
  const l = _geneLayout(template);
  return l.limbCount + l.muscleCount * 3 + l.nodeCount;
}

/* ================================================================== */
/*  Neural-network sizing                                             */
/* ================================================================== */

/** Brain architecture constants */
const BRAIN_INPUTS  = 9;
const BRAIN_HIDDEN  = 12;
const BRAIN_OUTPUTS = 4;

/** Total brain weights + biases */
const BRAIN_GENE_COUNT =
  BRAIN_INPUTS * BRAIN_HIDDEN + BRAIN_HIDDEN +       // input→hidden weights + hidden biases
  BRAIN_HIDDEN * BRAIN_OUTPUTS + BRAIN_OUTPUTS;       // hidden→output weights + output biases

/* ================================================================== */
/*  MechDNA                                                           */
/* ================================================================== */

class MechDNA {
  /**
   * @param {string} template  'biped' | 'tripod' | 'quadruped'
   */
  constructor(template) {
    this.template = template;

    /** Physical genes [0, 1] */
    this.physicalGenes = new Array(_physicalGeneCount(template)).fill(0.5);

    /** Brain genes [-1, 1] — weights for the aiming neural network */
    this.brainGenes = new Array(BRAIN_GENE_COUNT).fill(0);

    /** Cached fitness (set externally) */
    this.fitness = 0;
  }

  /* ================================================================ */
  /*  Decode physical genes into a config object                      */
  /* ================================================================ */

  /**
   * Convert normalised genes into concrete mech parameters.
   * @returns {object} Config object for MechBody constructor
   */
  decode() {
    const layout = _geneLayout(this.template);
    let idx = 0;

    // Limb length multipliers: map [0,1] → [0.6, 1.4]
    const limbLengths = [];
    for (let i = 0; i < layout.limbCount; i++) {
      limbLengths.push(0.6 + this.physicalGenes[idx++] * 0.8);
    }

    // Muscle amplitudes: map [0,1] → [0.05, 0.5]
    const muscleAmps = [];
    for (let i = 0; i < layout.muscleCount; i++) {
      muscleAmps.push(0.05 + this.physicalGenes[idx++] * 0.45);
    }

    // Muscle frequencies: map [0,1] → [1, 8]
    const muscleFreqs = [];
    for (let i = 0; i < layout.muscleCount; i++) {
      muscleFreqs.push(1 + this.physicalGenes[idx++] * 7);
    }

    // Muscle phases: map [0,1] → [0, 2π]
    const musclePhases = [];
    for (let i = 0; i < layout.muscleCount; i++) {
      musclePhases.push(this.physicalGenes[idx++] * Math.PI * 2);
    }

    // Node mass multipliers: map [0,1] → [0.5, 3.0]
    const nodeMasses = [];
    for (let i = 0; i < layout.nodeCount; i++) {
      nodeMasses.push(0.5 + this.physicalGenes[idx++] * 2.5);
    }

    return { limbLengths, muscleAmps, muscleFreqs, musclePhases, nodeMasses };
  }

  /* ================================================================ */
  /*  Neural-network forward pass (turret brain)                      */
  /* ================================================================ */

  /**
   * Run the aiming network.
   *
   * Architecture: 9 → 12 (tanh) → 4 (tanh)
   *
   * @param {number[]} inputs  Length-9 input vector
   * @returns {number[]}       Length-4 output vector
   */
  forward(inputs) {
    const w = this.brainGenes;
    let ptr = 0;

    // Input → Hidden (BRAIN_INPUTS × BRAIN_HIDDEN weights, then BRAIN_HIDDEN biases)
    const hidden = new Array(BRAIN_HIDDEN);
    for (let h = 0; h < BRAIN_HIDDEN; h++) {
      let sum = 0;
      for (let i = 0; i < BRAIN_INPUTS; i++) {
        sum += inputs[i] * w[ptr++];
      }
      sum += w[ptr++]; // bias
      hidden[h] = Math.tanh(sum);
    }

    // Hidden → Output (BRAIN_HIDDEN × BRAIN_OUTPUTS weights, then BRAIN_OUTPUTS biases)
    const outputs = new Array(BRAIN_OUTPUTS);
    for (let o = 0; o < BRAIN_OUTPUTS; o++) {
      let sum = 0;
      for (let h = 0; h < BRAIN_HIDDEN; h++) {
        sum += hidden[h] * w[ptr++];
      }
      sum += w[ptr++]; // bias
      outputs[o] = Math.tanh(sum);
    }

    return outputs;
  }
}

/* ================================================================== */
/*  MechGA — Genetic Algorithm Operators                              */
/* ================================================================== */

const MechGA = {
  /* -------- creation -------- */

  /**
   * Create a random MechDNA for the given template.
   * @param {string} template
   * @returns {MechDNA}
   */
  createRandom(template) {
    const dna = new MechDNA(template);
    const layout = _geneLayout(template);

    for (let i = 0; i < dna.physicalGenes.length; i++) {
      dna.physicalGenes[i] = Math.random();
    }
    // Initialize muscle amplitudes to be small [0.0, 0.25] to ensure initial upright stability
    const ampStart = layout.limbCount;
    for (let i = ampStart; i < ampStart + layout.muscleCount; i++) {
      dna.physicalGenes[i] = Math.random() * 0.25;
    }

    for (let i = 0; i < dna.brainGenes.length; i++) {
      dna.brainGenes[i] = (Math.random() * 2 - 1);
    }
    return dna;
  },

  /* -------- fitness -------- */

  /**
   * Evaluate fitness for a mech body.
   * @param {MechBody} mech
   * @param {number} locoWeight     Weight for locomotion component
   * @param {number} targetWeight   Weight for targeting component
   * @returns {number}
   */
  evaluateFitness(mech, locoWeight, targetWeight) {
    let locoScore = mech.distanceMoved;
    if (mech.isFallen) {
      locoScore *= 0.05; // 95% penalty for falling over / rolling
    }
    // Add survival reward (up to 15s * 2.0 = 30 points) to guide biped balance
    const survivalReward = (mech.survivalTime || 0) * 2.0;

    const targetScore = mech.hits * 100 - mech.misses * 30;
    return locoWeight * (locoScore + survivalReward) + targetWeight * targetScore;
  },

  /* -------- selection -------- */

  /**
   * Select one individual from the population.
   * @param {MechDNA[]} population
   * @param {number[]} fitnesses
   * @param {string} method  'tournament' | 'roulette'
   * @returns {MechDNA}
   */
  select(population, fitnesses, method) {
    if (method === 'roulette') {
      return this._rouletteSelect(population, fitnesses);
    }
    return this._tournamentSelect(population, fitnesses);
  },

  _tournamentSelect(pop, fits) {
    let bestIdx = -1;
    let bestFit = -Infinity;
    for (let i = 0; i < 3; i++) {
      const idx = Math.floor(Math.random() * pop.length);
      if (fits[idx] > bestFit) {
        bestFit = fits[idx];
        bestIdx = idx;
      }
    }
    return pop[bestIdx];
  },

  _rouletteSelect(pop, fits) {
    // Shift fitnesses so all are positive
    const minFit = Math.min(...fits);
    const shifted = fits.map(f => f - minFit + 1);
    const total = shifted.reduce((a, b) => a + b, 0);
    let r = Math.random() * total;
    for (let i = 0; i < pop.length; i++) {
      r -= shifted[i];
      if (r <= 0) return pop[i];
    }
    return pop[pop.length - 1];
  },

  /* -------- crossover -------- */

  /**
   * Uniform crossover between two parents.
   * @param {MechDNA} parentA
   * @param {MechDNA} parentB
   * @param {number} rate  Per-gene crossover probability
   * @returns {MechDNA}
   */
  crossover(parentA, parentB, rate) {
    const child = new MechDNA(parentA.template);

    for (let i = 0; i < child.physicalGenes.length; i++) {
      child.physicalGenes[i] = Math.random() < rate
        ? parentB.physicalGenes[i]
        : parentA.physicalGenes[i];
    }

    for (let i = 0; i < child.brainGenes.length; i++) {
      child.brainGenes[i] = Math.random() < rate
        ? parentB.brainGenes[i]
        : parentA.brainGenes[i];
    }

    return child;
  },

  /* -------- mutation -------- */

  /**
   * Gaussian mutation.
   * @param {MechDNA} dna
   * @param {number} rate     Per-gene probability of mutation
   * @param {number} strength Standard deviation of the Gaussian noise
   */
  mutate(dna, rate, strength) {
    for (let i = 0; i < dna.physicalGenes.length; i++) {
      if (Math.random() < rate) {
        dna.physicalGenes[i] += this._gaussian() * strength;
        dna.physicalGenes[i] = Math.max(0, Math.min(1, dna.physicalGenes[i]));
      }
    }

    for (let i = 0; i < dna.brainGenes.length; i++) {
      if (Math.random() < rate) {
        dna.brainGenes[i] += this._gaussian() * strength;
        dna.brainGenes[i] = Math.max(-2, Math.min(2, dna.brainGenes[i]));
      }
    }
  },

  /**
   * Box-Muller Gaussian (mean 0, std 1).
   * @returns {number}
   */
  _gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  },

  /* -------- generational evolution -------- */

  /**
   * Produce the next generation.
   *
   * @param {MechDNA[]} population   Current generation
   * @param {number[]}  fitnesses    Parallel fitness array
   * @param {object}    config
   *   - mutationRate   {number}
   *   - mutationStrength {number}
   *   - crossoverRate  {number}
   *   - selectionMethod {string}
   * @returns {MechDNA[]}  New population (same length)
   */
  evolveGeneration(population, fitnesses, config) {
    const popSize = population.length;
    const next = [];

    // Sort indices by fitness descending
    const indices = population.map((_, i) => i);
    indices.sort((a, b) => fitnesses[b] - fitnesses[a]);

    // Elitism — keep top 2 unchanged
    const eliteCount = Math.min(2, popSize);
    for (let i = 0; i < eliteCount; i++) {
      const elite = population[indices[i]];
      const copy = new MechDNA(elite.template);
      copy.physicalGenes = elite.physicalGenes.slice();
      copy.brainGenes = elite.brainGenes.slice();
      next.push(copy);
    }

    // Fill remaining via selection + crossover + mutation
    while (next.length < popSize) {
      const pA = this.select(population, fitnesses, config.selectionMethod || 'tournament');
      const pB = this.select(population, fitnesses, config.selectionMethod || 'tournament');
      const child = this.crossover(pA, pB, config.crossoverRate || 0.5);
      this.mutate(child, config.mutationRate || 0.1, config.mutationStrength || 0.3);
      next.push(child);
    }

    return next;
  }
};
