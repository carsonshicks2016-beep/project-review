const ADJECTIVES = ["Apex","Blitz","Nova","Aero","Quantum","Zephyr","Vortex","Turbo","Hyper","Mach","Echo","Stealth","Rogue","Phantom","Neon"];
const NOUNS = ["Falcon","Spark","Pulse","Glider","Viper","Hawk","Dart","Striker","Hornet","Mantis","Spectre","Drifter","Wraith","Comet","Bolt"];
const generatedNames = new Set();

function generateUniqueName() {
  let name = "";
  let attempts = 0;
  do {
    const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
    const noun = NOUNS[Math.floor(Math.random() * NOUNS.length)];
    const num = Math.floor(Math.random() * 900) + 100;
    name = `${adj} ${noun}-${num}`;
    attempts++;
  } while (generatedNames.has(name) && attempts < 1000);
  generatedNames.add(name);
  return name;
}

export class GeneticAlgorithm {
  constructor(popSize = 120, genomeLength, config = {}) {
    this.popSize = popSize;
    this.genomeLength = genomeLength;
    // He-style init scale depends on fan-in. This was hardcoded to sqrt(2/14) for the
    // original 14-input network; with a different input count the scale is wrong and
    // tanh units start saturated.
    this.inputSize = config.inputSize || 14;
    this.mutationRate = config.mutationRate || 0.08;
    this.mutationScale = config.mutationScale || 0.2;
    this.elitismCount = Math.floor(popSize * 0.1);
    this.generation = 0;
    this.bestFitness = -Infinity;
    this.avgFitness = 0;
    
    this.population = Array.from({ length: popSize }, () => this.randomGenome());
    this.metadata = Array.from({ length: popSize }, () => ({
        name: generateUniqueName(),
        lineage: []
    }));
  }

  randomGenome() {
    const genome = new Float32Array(this.genomeLength);
    const scale = Math.sqrt(2 / this.inputSize);
    for (let i = 0; i < this.genomeLength; i++) {
      genome[i] = (Math.random() * 2 - 1) * scale;
    }
    return genome;
  }

  nextGeneration(fitnessScores) {
    const rankedList = fitnessScores.map((score, index) => ({
      score,
      genome: this.population[index],
      meta: this.metadata[index],
      index
    }));

    rankedList.sort((a, b) => b.score - a.score);

    this.bestFitness = rankedList[0].score;
    this.avgFitness = rankedList.reduce((sum, item) => sum + item.score, 0) / this.popSize;
    const worstFitness = rankedList[rankedList.length - 1].score;

    const newPopulation = [];
    const newMetadata = [];

    // Elitism
    for (let i = 0; i < this.elitismCount; i++) {
      newPopulation.push(new Float32Array(rankedList[i].genome));
      newMetadata.push({
          name: rankedList[i].meta.name, 
          lineage: [...rankedList[i].meta.lineage] 
      });
    }

    // Breed rest
    while (newPopulation.length < this.popSize) {
      const parentA = this.tournamentSelect(rankedList);
      const parentB = this.tournamentSelect(rankedList);
      const child = this.crossover(parentA.genome, parentB.genome);
      this.mutate(child);
      
      const childName = generateUniqueName();
      
      // Combine lineages, keeping only the last 3 to avoid infinite arrays
      let combinedLineage = [];
      if (parentA.meta.name === parentB.meta.name) {
          combinedLineage = [...parentA.meta.lineage, parentA.meta.name];
      } else {
          combinedLineage = [...parentA.meta.lineage, parentA.meta.name, parentB.meta.name];
      }
      
      newPopulation.push(child);
      newMetadata.push({
          name: childName,
          lineage: combinedLineage.slice(-3) // Keep history bounded
      });
    }

    this.population = newPopulation;
    this.metadata = newMetadata;
    this.generation++;

    return {
      bestFitness: this.bestFitness,
      avgFitness: this.avgFitness,
      worstFitness,
      bestGenomeIndex: rankedList[0].index,
      generation: this.generation
    };
  }

  tournamentSelect(rankedList, k = 4) {
    let best = null;
    for (let i = 0; i < k; i++) {
      const candidate = rankedList[Math.floor(Math.random() * rankedList.length)];
      if (!best || candidate.score > best.score) {
        best = candidate;
      }
    }
    return best; // Return entire object so we have genome AND meta
  }

  crossover(parentA, parentB) {
    const child = new Float32Array(this.genomeLength);
    for (let i = 0; i < this.genomeLength; i++) {
      child[i] = Math.random() < 0.5 ? parentA[i] : parentB[i];
    }
    return child;
  }

  mutate(genome) {
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < this.mutationRate) {
        if (Math.random() < 0.1) {
          // 10% chance: full reset to random value in [-0.5, 0.5]
          genome[i] = Math.random() - 0.5;
        } else {
          // 90% chance: add Gaussian noise with Box-Muller transform
          let u = 1 - Math.random();
          let v = Math.random();
          let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
          genome[i] += z * this.mutationScale;
        }
      }
    }
  }

  getPopulation() {
    return this.population;
  }

  getBestGenome() {
    // Note: this assumes the population is already sorted by nextGeneration
    return new Float32Array(this.population[0]);
  }
  
  resizePopulation(newSize) {
    if (newSize === this.popSize) return;
    
    if (newSize < this.popSize) {
        // Shrinking: keep the best ones (assuming currently sorted)
        this.population.length = newSize;
        this.metadata.length = newSize;
    } else {
        // Growing: add new random/mutated genomes to fill the gap
        const diff = newSize - this.popSize;
        for (let i = 0; i < diff; i++) {
            // We can either create purely random ones, or mutate the current best
            // Let's create random ones to keep diversity high
            this.population.push(this.randomGenome());
            this.metadata.push({
                name: generateUniqueName(),
                lineage: ['Spontaneous Generation']
            });
        }
    }
    
    this.popSize = newSize;
    this.elitismCount = Math.max(1, Math.floor(newSize * 0.1));
  }
  
  getBestMetadata() {
    return this.metadata[0];
  }
}
