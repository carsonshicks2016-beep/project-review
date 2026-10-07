/**
 * Darwin's Doodles - Genetic Algorithm Engine
 */

class GeneticAlgorithm {
    constructor() {
        // Range mappings for genome parameters (tuned for realistic biology/walking)
        this.ranges = {
            amplitude: { min: 0.0, max: 0.22 },   // max 22% contraction (prevents hyper-stretching)
            frequency: { min: 0.4, max: 1.8 },   // max 1.8Hz (smoother, coordinated cycles)
            phase: { min: 0.0, max: Math.PI * 2 },
            lengthMult: { min: 0.85, max: 1.15 }  // keeps bones close to default proportions (+/- 15%)
        };
    }

    /**
     * Creates a random genome: array of floats in [0, 1]
     */
    createRandomGenome(length) {
        const genome = [];
        for (let i = 0; i < length; i++) {
            genome.push(Math.random());
        }
        return genome;
    }

    /**
     * Map a genome array to a creature's muscle parameters
     */
    applyGenome(creature, genome) {
        creature.dna = [...genome];
        const muscles = creature.constraints.filter(c => c.isMuscle);
        
        for (let i = 0; i < muscles.length; i++) {
            const muscle = muscles[i];
            const geneOffset = i * 4;

            if (geneOffset + 3 >= genome.length) break;

            const gAmp = genome[geneOffset];
            const gFreq = genome[geneOffset + 1];
            const gPhase = genome[geneOffset + 2];
            const gLen = genome[geneOffset + 3];

            // Map [0, 1] genes to physical domain values
            muscle.amplitude = this.ranges.amplitude.min + gAmp * (this.ranges.amplitude.max - this.ranges.amplitude.min);
            muscle.frequency = this.ranges.frequency.min + gFreq * (this.ranges.frequency.max - this.ranges.frequency.min);
            muscle.phase = this.ranges.phase.min + gPhase * (this.ranges.phase.max - this.ranges.phase.min);
            
            // Adjust length of physical muscle constraint
            const lenMult = this.ranges.lengthMult.min + gLen * (this.ranges.lengthMult.max - this.ranges.lengthMult.min);
            muscle.length = muscle.baseLength * lenMult;
        }
    }

    /**
     * Gets the number of genes needed for a specific creature preset
     */
    getGenomeLengthForCreature(creature) {
        const muscles = creature.constraints.filter(c => c.isMuscle);
        return muscles.length * 4; // 4 genes per muscle
    }

    /**
     * Selection: Tournament Selection
     */
    tournamentSelect(population, tournamentSize = 3) {
        let best = null;
        for (let i = 0; i < tournamentSize; i++) {
            const ind = population[Math.floor(Math.random() * population.length)];
            if (best === null || ind.fitness > best.fitness) {
                best = ind;
            }
        }
        return best;
    }

    /**
     * Selection: Roulette Wheel Selection
     */
    rouletteSelect(population) {
        // Shift fitness values to be strictly positive to avoid negative probability
        let minFitness = Infinity;
        for (const ind of population) {
            if (ind.fitness < minFitness) minFitness = ind.fitness;
        }

        const shift = minFitness < 0 ? -minFitness + 1 : 0.01;
        
        let sum = 0;
        for (const ind of population) {
            sum += (ind.fitness + shift);
        }

        let pick = Math.random() * sum;
        let currentSum = 0;
        
        for (const ind of population) {
            currentSum += (ind.fitness + shift);
            if (currentSum >= pick) {
                return ind;
            }
        }
        return population[population.length - 1];
    }

    /**
     * Crossover: Uniform Crossover
     */
    crossover(parentA, parentB) {
        const child = [];
        for (let i = 0; i < parentA.length; i++) {
            // 50% chance from parent A, 50% from parent B
            if (Math.random() < 0.5) {
                child.push(parentA[i]);
            } else {
                child.push(parentB[i]);
            }
        }
        return child;
    }

    /**
     * Mutation: Hybrid Gaussian and Random Reset
     */
    mutate(genome, rate) {
        const mutated = [];
        for (let i = 0; i < genome.length; i++) {
            if (Math.random() < rate) {
                // 15% chance of complete random reset (helps explore new valleys)
                if (Math.random() < 0.15) {
                    mutated.push(Math.random());
                } else {
                    // 85% chance of Gaussian perturbation
                    const factor = 0.12; // std deviation equivalent
                    const u1 = Math.random() || 0.001; // Avoid 0
                    const u2 = Math.random();
                    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2); // Box-Muller transform
                    let newGene = genome[i] + z0 * factor;
                    
                    // Clamp to [0, 1] range
                    newGene = Math.max(0, Math.min(1, newGene));
                    mutated.push(newGene);
                }
            } else {
                mutated.push(genome[i]);
            }
        }
        return mutated;
    }

    /**
     * Evolve a population of creatures into the next generation
     */
    evolveGeneration(population, numGenes, mutationRate, elitismCount, selectionMethod, presetType, startX, startY) {
        // 1. Sort population by fitness in descending order (highest fitness first)
        const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
        const nextGenGenomes = [];

        // 2. Carry over elites directly
        for (let i = 0; i < Math.min(elitismCount, sorted.length); i++) {
            nextGenGenomes.push([...sorted[i].dna]);
        }

        // 3. Fill the rest of the population with crossover and mutation
        const offspringCount = population.length - nextGenGenomes.length;
        for (let i = 0; i < offspringCount; i++) {
            // Select parents
            let parentA, parentB;
            if (selectionMethod === 'roulette') {
                parentA = this.rouletteSelect(sorted);
                parentB = this.rouletteSelect(sorted);
            } else {
                parentA = this.tournamentSelect(sorted, 3);
                parentB = this.tournamentSelect(sorted, 3);
            }

            // Perform crossover
            let childGenome = this.crossover(parentA.dna, parentB.dna);

            // Perform mutation
            childGenome = this.mutate(childGenome, mutationRate);

            nextGenGenomes.push(childGenome);
        }

        // 4. Rebuild population creatures with the new genomes
        const newPopulation = [];
        for (let i = 0; i < nextGenGenomes.length; i++) {
            const creature = CreaturePresets.get(presetType, i, startX, startY);
            this.applyGenome(creature, nextGenGenomes[i]);
            newPopulation.push(creature);
        }

        return newPopulation;
    }
}
