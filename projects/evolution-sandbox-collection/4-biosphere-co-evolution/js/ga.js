/**
 * ga.js
 * Genetic Algorithm helper functions for managing Rabbit populations.
 */

const RabbitGA = {
    clamp: function(val, min, max) {
        return Math.max(min, Math.min(max, val));
    },

    createRandomRabbit: function(x, y) {
        const rabbit = new Rabbit(x, y);
        rabbit.brain = new RabbitBrain();
        return rabbit;
    },

    /**
     * Asexual reproduction: clones the rabbit and mutates its physical and brain traits.
     * Energy is split between the parent and child.
     */
    reproduceAsexually: function(parent, mutationRate) {
        const mutationStrength = 0.12;

        // Mutate physical genes
        const speedGene = this.clamp(
            parent.genes.speedGene + (Math.random() - 0.5) * mutationStrength,
            0.5,
            2.0
        );
        const visionGene = this.clamp(
            parent.genes.visionGene + (Math.random() - 0.5) * mutationStrength,
            0.5,
            2.0
        );
        const sizeGene = this.clamp(
            parent.genes.sizeGene + (Math.random() - 0.5) * mutationStrength,
            0.5,
            2.0
        );

        const childGenes = {
            speedGene,
            visionGene,
            sizeGene
        };

        // Spawn child near parent
        const spawnX = this.clamp(parent.x + (Math.random() - 0.5) * 20, 20, 780);
        const spawnY = this.clamp(parent.y + (Math.random() - 0.5) * 20, 20, 580);

        const child = new Rabbit(spawnX, spawnY, childGenes, parent.generation + 1);
        
        // Clone and mutate neural brain weights
        if (parent.brain) {
            child.brain = parent.brain.clone();
            child.brain.mutate(mutationRate, 0.15); // Brain mutation strength
        } else {
            child.brain = new RabbitBrain();
        }

        // Split parent energy with offspring
        const energySplit = parent.energy * 0.45;
        parent.energy -= energySplit;
        child.energy = energySplit;

        return child;
    },

    /**
     * Sexual reproduction: crosses over physical genes and brain weights from two parents.
     * Both parents contribute to the offspring's initial energy.
     */
    reproduceSexually: function(parentA, parentB, mutationRate) {
        const mutationStrength = 0.08;

        const crossoverGene = (geneA, geneB) => {
            const mix = Math.random();
            if (mix < 0.4) return geneA;
            if (mix < 0.8) return geneB;
            return (geneA + geneB) / 2; // Hybrid blend
        };

        // Crossover + small mutation
        const speedGene = this.clamp(
            crossoverGene(parentA.genes.speedGene, parentB.genes.speedGene) + (Math.random() - 0.5) * mutationStrength,
            0.5,
            2.0
        );
        const visionGene = this.clamp(
            crossoverGene(parentA.genes.visionGene, parentB.genes.visionGene) + (Math.random() - 0.5) * mutationStrength,
            0.5,
            2.0
        );
        const sizeGene = this.clamp(
            crossoverGene(parentA.genes.sizeGene, parentB.genes.sizeGene) + (Math.random() - 0.5) * mutationStrength,
            0.5,
            2.0
        );

        const childGenes = {
            speedGene,
            visionGene,
            sizeGene
        };

        // Spawn child in the middle of parents
        const spawnX = this.clamp((parentA.x + parentB.x) / 2 + (Math.random() - 0.5) * 15, 20, 780);
        const spawnY = this.clamp((parentA.y + parentB.y) / 2 + (Math.random() - 0.5) * 15, 20, 580);

        const childGen = Math.max(parentA.generation, parentB.generation) + 1;
        const child = new Rabbit(spawnX, spawnY, childGenes, childGen);

        // Neural network crossover and mutation
        if (parentA.brain && parentB.brain) {
            child.brain = parentA.brain.crossover(parentB.brain);
            child.brain.mutate(mutationRate, 0.15);
        } else {
            child.brain = new RabbitBrain();
        }

        // Energy split from both parents
        const energyFromA = parentA.energy * 0.3;
        const energyFromB = parentB.energy * 0.3;
        parentA.energy -= energyFromA;
        parentB.energy -= energyFromB;
        child.energy = energyFromA + energyFromB;

        return child;
    }
};
