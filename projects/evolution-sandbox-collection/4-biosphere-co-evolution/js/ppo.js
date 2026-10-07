/**
 * ppo.js
 * PPO Reinforcement Learning manager for Fox agents.
 */

class PPOLearner {
    constructor(numInputs = 10, numHidden = 16, numOutputs = 4) {
        this.globalBrain = new PPONetwork(numInputs, numHidden, numOutputs);
        this.globalMemory = [];
        this.activeTrajectories = {}; // Maps foxId -> array of steps
        this.updateInterval = 600; // Update neural brain when memory hits 600 steps
    }

    /**
     * Record a single time step transition for a fox.
     * If the step is terminal (done = true), finalize trajectory and push to memory.
     */
    recordStep(foxId, state, action, prob, value, reward, done) {
        if (!this.activeTrajectories[foxId]) {
            this.activeTrajectories[foxId] = [];
        }

        this.activeTrajectories[foxId].push({
            state: [...state],
            action: action,
            oldProb: prob,
            value: value,
            reward: reward,
            done: done
        });

        if (done) {
            const trajectory = this.activeTrajectories[foxId];
            if (trajectory && trajectory.length > 0) {
                this.calculateGAE(trajectory, 0.98, 0.95, 0.0);
                this.globalMemory.push(...trajectory);
            }
            delete this.activeTrajectories[foxId];
        }
    }

    /**
     * Boostrap remaining steps for currently alive foxes before training.
     */
    bootstrapActive(activeFoxes) {
        for (const fox of activeFoxes) {
            const trajectory = this.activeTrajectories[fox.id];
            if (trajectory && trajectory.length > 0) {
                const lastVal = fox.lastValue || 0;
                this.calculateGAE(trajectory, 0.98, 0.95, lastVal);
                this.globalMemory.push(...trajectory);
                // Clear trajectory buffer so it starts fresh in next episode loop
                this.activeTrajectories[fox.id] = [];
            }
        }
    }

    /**
     * Calculate Generalized Advantage Estimator (GAE) and target value returns.
     */
    calculateGAE(trajectory, gamma, lambda, lastValue) {
        const T = trajectory.length;
        if (T === 0) return;

        const advantages = new Array(T);
        const targetValues = new Array(T);

        let nextValue = lastValue;
        let gae = 0;

        for (let t = T - 1; t >= 0; t--) {
            const step = trajectory[t];
            const nonTerminal = step.done ? 0 : 1;
            
            // Temporal Difference error
            const delta = step.reward + gamma * nextValue * nonTerminal - step.value;
            
            // GAE accumulator
            gae = delta + gamma * lambda * nonTerminal * gae;
            advantages[t] = gae;
            targetValues[t] = gae + step.value;

            nextValue = step.value;
        }

        // Write outputs back into the trajectory steps
        for (let t = 0; t < T; t++) {
            trajectory[t].advantage = advantages[t];
            trajectory[t].targetValue = targetValues[t];
        }
    }

    /**
     * Execute the PPO training algorithm epochs using accumulated memory.
     * Returns true if optimization was performed.
     */
    train(lr = 0.0003, entropyCoeff = 0.01) {
        if (this.globalMemory.length < 128) {
            return false; // Not enough samples
        }

        // Normalize advantages to stabilize policy updates
        this.normalizeAdvantages(this.globalMemory);

        const epochs = 4;
        const miniBatchSize = 64;
        const clipEps = 0.2;

        for (let e = 0; e < epochs; e++) {
            this.shuffle(this.globalMemory);

            for (let i = 0; i < this.globalMemory.length; i += miniBatchSize) {
                const batch = this.globalMemory.slice(i, i + miniBatchSize);
                this.globalBrain.trainStep(batch, lr, clipEps, entropyCoeff);
            }
        }

        // Flush training buffer
        this.globalMemory = [];
        return true;
    }

    normalizeAdvantages(memory) {
        let sum = 0;
        for (const step of memory) {
            sum += step.advantage;
        }
        const mean = sum / memory.length;

        let sqSum = 0;
        for (const step of memory) {
            sqSum += Math.pow(step.advantage - mean, 2);
        }
        const std = Math.sqrt(sqSum / memory.length) + 1e-8;

        for (const step of memory) {
            step.advantage = (step.advantage - mean) / std;
        }
    }

    shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            const temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    clear() {
        this.globalMemory = [];
        this.activeTrajectories = {};
    }
}
