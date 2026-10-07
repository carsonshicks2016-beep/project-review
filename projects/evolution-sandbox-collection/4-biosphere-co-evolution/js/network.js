/**
 * network.js
 * Lightweight custom neural network engine supporting:
 * - RabbitBrain: Feedforward controller for Genetic Algorithms.
 * - PPONetwork: Actor-Critic policy gradient network with a shared hidden layer and online Adam optimizer backpropagation.
 */

class RabbitBrain {
    constructor(numInputs = 4, numHidden = 8, numOutputs = 2) {
        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.numOutputs = numOutputs;

        // Weight matrices and bias vectors
        this.w1 = this.initWeights(numInputs, numHidden);
        this.b1 = new Array(numHidden).fill(0).map(() => (Math.random() - 0.5) * 0.1);
        this.w2 = this.initWeights(numHidden, numOutputs);
        this.b2 = new Array(numOutputs).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    }

    initWeights(rows, cols) {
        return new Array(rows).fill(0).map(() => 
            new Array(cols).fill(0).map(() => (Math.random() - 0.5) * 0.4)
        );
    }

    forward(inputs) {
        // Hidden layer with Tanh
        const hidden = new Array(this.numHidden);
        for (let j = 0; j < this.numHidden; j++) {
            let sum = this.b1[j];
            for (let i = 0; i < this.numInputs; i++) {
                sum += inputs[i] * this.w1[i][j];
            }
            hidden[j] = Math.tanh(sum);
        }

        // Output layer with Tanh (steering in [-1, 1], thrust in [-1, 1])
        const outputs = new Array(this.numOutputs);
        for (let j = 0; j < this.numOutputs; j++) {
            let sum = this.b2[j];
            for (let i = 0; i < this.numHidden; i++) {
                sum += hidden[i] * this.w2[i][j];
            }
            outputs[j] = Math.tanh(sum);
        }
        return outputs;
    }

    clone() {
        const copy = new RabbitBrain(this.numInputs, this.numHidden, this.numOutputs);
        copy.w1 = this.w1.map(row => [...row]);
        copy.b1 = [...this.b1];
        copy.w2 = this.w2.map(row => [...row]);
        copy.b2 = [...this.b2];
        return copy;
    }

    mutate(rate, strength = 0.2) {
        // Mutate w1
        for (let i = 0; i < this.numInputs; i++) {
            for (let j = 0; j < this.numHidden; j++) {
                if (Math.random() < rate) {
                    this.w1[i][j] += (Math.random() - 0.5) * strength;
                }
            }
        }
        // Mutate b1
        for (let j = 0; j < this.numHidden; j++) {
            if (Math.random() < rate) {
                this.b1[j] += (Math.random() - 0.5) * strength;
            }
        }
        // Mutate w2
        for (let i = 0; i < this.numHidden; i++) {
            for (let j = 0; j < this.numOutputs; j++) {
                if (Math.random() < rate) {
                    this.w2[i][j] += (Math.random() - 0.5) * strength;
                }
            }
        }
        // Mutate b2
        for (let j = 0; j < this.numOutputs; j++) {
            if (Math.random() < rate) {
                this.b2[j] += (Math.random() - 0.5) * strength;
            }
        }
    }

    crossover(partner) {
        const child = new RabbitBrain(this.numInputs, this.numHidden, this.numOutputs);
        
        // Single point or uniform crossover
        for (let i = 0; i < this.numInputs; i++) {
            for (let j = 0; j < this.numHidden; j++) {
                child.w1[i][j] = Math.random() < 0.5 ? this.w1[i][j] : partner.w1[i][j];
            }
        }
        for (let j = 0; j < this.numHidden; j++) {
            child.b1[j] = Math.random() < 0.5 ? this.b1[j] : partner.b1[j];
        }
        for (let i = 0; i < this.numHidden; i++) {
            for (let j = 0; j < this.numOutputs; j++) {
                child.w2[i][j] = Math.random() < 0.5 ? this.w2[i][j] : partner.w2[i][j];
            }
        }
        for (let j = 0; j < this.numOutputs; j++) {
            child.b2[j] = Math.random() < 0.5 ? this.b2[j] : partner.b2[j];
        }
        return child;
    }
}

class PPONetwork {
    constructor(numInputs = 10, numHidden = 16, numActorOutputs = 4) {
        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.numActorOutputs = numActorOutputs;

        // Initialize weights using Xavier style initialization
        this.wShared = this.initWeights(numInputs, numHidden);
        this.bShared = new Array(numHidden).fill(0);

        this.wActor = this.initWeights(numHidden, numActorOutputs);
        this.bActor = new Array(numActorOutputs).fill(0);

        this.wCritic = this.initWeights(numHidden, 1);
        this.bCritic = [0];

        // Adam optimizer parameters
        this.t = 0;
        this.m_wShared = this.zerosLike(this.wShared);
        this.v_wShared = this.zerosLike(this.wShared);
        this.m_bShared = this.zerosLike(this.bShared);
        this.v_bShared = this.zerosLike(this.bShared);

        this.m_wActor = this.zerosLike(this.wActor);
        this.v_wActor = this.zerosLike(this.wActor);
        this.m_bActor = this.zerosLike(this.bActor);
        this.v_bActor = this.zerosLike(this.bActor);

        this.m_wCritic = this.zerosLike(this.wCritic);
        this.v_wCritic = this.zerosLike(this.wCritic);
        this.m_bCritic = this.zerosLike(this.bCritic);
        this.v_bCritic = this.zerosLike(this.bCritic);
    }

    initWeights(rows, cols) {
        const limit = Math.sqrt(6 / (rows + cols));
        return new Array(rows).fill(0).map(() => 
            new Array(cols).fill(0).map(() => (Math.random() - 0.5) * 2 * limit)
        );
    }

    zerosLike(arr) {
        if (Array.isArray(arr[0])) {
            return arr.map(row => new Array(row.length).fill(0));
        }
        return new Array(arr.length).fill(0);
    }

    forward(state) {
        // Shared Hidden layer (Leaky ReLU)
        const hidden = new Array(this.numHidden);
        for (let j = 0; j < this.numHidden; j++) {
            let sum = this.bShared[j];
            for (let i = 0; i < this.numInputs; i++) {
                sum += state[i] * this.wShared[i][j];
            }
            hidden[j] = sum > 0 ? sum : 0.05 * sum;
        }

        // Actor Head logits (Linear before Softmax)
        const actorLogits = new Array(this.numActorOutputs);
        let maxLogit = -Infinity;
        for (let j = 0; j < this.numActorOutputs; j++) {
            let sum = this.bActor[j];
            for (let i = 0; i < this.numHidden; i++) {
                sum += hidden[i] * this.wActor[i][j];
            }
            actorLogits[j] = sum;
            if (sum > maxLogit) maxLogit = sum;
        }

        // Softmax conversion to probabilities
        const probs = new Array(this.numActorOutputs);
        let sumExp = 0;
        for (let j = 0; j < this.numActorOutputs; j++) {
            probs[j] = Math.exp(actorLogits[j] - maxLogit);
            sumExp += probs[j];
        }
        for (let j = 0; j < this.numActorOutputs; j++) {
            probs[j] = Math.max(1e-8, probs[j] / sumExp); // numerical stability clip
        }

        // Critic Head state value (Linear activation)
        let criticValue = this.bCritic[0];
        for (let i = 0; i < this.numHidden; i++) {
            criticValue += hidden[i] * this.wCritic[i][0];
        }

        return {
            hidden,
            probs,
            criticValue
        };
    }

    act(state, evaluate = false) {
        const { probs, criticValue } = this.forward(state);

        if (evaluate) {
            // Choose greedily
            let bestAction = 0;
            let bestProb = -1;
            for (let i = 0; i < probs.length; i++) {
                if (probs[i] > bestProb) {
                    bestProb = probs[i];
                    bestAction = i;
                }
            }
            return { action: bestAction, prob: bestProb, value: criticValue };
        }

        // Stochastic sampling
        const rand = Math.random();
        let cumulative = 0;
        let action = 0;
        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (rand <= cumulative) {
                action = i;
                break;
            }
        }
        return { action, prob: probs[action], value: criticValue };
    }

    trainStep(batch, lr = 0.0003, clipEps = 0.2, entropyCoeff = 0.01) {
        // Initialize gradient accumulation stores
        const d_wShared = this.zerosLike(this.wShared);
        const d_bShared = this.zerosLike(this.bShared);
        const d_wActor = this.zerosLike(this.wActor);
        const d_bActor = this.zerosLike(this.bActor);
        const d_wCritic = this.zerosLike(this.wCritic);
        const d_bCritic = this.zerosLike(this.bCritic);

        const batchSize = batch.length;
        if (batchSize === 0) return;

        // Process all samples in the trajectory batch
        for (const sample of batch) {
            const { state, action, oldProb, advantage, targetValue } = sample;

            // 1. Forward Pass
            const { hidden, probs, criticValue } = this.forward(state);

            // 2. Policy (Actor) Gradient with Clipping
            const ratio = probs[action] / oldProb;
            const unclipped = ratio * advantage;
            const clipped = Math.max(1 - clipEps, Math.min(1 + clipEps, ratio)) * advantage;

            const dActorLogit = new Array(this.numActorOutputs).fill(0);
            
            // We want to maximize the clipped surrogate objective, meaning we minimize its negative.
            // If the unclipped term is the active constraint in min(unclipped, clipped):
            if (unclipped <= clipped) {
                // Gradient of -unclipped w.r.t logits
                for (let i = 0; i < this.numActorOutputs; i++) {
                    const delta = (i === action) ? 1 : 0;
                    dActorLogit[i] = -ratio * advantage * (delta - probs[i]);
                }
            } else {
                // Clipped is active, so gradient w.r.t ratio (and thus logits) is zero.
                // dActorLogit remains 0.
            }

            // 3. Entropy Regularization Gradient
            // We want to maximize entropy, so we minimize -entropyCoeff * entropy.
            let entropy = 0;
            for (let i = 0; i < this.numActorOutputs; i++) {
                entropy -= probs[i] * Math.log(probs[i]);
            }
            for (let i = 0; i < this.numActorOutputs; i++) {
                const logP = Math.log(probs[i]);
                dActorLogit[i] += entropyCoeff * probs[i] * (logP + entropy);
            }

            // 4. Value (Critic) Gradient
            // Minimize MSE loss = 0.5 * (V - V_target)^2
            // Derivative w.r.t Critic Logit is simply (V - V_target)
            const dCritic = criticValue - targetValue;

            // 5. Accumulate Head Gradients
            for (let j = 0; j < this.numActorOutputs; j++) {
                d_bActor[j] += dActorLogit[j] / batchSize;
                for (let i = 0; i < this.numHidden; i++) {
                    d_wActor[i][j] += (dActorLogit[j] * hidden[i]) / batchSize;
                }
            }

            d_bCritic[0] += dCritic / batchSize;
            for (let i = 0; i < this.numHidden; i++) {
                d_wCritic[i][0] += (dCritic * hidden[i]) / batchSize;
            }

            // 6. Backpropagate to Shared Hidden Layer
            const dHidden = new Array(this.numHidden).fill(0);
            for (let i = 0; i < this.numHidden; i++) {
                let sum = dCritic * this.wCritic[i][0];
                for (let j = 0; j < this.numActorOutputs; j++) {
                    sum += dActorLogit[j] * this.wActor[i][j];
                }
                dHidden[i] = sum;
            }

            // Backprop through Leaky ReLU activation
            const dHiddenInput = new Array(this.numHidden);
            for (let j = 0; j < this.numHidden; j++) {
                dHiddenInput[j] = dHidden[j] * (hidden[j] > 0 ? 1.0 : 0.05);
            }

            // Accumulate Shared Layer Gradients
            for (let j = 0; j < this.numHidden; j++) {
                d_bShared[j] += dHiddenInput[j] / batchSize;
                for (let i = 0; i < this.numInputs; i++) {
                    d_wShared[i][j] += (dHiddenInput[j] * state[i]) / batchSize;
                }
            }
        }

        // 7. Update Weights and Biases using Adam Optimizer
        this.t++;
        const beta1 = 0.9;
        const beta2 = 0.999;
        const eps = 1e-8;

        this.applyAdam(this.wShared, d_wShared, this.m_wShared, this.v_wShared, lr, beta1, beta2, eps);
        this.applyAdam(this.bShared, d_bShared, this.m_bShared, this.v_bShared, lr, beta1, beta2, eps);
        this.applyAdam(this.wActor, d_wActor, this.m_wActor, this.v_wActor, lr, beta1, beta2, eps);
        this.applyAdam(this.bActor, d_bActor, this.m_bActor, this.v_bActor, lr, beta1, beta2, eps);
        this.applyAdam(this.wCritic, d_wCritic, this.m_wCritic, this.v_wCritic, lr, beta1, beta2, eps);
        this.applyAdam(this.bCritic, d_bCritic, this.m_bCritic, this.v_bCritic, lr, beta1, beta2, eps);
    }

    applyAdam(param, grad, m, v, lr, beta1, beta2, eps) {
        for (let i = 0; i < param.length; i++) {
            if (Array.isArray(param[i])) {
                // Recursive matrix application
                this.applyAdam(param[i], grad[i], m[i], v[i], lr, beta1, beta2, eps);
            } else {
                const g = grad[i];
                // Accumulate moments
                m[i] = beta1 * m[i] + (1 - beta1) * g;
                v[i] = beta2 * v[i] + (1 - beta2) * g * g;

                // Bias corrections
                const mCorr = m[i] / (1 - Math.pow(beta1, this.t));
                const vCorr = v[i] / (1 - Math.pow(beta2, this.t));

                // Step update
                param[i] -= (lr * mCorr) / (Math.sqrt(vCorr) + eps);
            }
        }
    }
}
