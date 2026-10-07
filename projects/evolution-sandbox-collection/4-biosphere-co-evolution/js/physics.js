/**
 * physics.js
 * Continuous 2D Physics Kinematics & Sensory Vision for the Biosphere Ecosystem.
 */

class Carrot {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.radius = 6;
        this.energyValue = 80;
        this.isDead = false;
    }

    draw(ctx) {
        ctx.save();
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#10B981'; // Neon Green
        ctx.fillStyle = '#10B981';
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
}

class Agent {
    constructor(x, y, radius, angle, maxSpeed, color) {
        this.x = x;
        this.y = y;
        this.vx = 0;
        this.vy = 0;
        this.radius = radius;
        this.angle = angle; // Heading angle in radians
        this.maxSpeed = maxSpeed;
        this.color = color;
        this.drag = 0.92; // Friction factor
        this.energy = 150;
        this.maxEnergy = 300;
        this.isDead = false;
        this.id = Math.random().toString(36).substr(2, 9);
    }

    updatePhysics(width, height) {
        // Apply friction
        this.vx *= this.drag;
        this.vy *= this.drag;

        // Apply velocity to position
        this.x += this.vx;
        this.y += this.vy;

        // Keep angle in [-PI, PI]
        while (this.angle > Math.PI) this.angle -= 2 * Math.PI;
        while (this.angle < -Math.PI) this.angle += 2 * Math.PI;

        // Wall collision deflection
        return this.handleWallCollisions(width, height);
    }

    handleWallCollisions(width, height) {
        let wallHit = false;
        if (this.x < this.radius) {
            this.x = this.radius;
            this.vx = -this.vx * 0.5;
            wallHit = true;
        } else if (this.x > width - this.radius) {
            this.x = width - this.radius;
            this.vx = -this.vx * 0.5;
            wallHit = true;
        }

        if (this.y < this.radius) {
            this.y = this.radius;
            this.vy = -this.vy * 0.5;
            wallHit = true;
        } else if (this.y > height - this.radius) {
            this.y = height - this.radius;
            this.vy = -this.vy * 0.5;
            wallHit = true;
        }
        return wallHit;
    }

    steer(turnSignal) {
        // turnSignal is in range [-1, 1]
        const maxTurnSpeed = 0.12; // Radians per step
        this.angle += turnSignal * maxTurnSpeed;
    }

    accelerate(thrustSignal) {
        // thrustSignal is in range [0, 1] or [-1, 1] (but we clamp to [0, 1] for acceleration)
        const thrust = Math.max(0, thrustSignal);
        const accelerationRate = 0.6;
        this.vx += Math.cos(this.angle) * thrust * accelerationRate;
        this.vy += Math.sin(this.angle) * thrust * accelerationRate;

        // Speed limit check
        const speed = Math.hypot(this.vx, this.vy);
        if (speed > this.maxSpeed) {
            this.vx = (this.vx / speed) * this.maxSpeed;
            this.vy = (this.vy / speed) * this.maxSpeed;
        }
    }

    findNearestEntity(entities, visionRadius, visionAngle) {
        let nearest = null;
        let minDist = Infinity;
        let minRelAngle = 0;

        for (const entity of entities) {
            if (entity === this || entity.isDead) continue;

            const dx = entity.x - this.x;
            const dy = entity.y - this.y;
            const dist = Math.hypot(dx, dy);

            if (dist > visionRadius) continue;

            const angleToTarget = Math.atan2(dy, dx);
            let relAngle = angleToTarget - this.angle;

            while (relAngle > Math.PI) relAngle -= 2 * Math.PI;
            while (relAngle < -Math.PI) relAngle += 2 * Math.PI;

            if (Math.abs(relAngle) <= visionAngle / 2) {
                if (dist < minDist) {
                    minDist = dist;
                    minRelAngle = relAngle;
                    nearest = entity;
                }
            }
        }

        return { entity: nearest, dist: minDist, relAngle: minRelAngle };
    }

    getDistanceToNearestWall(width, height) {
        const distLeft = this.x;
        const distRight = width - this.x;
        const distTop = this.y;
        const distBottom = height - this.y;
        return Math.min(distLeft, distRight, distTop, distBottom);
    }
}

class Rabbit extends Agent {
    constructor(x, y, genes = null, generation = 1) {
        // Rabbit physical DNA definitions
        const baseSpeed = 3.2;
        const baseVisionRadius = 140;
        const baseRadius = 8;

        // Genes mapping
        let speedGene = 1.0;
        let visionGene = 1.0;
        let sizeGene = 1.0;

        if (genes) {
            speedGene = genes.speedGene;
            visionGene = genes.visionGene;
            sizeGene = genes.sizeGene;
        }

        const radius = baseRadius * sizeGene;
        const maxSpeed = baseSpeed * speedGene;

        super(x, y, radius, Math.random() * Math.PI * 2, maxSpeed, '#38BDF8'); // Sky Blue

        this.genes = genes || {
            speedGene: 0.8 + Math.random() * 0.4, // [0.8, 1.2]
            visionGene: 0.8 + Math.random() * 0.4, // [0.8, 1.2]
            sizeGene: 0.8 + Math.random() * 0.4, // [0.8, 1.2]
            // Brain weights will be attached in ga.js
            brainWeights: null
        };

        this.generation = generation;
        this.visionRadius = baseVisionRadius * this.genes.visionGene;
        this.visionAngle = Math.PI * (2 / 3); // 120 degrees
        
        // Dynamic stats
        this.energy = 150;
        this.maxEnergy = 300;
        
        // Evolutionary balancing cost: Larger size, higher speed and wider vision drain energy faster.
        this.energyCostPerStep = 0.15 * (
            Math.pow(this.genes.speedGene, 2) * 
            Math.pow(this.genes.sizeGene, 3) + 
            (this.genes.visionGene - 1.0) * 0.2
        );
        this.energyCostPerStep = Math.max(0.05, this.energyCostPerStep);
    }

    update(carrots, foxes, width, height) {
        // Drain energy
        this.energy -= this.energyCostPerStep;
        if (this.energy <= 0) {
            this.isDead = true;
            return;
        }

        // Get brain sensors
        const carrotSensor = this.findNearestEntity(carrots, this.visionRadius, this.visionAngle);
        const foxSensor = this.findNearestEntity(foxes, this.visionRadius, this.visionAngle);

        // Normalize sensors: if nothing found, we feed 1.0 for distance, 0.0 for angle
        const carrotDistNorm = carrotSensor.entity ? carrotSensor.dist / this.visionRadius : 1.0;
        const carrotAngleNorm = carrotSensor.entity ? carrotSensor.relAngle / (this.visionAngle / 2) : 0.0;
        const foxDistNorm = foxSensor.entity ? foxSensor.dist / this.visionRadius : 1.0;
        const foxAngleNorm = foxSensor.entity ? foxSensor.relAngle / (this.visionAngle / 2) : 0.0;

        // Neural net mapping
        const inputs = [carrotDistNorm, carrotAngleNorm, foxDistNorm, foxAngleNorm];
        let outputs = [0, 0]; // [steer, thrust]

        if (this.brain) {
            outputs = this.brain.forward(inputs);
        } else {
            // Default random behavior if no brain
            outputs = [(Math.random() - 0.5) * 2, Math.random()];
        }

        // Apply brain decisions
        this.steer(outputs[0]);
        this.accelerate(outputs[1]);

        // Physics update
        const wallHit = this.updatePhysics(width, height);
        if (wallHit) {
            this.energy -= 1.0; // Wall bump energy penalty
        }

        // Eat carrot check
        for (const carrot of carrots) {
            if (carrot.isDead) continue;
            const dist = Math.hypot(carrot.x - this.x, carrot.y - this.y);
            if (dist < this.radius + carrot.radius) {
                carrot.isDead = true;
                this.energy = Math.min(this.maxEnergy, this.energy + carrot.energyValue);
            }
        }
    }

    draw(ctx, showVision, highlightTarget) {
        ctx.save();

        // 1. Vision Cone
        if (showVision) {
            ctx.fillStyle = 'rgba(56, 189, 248, 0.03)';
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.1)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.arc(
                this.x, this.y, 
                this.visionRadius, 
                this.angle - this.visionAngle / 2, 
                this.angle + this.visionAngle / 2
            );
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }

        // 2. Draw Rabbit Body (Facing angle)
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);

        // Body Shadow Glow
        ctx.shadowBlur = 8;
        ctx.shadowColor = this.color;

        ctx.fillStyle = this.color;
        ctx.beginPath();
        // Triangle shape facing forward
        ctx.moveTo(this.radius * 1.5, 0);
        ctx.lineTo(-this.radius, -this.radius * 0.8);
        ctx.lineTo(-this.radius, this.radius * 0.8);
        ctx.closePath();
        ctx.fill();

        // Ears
        ctx.fillStyle = '#E0F2FE';
        ctx.beginPath();
        ctx.ellipse(-this.radius * 0.5, -this.radius * 0.6, this.radius * 0.6, this.radius * 0.25, Math.PI / 4, 0, Math.PI * 2);
        ctx.ellipse(-this.radius * 0.5, this.radius * 0.6, this.radius * 0.6, this.radius * 0.25, -Math.PI / 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }
}

class Fox extends Agent {
    constructor(x, y) {
        const radius = 13;
        const maxSpeed = 3.6;
        super(x, y, radius, Math.random() * Math.PI * 2, maxSpeed, '#F97316'); // Neon Orange

        this.visionRadius = 180;
        this.visionAngle = Math.PI * (3 / 4); // 135 degrees
        this.maxEnergy = 400;
        this.energy = 250;

        // Base step costs
        this.energyCostPerStep = 0.18;
        this.targetLock = null; // Reference to rabbit currently locked
        
        // PPO Stats
        this.rabbitsCaught = 0;
        this.wallCrashes = 0;
        
        // Inputs & outputs placeholders for training/rendering
        this.lastState = null;
        this.lastAction = null;
        this.lastActionProb = null;
        this.lastValue = null;
    }

    update(rabbits, otherFoxes, width, height) {
        // Drain energy
        this.energy -= this.energyCostPerStep;
        if (this.energy <= 0) {
            this.isDead = true;
            return;
        }

        // Find sensors
        const rabbitSensor = this.findNearestEntity(rabbits, this.visionRadius, this.visionAngle);
        const foxSensor = this.findNearestEntity(otherFoxes, this.visionRadius, this.visionAngle);
        const wallDist = this.getDistanceToNearestWall(width, height);

        // Keep track of locked target
        this.targetLock = rabbitSensor.entity;

        // Fox State Vector (10 inputs)
        // 1. vx (norm)
        // 2. vy (norm)
        // 3. angle (cos)
        // 4. angle (sin)
        // 5. rabbit_rel_x (norm)
        // 6. rabbit_rel_y (norm)
        // 7. wall_dist (norm)
        // 8. energy_level (norm)
        // 9. other_fox_dist (norm)
        // 10. other_fox_angle (norm)
        
        const vxNorm = this.vx / this.maxSpeed;
        const vyNorm = this.vy / this.maxSpeed;
        const cosAngle = Math.cos(this.angle);
        const sinAngle = Math.sin(this.angle);

        let rxRel = 0;
        let ryRel = 0;
        if (rabbitSensor.entity) {
            // Rel position relative to fox's orientation
            const dx = rabbitSensor.entity.x - this.x;
            const dy = rabbitSensor.entity.y - this.y;
            // Rotate by -this.angle to make it frame-of-reference local
            rxRel = (dx * cosAngle + dy * sinAngle) / this.visionRadius;
            ryRel = (-dx * sinAngle + dy * cosAngle) / this.visionRadius;
        }

        const wallDistNorm = Math.min(1.0, wallDist / this.visionRadius);
        const energyNorm = this.energy / this.maxEnergy;

        const otherFoxDistNorm = foxSensor.entity ? foxSensor.dist / this.visionRadius : 1.0;
        const otherFoxAngleNorm = foxSensor.entity ? foxSensor.relAngle / (this.visionAngle / 2) : 0.0;

        const state = [
            vxNorm, vyNorm, cosAngle, sinAngle,
            rxRel, ryRel, wallDistNorm, energyNorm,
            otherFoxDistNorm, otherFoxAngleNorm
        ];

        this.lastState = state;

        // Brain decision made via PPO engine externally
        // Actions: 0 = Turn Left, 1 = Turn Right, 2 = Accelerate, 3 = Coast
        let action = 3;
        if (this.brain) {
            // Select action using PPO policy (stochastic during training, deterministic in inference)
            const result = this.brain.act(state);
            action = result.action;
            this.lastAction = action;
            this.lastActionProb = result.prob;
            this.lastValue = result.value;
        } else {
            action = Math.floor(Math.random() * 4);
        }

        // Apply Action
        if (action === 0) {
            this.steer(-1.0);
        } else if (action === 1) {
            this.steer(1.0);
        } else if (action === 2) {
            this.accelerate(1.0);
        } else if (action === 3) {
            // Coast, do nothing (drag slows down)
        }

        // Physics update
        const wallHit = this.updatePhysics(width, height);
        if (wallHit) {
            this.wallCrashes++;
            this.energy -= 4.0; // PPO fox gets heavier wall penalty
        }

        // Eat rabbit check
        let caughtRabbit = null;
        for (const rabbit of rabbits) {
            if (rabbit.isDead) continue;
            const dist = Math.hypot(rabbit.x - this.x, rabbit.y - this.y);
            if (dist < this.radius + rabbit.radius) {
                rabbit.isDead = true;
                this.energy = Math.min(this.maxEnergy, this.energy + 120);
                this.rabbitsCaught++;
                caughtRabbit = rabbit;
                break; // Eaten only one rabbit per tick
            }
        }

        return {
            state: state,
            action: action,
            caughtRabbit: caughtRabbit,
            wallHit: wallHit
        };
    }

    draw(ctx, showVision, highlightTarget) {
        ctx.save();

        // 1. Vision Cone
        if (showVision) {
            ctx.fillStyle = 'rgba(249, 115, 22, 0.02)';
            ctx.strokeStyle = 'rgba(249, 115, 22, 0.08)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.arc(
                this.x, this.y, 
                this.visionRadius, 
                this.angle - this.visionAngle / 2, 
                this.angle + this.visionAngle / 2
            );
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }

        // 2. Highlight Target Lock
        if (highlightTarget && this.targetLock && !this.targetLock.isDead) {
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)'; // Red target line
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(this.x, this.y);
            ctx.lineTo(this.targetLock.x, this.targetLock.y);
            ctx.stroke();
            ctx.setLineDash([]); // Reset dash

            // Draw a tiny target lock circle on the rabbit
            ctx.strokeStyle = '#EF4444';
            ctx.beginPath();
            ctx.arc(this.targetLock.x, this.targetLock.y, this.targetLock.radius + 4, 0, Math.PI * 2);
            ctx.stroke();
        }

        // 3. Draw Fox Body
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);

        // Body Glow
        ctx.shadowBlur = 10;
        ctx.shadowColor = this.color;

        ctx.fillStyle = this.color;
        ctx.beginPath();
        // Fox shape: elongated arrow triangle
        ctx.moveTo(this.radius * 1.8, 0);
        ctx.lineTo(-this.radius, -this.radius * 0.7);
        ctx.lineTo(-this.radius * 0.6, 0);
        ctx.lineTo(-this.radius, this.radius * 0.7);
        ctx.closePath();
        ctx.fill();

        // Big Ears
        ctx.fillStyle = '#FFEDD5';
        ctx.beginPath();
        ctx.moveTo(-this.radius * 0.2, -this.radius * 0.5);
        ctx.lineTo(-this.radius * 0.8, -this.radius * 1.2);
        ctx.lineTo(-this.radius * 0.8, -this.radius * 0.4);
        ctx.closePath();
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(-this.radius * 0.2, this.radius * 0.5);
        ctx.lineTo(-this.radius * 0.8, this.radius * 1.2);
        ctx.lineTo(-this.radius * 0.8, this.radius * 0.4);
        ctx.closePath();
        ctx.fill();

        // White Tip Tail
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(-this.radius * 1.1, 0, this.radius * 0.35, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }
}
