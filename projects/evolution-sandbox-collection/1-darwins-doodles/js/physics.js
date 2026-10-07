/**
 * Darwin's Doodles - Verlet Integration Physics Engine
 */

class Node {
    constructor(x, y, radius = 8, mass = 1.0) {
        this.x = x;
        this.y = y;
        this.oldX = x;
        this.oldY = y;
        this.radius = radius;
        this.mass = mass;
        this.onGround = false;
        this.originalX = x; // useful for relative offsets
        this.originalY = y;
    }

    update(gravity, damping, dt) {
        // Compute velocity from Verlet history
        const vx = (this.x - this.oldX) * damping;
        const vy = (this.y - this.oldY) * damping;

        this.oldX = this.x;
        this.oldY = this.y;

        // Apply forces
        this.x += vx;
        this.y += vy + gravity * dt * dt;
    }

    reset(offsetX = 0, offsetY = 0) {
        this.x = this.originalX + offsetX;
        this.y = this.originalY + offsetY;
        this.oldX = this.x;
        this.oldY = this.y;
        this.onGround = false;
    }
}

class Bone {
    constructor(nodeA, nodeB, stiffness = 0.85) {
        this.nodeA = nodeA;
        this.nodeB = nodeB;
        this.stiffness = stiffness;
        
        // Calculate rest length
        const dx = nodeB.x - nodeA.x;
        const dy = nodeB.y - nodeA.y;
        this.length = Math.sqrt(dx * dx + dy * dy);
        this.isMuscle = false;
    }

    resolve() {
        const dx = this.nodeB.x - this.nodeA.x;
        const dy = this.nodeB.y - this.nodeA.y;
        const currentLength = Math.sqrt(dx * dx + dy * dy);

        if (currentLength === 0) return;

        const diff = this.length - currentLength;
        const percent = (diff / currentLength) * this.stiffness * 0.5;
        const offsetX = dx * percent;
        const offsetY = dy * percent;

        // Share displacement inversely proportional to mass
        const massSum = this.nodeA.mass + this.nodeB.mass;
        const factorA = this.nodeB.mass / massSum;
        const factorB = this.nodeA.mass / massSum;

        this.nodeA.x -= offsetX * factorA;
        this.nodeA.y -= offsetY * factorA;
        this.nodeB.x += offsetX * factorB;
        this.nodeB.y += offsetY * factorB;
    }
}

class Muscle {
    constructor(attachmentA, attachmentB, ratioA = 0, ratioB = 0, stiffness = 0.8) {
        this.attachmentA = attachmentA; // Node or Bone
        this.ratioA = ratioA;           // ratio along attachmentA if Bone
        this.attachmentB = attachmentB; // Node or Bone
        this.ratioB = ratioB;           // ratio along attachmentB if Bone
        this.stiffness = stiffness;
        
        this.isMuscle = true;
        this.amplitude = 0.0; // range [0, 0.22]
        this.frequency = 1.0; // oscillations per second
        this.phase = 0.0;     // phase offset [0, 2*PI]

        // Calculate initial rest length between attachment points
        const p1 = this.getAttachmentPoint(attachmentA, ratioA);
        const p2 = this.getAttachmentPoint(attachmentB, ratioB);
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        this.baseLength = Math.sqrt(dx * dx + dy * dy);
        this.length = this.baseLength;
    }

    getAttachmentPoint(attachment, ratio) {
        if (attachment instanceof Node) {
            return { x: attachment.x, y: attachment.y, mass: attachment.mass };
        } else { // Bone
            const nodeA = attachment.nodeA;
            const nodeB = attachment.nodeB;
            const px = nodeA.x + ratio * (nodeB.x - nodeA.x);
            const py = nodeA.y + ratio * (nodeB.y - nodeA.y);
            const mass = (1 - ratio) * nodeA.mass + ratio * nodeB.mass;
            return { x: px, y: py, mass: mass };
        }
    }

    resolve(time) {
        const p1 = this.getAttachmentPoint(this.attachmentA, this.ratioA);
        const p2 = this.getAttachmentPoint(this.attachmentB, this.ratioB);

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const currentLength = Math.sqrt(dx * dx + dy * dy);

        if (currentLength === 0) return;

        // Dynamic length modulation based on sinus oscillation
        const scale = 1.0 + this.amplitude * Math.sin(time * this.frequency * Math.PI * 2 + this.phase);
        const targetLength = this.length * scale;

        const diff = targetLength - currentLength;
        const percent = (diff / currentLength) * this.stiffness;
        const offsetX = dx * percent;
        const offsetY = dy * percent;

        // Share displacement inversely proportional to mass
        const mSum = p1.mass + p2.mass;
        if (mSum === 0) return;
        const factorA = p2.mass / mSum;
        const factorB = p1.mass / mSum;

        const shiftX1 = offsetX * factorA;
        const shiftY1 = offsetY * factorA;
        const shiftX2 = -offsetX * factorB;
        const shiftY2 = -offsetY * factorB;

        // Distribute displacements back to nodes
        this.applyShift(this.attachmentA, this.ratioA, shiftX1, shiftY1);
        this.applyShift(this.attachmentB, this.ratioB, shiftX2, shiftY2);
    }

    applyShift(attachment, ratio, sx, sy) {
        if (attachment instanceof Node) {
            attachment.x -= sx;
            attachment.y -= sy;
        } else { // Bone
            const nodeA = attachment.nodeA;
            const nodeB = attachment.nodeB;
            
            // Distribute shift to the two endpoints based on lever arm ratio
            nodeA.x -= sx * (1 - ratio);
            nodeA.y -= sy * (1 - ratio);
            nodeB.x -= sx * ratio;
            nodeB.y -= sy * ratio;
        }
    }
}

class Terrain {
    constructor(type = 'flat') {
        this.type = type;
        this.baseHeight = 480; // Ground level on canvas
    }

    getHeight(x) {
        switch (this.type) {
            case 'bumpy':
                // Sinusoidal hills
                return this.baseHeight + 35 * Math.sin(x * 0.015) + 15 * Math.sin(x * 0.045);
            
            case 'hurdles':
                // Flat ground with periodic rectangular blocks (hurdles)
                const ground = this.baseHeight;
                const hurdleSpacing = 350;
                const hurdleWidth = 20;
                const hurdleHeight = 45;
                const relativeX = x % hurdleSpacing;
                
                // Exclude the starting zone (x < 300) from hurdles to let creatures spawn safely
                if (x > 300 && relativeX < hurdleWidth) {
                    return ground - hurdleHeight;
                }
                return ground;
                
            case 'flat':
            default:
                return this.baseHeight;
        }
    }

    // Returns surface normal angle at position x (approximate)
    getNormal(x) {
        const delta = 1.0;
        const h1 = this.getHeight(x - delta);
        const h2 = this.getHeight(x + delta);
        const slope = (h2 - h1) / (2 * delta);
        return Math.atan2(-slope, 1);
    }
}

class Creature {
    constructor(id, nodes = [], constraints = []) {
        this.id = id;
        this.nodes = nodes;
        this.constraints = constraints;
        this.color = 'rgba(255,255,255,0.35)';
        this.isBest = false;
        this.fitness = 0;
        this.dna = []; // Array of float genes [0, 1]
    }

    getCenterOfMass() {
        if (this.nodes.length === 0) return { x: 0, y: 0 };
        let sumX = 0;
        let sumY = 0;
        for (const node of this.nodes) {
            sumX += node.x;
            sumY += node.y;
        }
        return {
            x: sumX / this.nodes.length,
            y: sumY / this.nodes.length
        };
    }

    update(gravity, damping, dt) {
        for (const node of this.nodes) {
            node.update(gravity, damping, dt);
        }
    }

    resolveConstraints(time) {
        for (const constraint of this.constraints) {
            constraint.resolve(time);
        }
    }

    handleCollisions(terrain) {
        for (const node of this.nodes) {
            const terrainY = terrain.getHeight(node.x);
            node.onGround = false;

            if (node.y > terrainY - node.radius) {
                // Collision! Push back
                node.y = terrainY - node.radius;
                node.onGround = true;

                // Ground friction: damp velocity along horizontal direction
                const vx = node.x - node.oldX;
                const groundFriction = 0.45; // High friction to allow crawling/pushing
                node.oldX = node.x - vx * (1 - groundFriction);

                // Vertical dampening (low bounce for walking simulation)
                const vy = node.y - node.oldY;
                if (vy > 0) {
                    node.oldY = node.y + vy * 0.05; 
                }
            }
        }
    }

    reset(offsetX = 0, offsetY = 0) {
        for (const node of this.nodes) {
            node.reset(offsetX, offsetY);
        }
        this.fitness = 0;
    }
}

class PhysicsWorld {
    constructor() {
        this.creatures = [];
        this.terrain = new Terrain('flat');
        this.gravity = 980.0;     // gravity constant (9.8m/s^2 * 100px/m)
        this.damping = 0.985;     // air resistance damping
        this.time = 0.0;         // accumulated simulation time
        this.solverIterations = 6; // stiffness solver iterations
    }

    setTerrain(type) {
        this.terrain = new Terrain(type);
    }

    addCreature(creature) {
        this.creatures.push(creature);
    }

    clear() {
        this.creatures = [];
        this.time = 0.0;
    }

    step(dt) {
        // 1. Update positions (Verlet step)
        for (const creature of this.creatures) {
            creature.update(this.gravity, this.damping, dt);
        }

        // 2. Resolve internal constraints (multiple iterations for stiffness)
        for (let i = 0; i < this.solverIterations; i++) {
            for (const creature of this.creatures) {
                creature.resolveConstraints(this.time);
            }
        }

        // 3. Resolve environmental collisions
        for (const creature of this.creatures) {
            creature.handleCollisions(this.terrain);
        }

        // 4. Update internal clock
        this.time += dt;
    }

    resetAll(offsetX = 0, offsetY = 0) {
        this.time = 0.0;
        for (const creature of this.creatures) {
            creature.reset(offsetX, offsetY);
        }
    }
}
