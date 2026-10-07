/**
 * Darwin's Doodles - Creature Presets Templates
 */

const CreaturePresets = {
    /**
     * Helper to build a creature from template data
     */
    buildFromTemplate(id, template, startX = 150, startY = 320) {
        const nodes = template.nodes.map(n => {
            return new Node(startX + n.dx, startY + n.dy, n.radius || 8, n.mass || 1.0);
        });

        const constraints = [];
        
        // 1. Build Bones first
        template.constraints.forEach(c => {
            if (!c.isMuscle) {
                const bone = new Bone(nodes[c.a], nodes[c.b], c.stiffness || 0.85);
                constraints.push(bone);
            }
        });

        // 2. Build Muscles
        template.constraints.forEach(c => {
            if (c.isMuscle) {
                if (c.ratioA !== undefined) {
                    // Bone-to-Bone muscle
                    const boneA = constraints[c.a];
                    const boneB = constraints[c.b];
                    const muscle = new Muscle(boneA, boneB, c.ratioA, c.ratioB, c.stiffness || 0.8);
                    constraints.push(muscle);
                } else {
                    // Joint-to-Joint muscle (Presets)
                    const muscle = new Muscle(nodes[c.a], nodes[c.b], 0, 0, c.stiffness || 0.8);
                    constraints.push(muscle);
                }
            }
        });

        const creature = new Creature(id, nodes, constraints);
        creature.color = template.color || 'rgba(16, 185, 129, 0.4)';
        return creature;
    },

    templates: {
        biped: {
            color: 'rgba(14, 165, 233, 0.45)', // Sky blue theme
            nodes: [
                { dx: 0, dy: -60, radius: 11, mass: 2.0 },  // 0: Torso (heavy, high up)
                { dx: -18, dy: -25, radius: 7, mass: 1.0 }, // 1: Hip Left
                { dx: 18, dy: -25, radius: 7, mass: 1.0 },  // 2: Hip Right
                { dx: -25, dy: 10, radius: 6, mass: 0.8 },  // 3: Knee Left
                { dx: 25, dy: 10, radius: 6, mass: 0.8 },   // 4: Knee Right
                { dx: -30, dy: 50, radius: 8, mass: 1.2 },  // 5: Foot Left
                { dx: 30, dy: 50, radius: 8, mass: 1.2 }   // 6: Foot Right
            ],
            constraints: [
                // Torso / Pelvis bones (rigid structure)
                { a: 0, b: 1, stiffness: 0.9, isMuscle: false },
                { a: 0, b: 2, stiffness: 0.9, isMuscle: false },
                { a: 1, b: 2, stiffness: 0.9, isMuscle: false },
                
                // Upper Legs (hip to knee) - Muscle active
                { a: 1, b: 3, stiffness: 0.8, isMuscle: true },
                { a: 2, b: 4, stiffness: 0.8, isMuscle: true },
                
                // Lower Legs (knee to foot) - Muscle active
                { a: 3, b: 5, stiffness: 0.8, isMuscle: true },
                { a: 4, b: 6, stiffness: 0.8, isMuscle: true },
                
                // Hip to Foot direct muscles (adds extension force)
                { a: 1, b: 5, stiffness: 0.7, isMuscle: true },
                { a: 2, b: 6, stiffness: 0.7, isMuscle: true },

                // Across body muscle constraints (crossover coordination)
                { a: 0, b: 3, stiffness: 0.6, isMuscle: true },
                { a: 0, b: 4, stiffness: 0.6, isMuscle: true }
            ]
        },

        crawler: {
            color: 'rgba(236, 72, 153, 0.45)', // Rose/Pink theme
            nodes: [
                { dx: -60, dy: 30, radius: 8, mass: 1.0 },  // 0: Segment 1 (Tail)
                { dx: -20, dy: 30, radius: 8, mass: 1.0 },  // 1: Segment 2
                { dx: 20, dy: 30, radius: 8, mass: 1.0 },   // 2: Segment 3
                { dx: 60, dy: 30, radius: 9, mass: 1.2 },   // 3: Segment 4 (Head)
                
                { dx: -40, dy: 0, radius: 6, mass: 0.8 },   // 4: Top Spine 1
                { dx: 0, dy: -10, radius: 6, mass: 0.8 },   // 5: Top Spine 2
                { dx: 40, dy: 0, radius: 6, mass: 0.8 }     // 6: Top Spine 3
            ],
            constraints: [
                // Ground segments (horizontal) - Evolved muscles
                { a: 0, b: 1, stiffness: 0.8, isMuscle: true },
                { a: 1, b: 2, stiffness: 0.8, isMuscle: true },
                { a: 2, b: 3, stiffness: 0.8, isMuscle: true },
                
                // Vertical columns (stabilizers)
                { a: 0, b: 4, stiffness: 0.9, isMuscle: false },
                { a: 1, b: 4, stiffness: 0.9, isMuscle: false },
                { a: 1, b: 5, stiffness: 0.9, isMuscle: false },
                { a: 2, b: 5, stiffness: 0.9, isMuscle: false },
                { a: 2, b: 6, stiffness: 0.9, isMuscle: false },
                { a: 3, b: 6, stiffness: 0.9, isMuscle: false },

                // Top ridge spine constraints
                { a: 4, b: 5, stiffness: 0.85, isMuscle: true },
                { a: 5, b: 6, stiffness: 0.85, isMuscle: true },

                // Diagonals (drive the arching motion)
                { a: 0, b: 5, stiffness: 0.75, isMuscle: true },
                { a: 3, b: 5, stiffness: 0.75, isMuscle: true }
            ]
        },

        quadruped: {
            color: 'rgba(16, 185, 129, 0.45)', // Emerald theme
            nodes: [
                // Long horizontal torso (triangulated box)
                { dx: -45, dy: -40, radius: 9, mass: 1.5 },  // 0: Hip (Rear joint)
                { dx: 45, dy: -40, radius: 9, mass: 1.5 },   // 1: Shoulder (Front joint)
                { dx: 0, dy: -55, radius: 7, mass: 1.0 },    // 2: Back Spine Center
                
                // Rear Leg Left
                { dx: -55, dy: -5, radius: 6, mass: 0.8 },   // 3: Rear Knee Left
                { dx: -60, dy: 35, radius: 7, mass: 1.0 },   // 4: Rear Foot Left
                
                // Rear Leg Right
                { dx: -35, dy: -5, radius: 6, mass: 0.8 },   // 5: Rear Knee Right
                { dx: -30, dy: 35, radius: 7, mass: 1.0 },   // 6: Rear Foot Right
                
                // Front Leg Left
                { dx: 35, dy: -5, radius: 6, mass: 0.8 },    // 7: Front Knee Left
                { dx: 30, dy: 35, radius: 7, mass: 1.0 },    // 8: Front Foot Left
                
                // Front Leg Right
                { dx: 55, dy: -5, radius: 6, mass: 0.8 },    // 9: Front Knee Right
                { dx: 60, dy: 35, radius: 7, mass: 1.0 }     // 10: Front Foot Right
            ],
            constraints: [
                // Torso Frame
                { a: 0, b: 1, stiffness: 0.9, isMuscle: false },
                { a: 0, b: 2, stiffness: 0.9, isMuscle: false },
                { a: 1, b: 2, stiffness: 0.9, isMuscle: false },

                // Rear Leg Left
                { a: 0, b: 3, stiffness: 0.8, isMuscle: true },
                { a: 3, b: 4, stiffness: 0.8, isMuscle: true },
                { a: 0, b: 4, stiffness: 0.7, isMuscle: true }, // Hip-to-foot spring

                // Rear Leg Right
                { a: 0, b: 5, stiffness: 0.8, isMuscle: true },
                { a: 5, b: 6, stiffness: 0.8, isMuscle: true },
                { a: 0, b: 6, stiffness: 0.7, isMuscle: true },

                // Front Leg Left
                { a: 1, b: 7, stiffness: 0.8, isMuscle: true },
                { a: 7, b: 8, stiffness: 0.8, isMuscle: true },
                { a: 1, b: 8, stiffness: 0.7, isMuscle: true },

                // Front Leg Right
                { a: 1, b: 9, stiffness: 0.8, isMuscle: true },
                { a: 9, b: 10, stiffness: 0.8, isMuscle: true },
                { a: 1, b: 10, stiffness: 0.7, isMuscle: true },
                
                // Coordination muscle across spine
                { a: 2, b: 3, stiffness: 0.6, isMuscle: true },
                { a: 2, b: 9, stiffness: 0.6, isMuscle: true }
            ]
        },

        starfish: {
            color: 'rgba(245, 158, 11, 0.45)', // Amber/Gold theme
            nodes: [
                { dx: 0, dy: 0, radius: 10, mass: 1.8 },    // 0: Center core
                { dx: 0, dy: -45, radius: 7, mass: 0.9 },   // 1: Arm Top
                { dx: 43, dy: -14, radius: 7, mass: 0.9 },  // 2: Arm Right-Top
                { dx: 26, dy: 36, radius: 7, mass: 0.9 },   // 3: Arm Right-Bottom
                { dx: -26, dy: 36, radius: 7, mass: 0.9 },  // 4: Arm Left-Bottom
                { dx: -43, dy: -14, radius: 7, mass: 0.9 }  // 5: Arm Left-Top
            ],
            constraints: [
                // Core spokes (arms to center)
                { a: 0, b: 1, stiffness: 0.8, isMuscle: true },
                { a: 0, b: 2, stiffness: 0.8, isMuscle: true },
                { a: 0, b: 3, stiffness: 0.8, isMuscle: true },
                { a: 0, b: 4, stiffness: 0.8, isMuscle: true },
                { a: 0, b: 5, stiffness: 0.8, isMuscle: true },

                // Perimeter web (connects adjacent arms)
                { a: 1, b: 2, stiffness: 0.7, isMuscle: true },
                { a: 2, b: 3, stiffness: 0.7, isMuscle: true },
                { a: 3, b: 4, stiffness: 0.7, isMuscle: true },
                { a: 4, b: 5, stiffness: 0.7, isMuscle: true },
                { a: 5, b: 1, stiffness: 0.7, isMuscle: true }
            ]
        },

        custom: {
            color: 'rgba(168, 85, 247, 0.45)', // Custom purple theme
            nodes: [
                { dx: -20, dy: 0, radius: 8, mass: 1.0 },
                { dx: 20, dy: 0, radius: 8, mass: 1.0 }
            ],
            constraints: [
                { a: 0, b: 1, stiffness: 0.8, isMuscle: true }
            ]
        }
    },

    get(type, id, startX, startY) {
        const template = this.templates[type] || this.templates.biped;
        return this.buildFromTemplate(id, template, startX, startY);
    }
};
