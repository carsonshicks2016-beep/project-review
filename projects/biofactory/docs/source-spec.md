# BioFactory Source Spec

## North Star

BioFactory is not a normal ant simulator. It is a living supply-chain simulator:

- The colony is the main character.
- Ants are moving logistics packets.
- Pheromones are the biological circuit network.
- The nest is the factory.
- Resources are the production chain.
- Evolution is the optimizer.

The simulator should produce logistics behavior through local perception, demand pressure, pheromone gradients, congestion, resource placement, and evolutionary feedback. Complex strategies such as warehouse districts, efficient highways, flanking, or siege tactics should not be scripted directly.

## MVP Success Condition

Starting from one nest and scattered resources, ants should discover sources, carry resources back to storage, reinforce trails with pheromones, and gradually create visible logistics highways. Storage should create demand, resources should flow toward storage, and congestion should appear when many ants reuse a route.

## MVP Systems

- 2D terrain grid with continuous-position ants.
- Terrain affects speed, energy, pheromone decay, diffusion, visibility, and construction difficulty.
- One colony with a queen/storage center.
- Typed seeded storage chambers for leaves, food, water, protein, and waste.
- 500 ants by default.
- Leaves, water, protein, and waste as the first raw logistics resources.
- Simplified processing where stored leaves become food.
- Food, water, protein, waste, demand, and traffic pheromone layers.
- Resource demand states with shortage urgency, saturation penalties, critical thresholds, and bottleneck visibility.
- Chamber-owned inventory, aggregate colony totals, typed cargo dropoff, overflow fallback, and chamber inspection.
- Tiny feedforward neural brain per ant.
- Local behavior rules for target-resource choice, pickup, dropoff, pheromone laying, movement, and congestion avoidance.
- Pygame rendering, camera controls, speed controls, pause/step, and overlay toggles.
- Tests for core mechanics.

## Expansion Order

1. Multiple resource types. Implemented foundation: leaves, water, protein, waste.
2. Better demand system. Implemented foundation: resource urgency, saturation, and critical alerts.
3. Multiple storage chambers. Implemented foundation: seeded typed storage, chamber-local demand, and typed dropoff.
4. Basic processing chain.
5. Fungus farm.
6. Nursery.
7. Larvae and reproduction.
8. Role inference.
9. Traffic metrics.
10. Congestion penalties.
11. Nest digging.
12. Chamber construction.
13. Weather.
14. Seasons.
15. Predators.
16. Rival colonies.
17. Combat.
18. Territory pheromones.
19. Waste and recycling.
20. Disease and parasites.
21. Replay system.
22. Advanced evolution.
23. Industrial district emergence.
24. Large-scale optimization.

## Design Guardrails

- Keep simulation logic separate from rendering.
- Keep all constants config-driven.
- Use NumPy arrays for grid systems.
- Use spatial indexing for large entity counts as systems grow.
- Keep ant brains small enough for thousands of agents.
- Prefer local signals and rewards over central colony strategy.
- Add one major system at a time after the MVP proves visible trail formation.
