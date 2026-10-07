// Racing fitness.
//
// The previous version paid `actualTicks * dt * 10` simply for staying airborne, and
// capped the speed bonus at (5 - timePerGate) * 1000. Measured on a real run: 60s of
// flight earned 600 points of survival while the speed bonus earned 400, so dawdling
// literally scored better than hurrying. Halving lap time was worth ~2,000 net -- a fifth
// of one gate -- against a 10,000 penalty for missing a gate. A 5:1 risk/reward against
// flying fast, and the GA correctly learned to crawl.
//
// This version pays the time bonus *at the moment each gate is cleared*, so being early
// compounds over the lap and dying early earns nothing.

export const GATE_REWARD = 10000;
export const TIME_BONUS_PER_GATE = 4000; // Up to 40% of a gate for clearing it promptly
export const COMPLETION_BONUS = 50000;

export function calculateFitness({
  gatesPassed, totalGates, distToNextGate, prevGateDist, minDistToNextGate,
  actualTicks, crashed, dt, jitterPenalty, gateTimeBonus = 0
}) {
  let fitness = 0;

  // 1. Gate completion stays the dominant term: clearing more gates should still beat
  //    clearing fewer of them quickly.
  fitness += gatesPassed * GATE_REWARD;

  // 2. Accumulated time bonus, earned per gate at the tick it was cleared. Sized so a
  //    fast lap beats a slow one but never beats an extra gate outright.
  fitness += gateTimeBonus;

  // 3. Progress toward the next gate: the main gradient before any gate is cleared.
  if (prevGateDist > 0) {
    const progressFrac = Math.max(0, 1 - distToNextGate / prevGateDist);
    fitness += progressFrac * 5000;
  }

  // 4. Closest approach: gradient for drones that fly near a gate but miss it.
  if (minDistToNextGate < prevGateDist && prevGateDist > 0) {
    const approachFrac = Math.max(0, 1 - minDistToNextGate / prevGateDist);
    fitness += approachFrac * 2000;
  }

  // 5. Bootstrap only. Early populations need a reason to stay airborne at all, but once
  //    a drone is actually racing, elapsed time must stop paying -- that was the bug.
  if (gatesPassed === 0) {
    fitness += Math.min(actualTicks * dt, 5.0) * 200;
  }

  // 6. Crash penalty (mild - we still want exploration).
  if (crashed) fitness -= 500;

  // 7. Jitter, as a per-tick mean. Summing it made the penalty grow with flight time, so
  //    a drone was punished for staying alive longer.
  if (jitterPenalty && actualTicks > 0) {
    fitness -= (jitterPenalty / actualTicks) * 3000;
  }

  // 8. Completion.
  if (gatesPassed >= totalGates) fitness += COMPLETION_BONUS;

  // No clamp to 0.01: it collapsed every poor genome onto an identical score, leaving
  // tournament selection nothing to discriminate on across most of the population.
  return fitness;
}
