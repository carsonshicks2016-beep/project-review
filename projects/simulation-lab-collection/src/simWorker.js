// simWorker.js -- headless GA evaluation worker.
//
// Module worker: it imports the real DronePhysics, NeuralNet, gate collision, fitness and
// observation code instead of inlining copies. The inlined copies were how the same
// quaternion bug came to exist in two places at once and had to be fixed twice.
import { DronePhysics } from './dronePhysics.js';
import { NeuralNet, LAYER_SIZES } from './neuralNet.js';
import { checkGatePassing, distanceToGate } from './gateCollision.js';
import { calculateFitness, TIME_BONUS_PER_GATE } from './fitness.js';
import { VoxelGrid } from './collision/voxelGrid.js';
import { OBS_DIM, buildObservation } from './observation.js';
import { getOutdoorElevation } from './terrainHeight.js';

// ============================================================
// WORKER MESSAGE HANDLER
// ============================================================
const net = new NeuralNet(LAYER_SIZES);
let worldGrid = null;

self.onmessage = function(e) {
  const { genomes, gates, startPos, startYaw, maxTicks = 1800, dt = 1 / 60, startGateIdx = 0, environment, grid } = e.data;

  // Shipped once and cached: a couple of megabytes that only changes when the scan does.
  if (grid !== undefined) worldGrid = grid ? VoxelGrid.deserialize(grid) : null;
  const getTerrainHeight = environment === 'outdoor' ? getOutdoorElevation : null;
  const totalGates = gates.length;
  // Curriculum: the episode begins in front of startGateIdx, so the gates actually
  // available to clear are the ones from there to the end of the lap.
  const gatesAvailable = totalGates - startGateIdx;
  const results = [];

  for (let g = 0; g < genomes.length; g++) {
    const genome = new Float32Array(genomes[g]);
    const drone = new DronePhysics({ getTerrainHeight });
    // Solid geometry only applies to the scanned interior; outdoors the terrain function
    // is the ground and there is no scan to collide with.
    drone.worldGrid = environment === 'lidar' ? worldGrid : null;
    drone.reset(startPos || [0, 5, 0], startYaw || 0);

    let currentGateIdx = startGateIdx;
    let ticksSinceLastGate = 0;
    let topSpeed = 0;
    let actualTicks = 0;
    let jitterPenalty = 0;
    let gateTimeBonus = 0;
    let prevControls = null;
    const obs = new Float32Array(OBS_DIM);
    const trajectory = [];

    // Precompute initial distance to first gate
    let prevGateDist = distanceToGate(drone.pos, gates[startGateIdx]);
    let minDistToNextGate = prevGateDist;

    for (let tick = 0; tick < maxTicks; tick++) {
      if (!drone.alive) break;
      if (currentGateIdx >= totalGates) break; // Completed full circuit
      actualTicks = tick + 1;

      const curGate = gates[currentGateIdx];
      const nextGateIdx = (currentGateIdx + 1) % totalGates;
      const nextGate = gates[nextGateIdx];

      buildObservation(drone, gates, currentGateIdx, drone.worldGrid, obs, getTerrainHeight);

      const controls = net.forward(obs, genome);
      
      // Calculate Jitter (rapid change in controls across single frames)
      if (prevControls) {
          jitterPenalty += Math.abs(controls.pitch - prevControls.pitch) +
                           Math.abs(controls.roll - prevControls.roll) +
                           Math.abs(controls.yaw - prevControls.yaw) +
                           Math.abs(controls.thrust - prevControls.thrust);
      }
      prevControls = controls;

      const prevPos = [drone.pos[0], drone.pos[1], drone.pos[2]];
      drone.step(controls, dt);

      // Track speed
      const speed = Math.sqrt(drone.vel[0]**2 + drone.vel[1]**2 + drone.vel[2]**2);
      if (speed > topSpeed) topSpeed = speed;

      // Gate collision check
      const hit = checkGatePassing(prevPos, [drone.pos[0], drone.pos[1], drone.pos[2]], curGate);
      if (hit.status === 'PASSED') {
        currentGateIdx++;
        // Banked at the tick it was cleared: full value at t=0, nothing at the cap.
        gateTimeBonus += (1 - tick / maxTicks) * TIME_BONUS_PER_GATE;
        ticksSinceLastGate = 0;
        if (currentGateIdx < totalGates) {
          prevGateDist = distanceToGate(drone.pos, gates[currentGateIdx]);
          minDistToNextGate = prevGateDist;
        }
      }
      // CRASHED_FRAME: clipped the gate edge — don't kill, just note it

      // Track closest approach to current target gate
      if (currentGateIdx < totalGates) {
        const curDist = distanceToGate(drone.pos, gates[currentGateIdx]);
        if (curDist < minDistToNextGate) minDistToNextGate = curDist;
      }

      ticksSinceLastGate++;
      if (ticksSinceLastGate > 600) { // 10 seconds stagnation
        drone.alive = false;
      }

      // Record keyframe every 3 ticks for replay (~20fps)
      if (tick % 3 === 0) {
        trajectory.push(
          drone.pos[0], drone.pos[1], drone.pos[2],
          drone.quat[0], drone.quat[1], drone.quat[2], drone.quat[3]
        );
      }
    }

    const distToNext = currentGateIdx < totalGates
      ? distanceToGate(drone.pos, gates[currentGateIdx])
      : 0;

    const fitness = calculateFitness({
      gatesPassed: currentGateIdx - startGateIdx,
      totalGates: gatesAvailable,
      distToNextGate: distToNext,
      prevGateDist: prevGateDist || 30.0,
      minDistToNextGate,
      actualTicks,
      crashed: drone.crashed,
      dt,
      jitterPenalty,
      gateTimeBonus
    });

    results.push({
      fitness,
      gatesPassed: currentGateIdx - startGateIdx, // Cleared this episode
      hitWall: !!drone.hitWall,                   // Distinguishes a wall from the ground
      finalGateIdx: currentGateIdx,               // Absolute index, for the gate pips
      topSpeed: topSpeed * 3.6, // m/s to km/h
      trajectoryLength: trajectory.length / 7 // number of keyframes
    });

    // Only store full trajectory for potential best (top 3 fitness in this batch)
    results[results.length - 1].trajectory = new Float32Array(trajectory);
  }

  // Rank by fitness so we can ship the winner plus a few runners-up. The runners-up
  // are drawn as ghost paths on the main thread: seeing the spread of the population
  // is what makes "the GA is converging" legible instead of abstract.
  const order = results.map((_, i) => i).sort((a, b) => results[b].fitness - results[a].fitness);
  const bestIdx = order[0];
  const bestTrajectory = results[bestIdx].trajectory;

  // Excludes bestIdx, so no buffer appears twice in the transfer list.
  const GHOSTS_PER_WORKER = 3;
  const ghostIdx = order.slice(1, 1 + GHOSTS_PER_WORKER);
  const ghosts = ghostIdx.map(i => ({
    fitness: results[i].fitness,
    buffer: results[i].trajectory.buffer
  }));

  // Strip the rest to save transfer cost
  const keep = new Set([bestIdx, ...ghostIdx]);
  for (let i = 0; i < results.length; i++) {
    if (!keep.has(i)) delete results[i].trajectory;
  }

  self.postMessage({
    results: results.map(r => ({
      fitness: r.fitness,
      gatesPassed: r.gatesPassed,
      hitWall: r.hitWall,
      finalGateIdx: r.finalGateIdx,
      topSpeed: r.topSpeed
    })),
    bestLocalIdx: bestIdx,
    bestTrajectory: bestTrajectory.buffer,
    ghosts
  }, [bestTrajectory.buffer, ...ghosts.map(g => g.buffer)]);
};
