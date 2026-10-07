/**
 * Headless 30-day Ashvale run. Usage:
 *   npm run simulate            # seed "ashvale-1", 30 days
 *   npm run simulate -- mySeed 60
 */
import { createWorld, stepDay, describeEvent } from '../src/sim'

const seed = process.argv[2] ?? 'ashvale-1'
const days = Number(process.argv[3] ?? 30)

const world = createWorld(seed)
console.log(`Civilia — running ${world.name} for ${days} days (seed: ${seed})\n`)

for (let i = 0; i < days; i++) {
  stepDay(world)
}

const counts = new Map<string, number>()
for (const e of world.events) counts.set(e.type, (counts.get(e.type) ?? 0) + 1)

console.log(`Days simulated:  ${world.day - 1}`)
console.log(`Events emitted:  ${world.events.length}`)
console.log(`Rumors in town:  ${Object.keys(world.rumors).length}`)
console.log('\nEvent counts:')
for (const [type, n] of [...counts.entries()].sort((a, b) => b[1] - a[1])) {
  console.log(`  ${type.padEnd(18)} ${n}`)
}

console.log('\nAgents at end:')
for (const agent of Object.values(world.agents)) {
  const n = agent.needs
  console.log(
    `  ${agent.name.padEnd(14)} ${String(agent.jobId ?? 'no job').padEnd(8)}` +
      ` money=${String(agent.money).padStart(3)} hunger=${n.hunger.toFixed(0).padStart(3)}` +
      ` fatigue=${n.fatigue.toFixed(0).padStart(3)} status=${agent.status}` +
      ` rumorsKnown=${agent.knownRumorIds.length}`,
  )
}

console.log('\nRumors:')
for (const rumor of Object.values(world.rumors)) {
  console.log(`  [${rumor.id}] "${rumor.currentClaim}"`)
  console.log(
    `    believers=${rumor.believerIds.length} skeptics=${rumor.skepticIds.length}` +
      ` truthDistance=${rumor.truthDistance.toFixed(2)} variants=${rumor.variants.length}`,
  )
}

console.log('\nLast 15 events:')
for (const e of world.events.slice(-15)) {
  console.log(`  [d${e.day} ${e.phase}] ${describeEvent(world, e)}`)
}
