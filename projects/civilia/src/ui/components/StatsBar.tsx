import type { World } from '../../sim'

interface Props {
  world: World
}

export function StatsBar({ world }: Props) {
  const agents = Object.values(world.agents)
  const bakery = world.places['place_bakery']
  const tavern = world.places['place_tavern']
  const totalMoney = agents.reduce((sum, a) => sum + a.money, 0)
  const avgHunger = agents.reduce((sum, a) => sum + a.needs.hunger, 0) / agents.length
  const starving = agents.filter((a) => a.needs.hunger >= 85).length
  const rumors = Object.values(world.rumors)

  const hottest = rumors.reduce(
    (best, r) => (r.believerIds.length > (best?.believerIds.length ?? -1) ? r : best),
    rumors[0],
  )

  return (
    <div className="stats-bar">
      <span className="chip" title="bread at the bakery">
        🥖 bakery {bakery.foodStock} @ {bakery.foodPrice}c
      </span>
      <span className="chip" title="meals at the tavern">
        🍺 tavern {tavern.foodStock} @ {tavern.foodPrice}c
      </span>
      <span className="chip" title="all coins held by townsfolk">
        💰 {totalMoney} coins in town
      </span>
      <span className={`chip ${avgHunger > 60 ? 'alert' : ''}`} title="average hunger (0 full, 100 starving)">
        🍽️ hunger {avgHunger.toFixed(0)}
      </span>
      {starving > 0 && (
        <span className="chip alert" title="agents with hunger ≥ 85">
          🚨 {starving} starving
        </span>
      )}
      {bakery.foodStock === 0 && <span className="chip alert">🚨 bread shortage</span>}
      <span className="chip" title="rumors circulating">
        🗣️ {rumors.length} rumors
      </span>
      {hottest && (
        <span className="chip rumor-chip" title={`"${hottest.currentClaim}" — ${hottest.believerIds.length} believers, drift ${hottest.truthDistance.toFixed(2)}`}>
          🔥 “{hottest.currentClaim.length > 44 ? hottest.currentClaim.slice(0, 44) + '…' : hottest.currentClaim}”
        </span>
      )}
      <span className="chip dim" title="events in the log">
        📜 {world.events.length}
      </span>
    </div>
  )
}
