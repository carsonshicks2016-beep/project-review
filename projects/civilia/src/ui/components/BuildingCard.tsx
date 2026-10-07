import type { Agent, Building, MaterialId, World } from '../../sim'
import { householdColor } from '../palette'

interface Props {
  world: World
  placeId: string
  onSelectAgent: (id: string) => void
  onClose: () => void
}

const MATERIAL_ICON: Record<MaterialId, string> = {
  timber: '🌲',
  stone: '🪨',
  clay: '🧱',
  reeds: '🌾',
}

const STATE_LABEL: Record<Building['state'], string> = {
  planned: 'Planned',
  under_construction: 'Under construction',
  complete: '',
}

/** What is this agent doing right now, per their latest decision trace. */
function activityOf(world: World, agent: Agent): string {
  const traces = world.decisionTraces[agent.id]
  const last = traces?.[traces.length - 1]
  if (!last) return 'settling in'
  if (last.chosenIndex < 0) return 'idling'
  return last.candidates[last.chosenIndex].label
}

export function BuildingCard({ world, placeId, onSelectAgent, onClose }: Props) {
  const place = world.places[placeId]
  if (!place) return null
  const building = place.buildingId ? world.buildings[place.buildingId] : undefined
  const here = Object.values(world.agents).filter((a) => a.locationId === placeId)
  const inside = here.filter((a) => a.indoors)
  const outside = here.filter((a) => !a.indoors)
  const owner = building?.ownerHouseholdId ? world.households[building.ownerHouseholdId] : undefined

  const occupantRow = (agent: Agent) => (
    <li key={agent.id}>
      <button className="agent-link" onClick={() => onSelectAgent(agent.id)}>
        <span
          className="household-dot"
          style={{ background: householdColor(world, agent.householdId) }}
        />
        {agent.name}
      </button>
      <span className="occupant-activity"> — {activityOf(world, agent)}</span>
    </li>
  )

  return (
    <div className="place-card">
      <div className="place-card-head">
        <strong>{place.name}</strong>
        <button className="close-btn" onClick={onClose}>
          ✕
        </button>
      </div>

      {building && building.state !== 'complete' && (
        <div className="place-card-row building-state">
          <span className={`state-chip state-${building.state}`}>
            {STATE_LABEL[building.state]}
            {building.state === 'under_construction' && ` · ${Math.round(building.progress * 100)}%`}
          </span>
          <div className="dim-text building-reason">{building.reason}</div>
          <div className="materials-row">
            {(Object.keys(building.materialsNeeded) as MaterialId[])
              .filter((m) => building.materialsNeeded[m] > 0)
              .map((m) => (
                <span key={m} className="material-chip">
                  {MATERIAL_ICON[m]} {building.materialsDelivered[m]}/{building.materialsNeeded[m]}
                </span>
              ))}
          </div>
        </div>
      )}

      {owner && (
        <div className="place-card-row dim-text">
          Held by the {owner.name}
          {owner.name.includes('cottage') ? '' : ' household'}
        </div>
      )}
      {place.foodPrice > 0 && (
        <div className="place-card-row">
          🍞 {place.foodStock} in stock · {place.foodPrice} coins each
        </div>
      )}
      {(place.kind === 'farm' || place.kind === 'granary') && (
        <div className="place-card-row">🌾 {place.grainStore} grain stored</div>
      )}

      <div className="place-card-row occupants">
        <div className="occupants-heading">Inside ({inside.length})</div>
        {inside.length === 0 ? (
          <span className="dim-text">Nobody indoors.</span>
        ) : (
          <ul className="occupant-list">{inside.map(occupantRow)}</ul>
        )}
        <div className="occupants-heading">Outside ({outside.length})</div>
        {outside.length === 0 ? (
          <span className="dim-text">Nobody in the yard.</span>
        ) : (
          <ul className="occupant-list">{outside.map(occupantRow)}</ul>
        )}
      </div>
    </div>
  )
}
