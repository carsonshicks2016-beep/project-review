import type { World } from '../../sim'

interface Props {
  world: World
  selectedAgentId: string | null
  onSelectAgent: (id: string) => void
}

export function TownPanel({ world, selectedAgentId, onSelectAgent }: Props) {
  const agents = Object.values(world.agents)
  const places = Object.values(world.places)

  return (
    <div className="town-panel">
      <h2>Townsfolk</h2>
      <ul className="agent-list">
        {agents.map((agent) => (
          <li key={agent.id}>
            <button
              className={`agent-row ${agent.id === selectedAgentId ? 'selected' : ''}`}
              onClick={() => onSelectAgent(agent.id)}
            >
              <span className="agent-name">{agent.name}</span>
              <span className="agent-job">{agent.jobId ?? 'no job'}</span>
              <span className="agent-where">{world.places[agent.locationId]?.name}</span>
            </button>
          </li>
        ))}
      </ul>

      <h2>Places</h2>
      <ul className="place-list">
        {places.map((place) => {
          const here = agents.filter((a) => a.locationId === place.id)
          return (
            <li key={place.id} className="place-row">
              <div className="place-head">
                <span className="place-name">{place.name}</span>
                {place.foodPrice > 0 && (
                  <span className="place-stock">
                    bread ×{place.foodStock} @ {place.foodPrice}c
                  </span>
                )}
              </div>
              {here.length > 0 && (
                <div className="place-occupants">
                  {here.map((a) => (
                    <button key={a.id} className="occupant" onClick={() => onSelectAgent(a.id)}>
                      {a.name.split(' ')[0]}
                    </button>
                  ))}
                </div>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
