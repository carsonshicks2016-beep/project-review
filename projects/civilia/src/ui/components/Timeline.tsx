import { useState } from 'react'
import type { SimEvent, World } from '../../sim'
import { describeEvent } from '../../sim'

interface Props {
  world: World
  selectedAgentId: string | null
}

const MAX_RENDERED = 200

const CATEGORY_OF: Record<string, string> = {
  worked: 'work',
  wage_paid: 'work',
  produced_good: 'work',
  bought_good: 'economy',
  sold_good: 'economy',
  price_changed: 'economy',
  ate_food: 'daily',
  slept: 'daily',
  traveled: 'daily',
  talked: 'social',
  helped: 'social',
  insulted: 'conflict',
  shared_rumor: 'rumor',
  stole_item: 'crime',
  witnessed_crime: 'crime',
}

/** Low-signal event types hidden unless "show routine" is on. */
const ROUTINE_TYPES = new Set(['traveled', 'slept', 'wage_paid', 'sold_good', 'ate_food'])

export function Timeline({ world, selectedAgentId }: Props) {
  const [onlySelected, setOnlySelected] = useState(false)
  const [showRoutine, setShowRoutine] = useState(false)

  const involves = (e: SimEvent, id: string) =>
    e.actorIds.includes(id) || e.targetIds.includes(id) || e.witnessIds.includes(id)

  let events = world.events
  if (onlySelected && selectedAgentId) events = events.filter((e) => involves(e, selectedAgentId))
  if (!showRoutine) events = events.filter((e) => !ROUTINE_TYPES.has(e.type))
  const shown = events.slice(-MAX_RENDERED).reverse()

  return (
    <div className="timeline">
      <div className="timeline-head">
        <h2>Event timeline</h2>
        <span className="timeline-count">
          {events.length} events{events.length > MAX_RENDERED ? ` (showing last ${MAX_RENDERED})` : ''}
        </span>
        <label className="toggle">
          <input type="checkbox" checked={onlySelected} onChange={(e) => setOnlySelected(e.target.checked)} />
          selected agent only
        </label>
        <label className="toggle">
          <input type="checkbox" checked={showRoutine} onChange={(e) => setShowRoutine(e.target.checked)} />
          show routine
        </label>
      </div>
      <ol className="event-list">
        {shown.map((e) => (
          <li key={e.id} className={`event-row cat-${CATEGORY_OF[e.type] ?? 'daily'}`}>
            <span className="event-when">
              d{e.day} {e.phase}
            </span>
            <span className="event-type">{e.type}</span>
            <span className="event-desc">{describeEvent(world, e)}</span>
            {e.witnessIds.length > 0 && (
              <span className="event-witnesses" title="witnesses">
                👁 {e.witnessIds.length}
              </span>
            )}
          </li>
        ))}
        {shown.length === 0 && <li className="event-empty">No events yet — step the simulation.</li>}
      </ol>
    </div>
  )
}
