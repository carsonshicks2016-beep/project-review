import type { SimEvent, World } from '../../sim'
import { describeEvent } from '../../sim'

interface Props {
  world: World
  onSelectAgent: (id: string) => void
}

const MAX_SHOWN = 9

const TICKER_ICON: Partial<Record<SimEvent['type'], string>> = {
  stole_item: '🥷',
  witnessed_crime: '👁️',
  insulted: '💢',
  helped: '🤝',
  shared_rumor: '🤫',
  price_changed: '💰',
}

/** Is this event juicy enough for the live ticker? */
function notable(event: SimEvent): boolean {
  switch (event.type) {
    case 'stole_item':
      return (event.payload as any).caught === true
    case 'shared_rumor':
      return (event.payload as any).mutated === true
    case 'insulted':
      return (event.payload as any).publicly === true
    case 'witnessed_crime':
    case 'helped':
    case 'price_changed':
      return true
    default:
      return event.consequenceLevel >= 2
  }
}

export function EventTicker({ world, onSelectAgent }: Props) {
  const items = world.events.filter(notable).slice(-MAX_SHOWN).reverse()

  return (
    <div className="event-ticker">
      <div className="ticker-title">Town happenings</div>
      {items.map((event) => (
        <button
          key={event.id}
          className={`ticker-item tick-${event.type}`}
          onClick={() => event.actorIds[0] && onSelectAgent(event.actorIds[0])}
          title="click to inspect the agent involved"
        >
          <span className="ticker-icon">{TICKER_ICON[event.type] ?? '•'}</span>
          <span className="ticker-text">
            {event.type === 'shared_rumor'
              ? `Rumor twists: “${(event.payload as any).claim}”`
              : describeEvent(world, event)}
          </span>
          <span className="ticker-when">
            d{event.day} {event.phase}
          </span>
        </button>
      ))}
      {items.length === 0 && <div className="ticker-empty">Nothing scandalous yet…</div>}
    </div>
  )
}
