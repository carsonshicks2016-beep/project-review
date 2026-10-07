import type { World } from '../../sim'
import type { Utterance } from '../../sim'
import { SPEECH_COLORS } from '../palette'

/**
 * The "Why?" inspector — clicking any bubble opens the reasoning tree:
 * motivation, the historical context the line draws on (the exact memory,
 * with the day it formed and the events behind it), the knowledge that
 * influenced it (beliefs, with confidence and who told them), and for
 * gossip, how the claim landed on the listener.
 */

export type PhraseSelection =
  | { kind: 'event'; eventId: string }
  | { kind: 'thought'; agentId: string }

interface Props {
  world: World
  selection: PhraseSelection
  onSelectAgent: (id: string) => void
  onClose: () => void
}

const MOTIVE_LABEL: Record<string, string> = {
  genuine: 'Genuine',
  bonding: 'Bonding',
  premeditated: 'Premeditated',
  deceitful: 'Deceitful',
  performative: 'Performative',
  desperate: 'Desperate',
  venting: 'Venting',
}

export function PhraseCard({ world, selection, onSelectAgent, onClose }: Props) {
  let speakerId: string | undefined
  let listenerName: string | undefined
  let speech: Utterance | undefined
  let context: string | undefined
  let receptionFactors: string[] | undefined
  let receptionConfidence: number | undefined

  if (selection.kind === 'event') {
    const event = world.events.find((e) => e.id === selection.eventId)
    if (!event) return null
    speech = event.payload.speech as Utterance | undefined
    speakerId = event.actorIds[0]
    listenerName = event.targetIds[0] ? world.agents[event.targetIds[0]]?.name : undefined
    context = `Day ${event.day}, ${event.phase} · ${event.type.replace(/_/g, ' ')} · ${
      event.locationId ? world.places[event.locationId]?.name : 'somewhere'
    }`
    if (event.type === 'shared_rumor') {
      receptionFactors = event.payload.beliefFactors as string[] | undefined
      receptionConfidence = event.payload.confidence as number | undefined
    }
  } else {
    const agent = world.agents[selection.agentId]
    if (!agent) return null
    speakerId = agent.id
    const hungry = agent.needs.hunger > 72
    speech = {
      text: hungry ? 'So hungry…' : 'Dead on my feet…',
      category: 'status',
      motive: {
        kind: 'desperate',
        summary: hungry
          ? 'Not conversation — a need pressing hard enough to surface.'
          : 'Exhaustion talking; the body wants a bed, not company.',
        reasons: [
          `hunger ${agent.needs.hunger.toFixed(0)}/100`,
          `fatigue ${agent.needs.fatigue.toFixed(0)}/100`,
          `coins ${agent.money}`,
        ],
      },
    }
    context = 'Private thought (derived from needs, not spoken)'
  }

  if (!speech || !speakerId) return null
  const speaker = world.agents[speakerId]
  // Citations are speaking-time snapshots; the live memory adds detail if it
  // still exists (it may have faded since — that, too, is information).
  const citedMemory = speech.citedMemory
  const liveMemory = citedMemory ? world.memories[citedMemory.id] : undefined
  const citedBelief = speech.citedBelief
  const liveBelief = citedBelief ? world.beliefs[citedBelief.id] : undefined
  const memoryOriginEvent = liveMemory?.sourceEventIds[0]
    ? world.events.find((e) => e.id === liveMemory.sourceEventIds[0])
    : undefined

  return (
    <div className="phrase-card">
      <div className="phrase-card-head">
        <span
          className="phrase-quote"
          style={{ background: SPEECH_COLORS[speech.category] ?? SPEECH_COLORS.chat }}
        >
          “{speech.text}”
        </span>
        <button className="close-btn" onClick={onClose}>
          ✕
        </button>
      </div>
      <div className="phrase-card-row phrase-meta">
        <button className="agent-link" onClick={() => onSelectAgent(speaker.id)}>
          {speaker.name}
        </button>
        {listenerName && <span> → {listenerName}</span>}
        <span className="dim-text"> · {context}</span>
      </div>

      <div className="phrase-section-title">Motivation</div>
      <div className="phrase-card-row">
        <span className={`motive-chip motive-${speech.motive.kind}`}>
          {MOTIVE_LABEL[speech.motive.kind] ?? speech.motive.kind}
        </span>
        <span className="phrase-summary">{speech.motive.summary}</span>
      </div>
      <ul className="phrase-reasons">
        {speech.motive.reasons.map((reason, i) => (
          <li key={i}>{reason}</li>
        ))}
      </ul>

      {citedMemory && (
        <>
          <div className="phrase-section-title">Historical context</div>
          <div className="phrase-citation">
            <span className="citation-quote">“{citedMemory.summary}”</span>
            <span className="citation-meta">
              formed day {citedMemory.day}
              {world.day > citedMemory.day
                ? ` (${world.day - citedMemory.day} day${world.day - citedMemory.day === 1 ? '' : 's'} ago)`
                : ''}
              {liveMemory && liveMemory.timesRecalled > 0
                ? ` · dwelt on ×${liveMemory.timesRecalled}`
                : ''}
              {memoryOriginEvent
                ? ` · from: ${memoryOriginEvent.type.replace(/_/g, ' ')} on day ${memoryOriginEvent.day}, ${memoryOriginEvent.phase}`
                : ''}
              {!liveMemory ? ' · since faded from memory' : ''}
            </span>
          </div>
        </>
      )}

      {citedBelief && (
        <>
          <div className="phrase-section-title">Influencing knowledge</div>
          <div className="phrase-citation">
            <span className="citation-quote">“{citedBelief.claim}”</span>
            <span className="citation-meta">
              {Math.round(citedBelief.confidence * 100)}% convinced at the time · source: {citedBelief.source}
              {liveBelief && liveBelief.heardFromIds.length > 0
                ? ` · heard from ${liveBelief.heardFromIds
                    .map((id) => world.agents[id]?.name.split(' ')[0] ?? '?')
                    .join(', ')}`
                : ''}
              {liveBelief && Math.round(liveBelief.confidence * 100) !== Math.round(citedBelief.confidence * 100)
                ? ` · now ${Math.round(liveBelief.confidence * 100)}%`
                : ''}
            </span>
          </div>
        </>
      )}

      {receptionFactors && receptionFactors.length > 0 && (
        <>
          <div className="phrase-section-title">
            How it landed{receptionConfidence !== undefined ? ` (${Math.round(receptionConfidence * 100)}% convinced)` : ''}
          </div>
          <ul className="phrase-reasons">
            {receptionFactors.map((factor, i) => (
              <li key={i}>{factor}</li>
            ))}
          </ul>
        </>
      )}

      <button className="phrase-trace-btn" onClick={() => onSelectAgent(speaker.id)}>
        Open {speaker.name.split(' ')[0]}’s decision inspector →
      </button>
    </div>
  )
}
