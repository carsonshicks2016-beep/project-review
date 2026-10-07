import { useState } from 'react'
import type { World } from '../../sim'

interface Props {
  world: World
  agentId: string
}

/**
 * Shows why the agent chose what it chose: every candidate the decision
 * system scored, the seeded noise, and the reasons behind each score.
 */
export function DecisionInspector({ world, agentId }: Props) {
  const traces = world.decisionTraces[agentId] ?? []
  const [offset, setOffset] = useState(0)
  // Clamp in case the selected agent changed or traces got capped.
  const safeOffset = Math.min(offset, Math.max(0, traces.length - 1))
  const trace = traces[traces.length - 1 - safeOffset]

  if (!trace) {
    return (
      <div className="decision-inspector">
        <h3>Decision inspector</h3>
        <p className="none">No decisions yet — step the simulation.</p>
      </div>
    )
  }

  return (
    <div className="decision-inspector">
      <h3>Decision inspector</h3>
      <div className="trace-nav">
        <button disabled={safeOffset >= traces.length - 1} onClick={() => setOffset(safeOffset + 1)}>
          ← older
        </button>
        <span className="trace-when">
          Day {trace.day}, {trace.phase} at {world.places[trace.locationId]?.name}
        </span>
        <button disabled={safeOffset <= 0} onClick={() => setOffset(safeOffset - 1)}>
          newer →
        </button>
      </div>
      <ol className="candidate-list">
        {trace.candidates.map((c, i) => (
          <li key={i} className={`candidate ${i === trace.chosenIndex ? 'chosen' : ''}`}>
            <div className="candidate-head">
              <span className="candidate-label">
                {i === trace.chosenIndex ? '➤ ' : ''}
                {c.label}
              </span>
              <span className="candidate-score" title={`utility ${c.score.toFixed(1)} + noise ${c.noise.toFixed(1)}`}>
                {c.finalScore.toFixed(1)}
              </span>
            </div>
            <div className="candidate-reasons">
              {c.reasons.join(' · ')} · noise {c.noise >= 0 ? '+' : ''}
              {c.noise.toFixed(1)}
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
