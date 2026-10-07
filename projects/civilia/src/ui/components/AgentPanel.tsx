import type { BeliefSource, MemoryTone, World } from '../../sim'
import { DecisionInspector } from './DecisionInspector'

const SOURCE_ICON: Record<BeliefSource, string> = {
  witnessed: '👁',
  rumor: '🗣',
  inference: '💭',
}

const TRUTH_MARK: Record<'true' | 'false' | 'unknown', { mark: string; title: string }> = {
  true: { mark: '✓', title: 'canonically true' },
  false: { mark: '✗', title: 'canonically false' },
  unknown: { mark: '?', title: 'truth drifted or unverifiable' },
}

const TONE_EMOJI: Record<MemoryTone, string> = {
  warm: '💛',
  proud: '✨',
  angry: '🔥',
  ashamed: '😳',
  afraid: '😨',
  worried: '🌫',
  bitter: '🥀',
}

interface Props {
  world: World
  agentId: string | null
}

function Bar({ label, value, max = 100, tone }: { label: string; value: number; max?: number; tone: string }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div className="bar">
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div className={`bar-fill tone-${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="bar-value">{value.toFixed(0)}</span>
    </div>
  )
}

export function AgentPanel({ world, agentId }: Props) {
  const agent = agentId ? world.agents[agentId] : null
  if (!agent) return <div className="agent-panel empty">Select an agent to inspect them.</div>

  const household = world.households[agent.householdId]
  const relationships = Object.entries(agent.relationships)
    .map(([otherId, rel]) => ({ other: world.agents[otherId], rel }))
    .filter((r) => r.other)
    .sort(
      (a, b) =>
        b.rel.affection + b.rel.resentment + b.rel.trust - (a.rel.affection + a.rel.resentment + a.rel.trust),
    )
  const rumors = agent.knownRumorIds.map((id) => world.rumors[id]).filter(Boolean)
  const beliefs = agent.beliefIds
    .map((id) => world.beliefs[id])
    .filter(Boolean)
    .sort((a, b) => b.confidence * b.emotionalCharge - a.confidence * a.emotionalCharge)
  const memories = agent.memoryIds
    .map((id) => world.memories[id])
    .filter(Boolean)
    .sort(
      (a, b) =>
        b.strength * (b.importance + b.emotionalCharge) -
        a.strength * (a.importance + a.emotionalCharge),
    )

  return (
    <div className="agent-panel">
      <h2>{agent.name}</h2>
      <div className="agent-meta">
        {agent.age} years · {agent.jobId ?? 'no job'} · {household?.name} household
        <br />
        at {world.places[agent.locationId]?.name}
      </div>
      <div className="agent-vitals">
        <span>💰 {agent.money} coins</span>
        <span>🍞 {agent.breadInventory} bread</span>
        <span>⭐ status {agent.status.toFixed(0)}</span>
      </div>

      <h3>Needs</h3>
      <Bar label="hunger" value={agent.needs.hunger} tone={agent.needs.hunger > 65 ? 'bad' : 'warn'} />
      <Bar label="fatigue" value={agent.needs.fatigue} tone={agent.needs.fatigue > 65 ? 'bad' : 'warn'} />
      <Bar label="belonging" value={agent.needs.belonging} tone="good" />

      <h3>Traits</h3>
      <div className="traits">
        {Object.entries(agent.traits).map(([key, value]) => (
          <Bar key={key} label={key} value={value * 100} tone="neutral" />
        ))}
      </div>

      <h3>Relationships</h3>
      <ul className="rel-list">
        {relationships.slice(0, 9).map(({ other, rel }) => (
          <li key={other.id} className="rel-row">
            <span className="rel-name">
              {other.name}
              {rel.kinship === 'household' ? ' 🏠' : ''}
            </span>
            <span className="rel-dims">
              <span className="dim trust" title="trust">
                t {(rel.trust * 100).toFixed(0)}
              </span>
              <span className="dim affection" title="affection">
                ♥ {(rel.affection * 100).toFixed(0)}
              </span>
              <span className="dim resentment" title="resentment">
                ⚡ {(rel.resentment * 100).toFixed(0)}
              </span>
            </span>
          </li>
        ))}
      </ul>

      <h3>What they remember ({memories.length})</h3>
      <ul className="memory-list">
        {memories.slice(0, 8).map((memory) => (
          <li key={memory.id} className="memory-row">
            <span className="memory-tone" title={memory.tone}>
              {TONE_EMOJI[memory.tone]}
            </span>
            <span className="memory-body">
              “{memory.summary}”
              <span className="memory-meta">
                day {memory.createdDay} · {memory.kind}
                {memory.timesRecalled > 0 ? ` · dwelt on ×${memory.timesRecalled}` : ''}
              </span>
              <span className="memory-strength-track">
                <span
                  className="memory-strength-fill"
                  style={{ width: `${Math.round(memory.strength * 100)}%` }}
                />
              </span>
            </span>
          </li>
        ))}
        {memories.length === 0 && (
          <li className="memory-row none">Nothing has marked them yet.</li>
        )}
      </ul>

      <h3>What they believe ({beliefs.length})</h3>
      <ul className="belief-list">
        {beliefs.slice(0, 7).map((belief) => (
          <li key={belief.id} className="belief-row">
            <span className="belief-source" title={`source: ${belief.source}`}>
              {SOURCE_ICON[belief.source]}
            </span>
            <span className="belief-body">
              “{belief.claim}”
              <span className="belief-meta">
                <span
                  className={`belief-truth truth-${belief.truthStatus}`}
                  title={TRUTH_MARK[belief.truthStatus].title}
                >
                  {TRUTH_MARK[belief.truthStatus].mark}
                </span>
                {Math.round(belief.confidence * 100)}% sure
                {belief.heardFromIds.length > 0 &&
                  ` · heard from ${belief.heardFromIds
                    .map((id) => world.agents[id]?.name.split(' ')[0] ?? '?')
                    .join(', ')}`}
              </span>
              <span className="belief-confidence-track">
                <span
                  className="belief-confidence-fill"
                  style={{ width: `${Math.round(belief.confidence * 100)}%` }}
                />
              </span>
            </span>
          </li>
        ))}
        {beliefs.length === 0 && (
          <li className="belief-row none">They take the town at face value, so far.</li>
        )}
      </ul>

      <h3>Rumors they carry ({rumors.length})</h3>
      <ul className="rumor-list">
        {rumors.slice(-5).reverse().map((rumor) => (
          <li key={rumor.id} className="rumor-row">
            “{rumor.currentClaim}”
            <span className="rumor-meta">
              {rumor.believerIds.includes(agent.id) ? 'believes it' : 'doubts it'} · drift{' '}
              {rumor.truthDistance.toFixed(2)}
            </span>
          </li>
        ))}
        {rumors.length === 0 && <li className="rumor-row none">They have heard nothing worth repeating.</li>}
      </ul>

      <DecisionInspector world={world} agentId={agent.id} />
    </div>
  )
}
