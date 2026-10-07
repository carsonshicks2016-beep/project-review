import type { Agent, World } from '../../sim'
import { relationshipWith } from '../../sim/relationships'
import { householdColor } from '../palette'

interface Props {
  world: World
  selectedAgentId: string | null
  onSelectAgent: (id: string) => void
}

const VIEW_W = 880
const VIEW_H = 620
const CX = VIEW_W / 2
const CY = VIEW_H / 2
const RADIUS = 240

/**
 * The town's social web: agents around a circle (clustered by household),
 * with edges for the strongest bonds, grudges, and open debts. Click a
 * villager to open their drawer.
 */
export function SocialGraph({ world, selectedAgentId, onSelectAgent }: Props) {
  const agents = Object.values(world.agents).sort(
    (a, b) => a.householdId.localeCompare(b.householdId) || a.id.localeCompare(b.id),
  )
  const pos = new Map<string, { x: number; y: number }>()
  agents.forEach((agent, i) => {
    const angle = (i / agents.length) * Math.PI * 2 - Math.PI / 2
    pos.set(agent.id, { x: CX + Math.cos(angle) * RADIUS, y: CY + Math.sin(angle) * RADIUS })
  })

  interface Edge {
    a: Agent
    b: Agent
    kind: 'kinship' | 'affection' | 'resentment' | 'debt'
    strength: number
    title: string
  }
  const edges: Edge[] = []
  for (let i = 0; i < agents.length; i++) {
    for (let j = i + 1; j < agents.length; j++) {
      const a = agents[i]
      const b = agents[j]
      const ab = relationshipWith(a, b.id)
      const ba = relationshipWith(b, a.id)
      const fa = a.name.split(' ')[0]
      const fb = b.name.split(' ')[0]
      const detail =
        `${fa}→${fb}: trust ${(ab.trust * 100).toFixed(0)}%, affection ${(ab.affection * 100).toFixed(0)}%, ` +
        `resentment ${(ab.resentment * 100).toFixed(0)}%${ab.grievances ? `, ${ab.grievances} grievance(s)` : ''}\n` +
        `${fb}→${fa}: trust ${(ba.trust * 100).toFixed(0)}%, affection ${(ba.affection * 100).toFixed(0)}%, ` +
        `resentment ${(ba.resentment * 100).toFixed(0)}%${ba.grievances ? `, ${ba.grievances} grievance(s)` : ''}`

      if (ab.kinship === 'household') {
        edges.push({ a, b, kind: 'kinship', strength: 0.5, title: `Same household\n${detail}` })
      }
      const affection = (ab.affection + ba.affection) / 2
      if (affection >= 0.45) {
        edges.push({ a, b, kind: 'affection', strength: affection, title: detail })
      }
      const resentment = Math.max(ab.resentment, ba.resentment)
      if (resentment >= 0.3) {
        edges.push({ a, b, kind: 'resentment', strength: resentment, title: detail })
      }
      if (ab.debt >= 2 || ba.debt >= 2) {
        const owes = ab.debt >= ba.debt ? `${fa} owes ${fb} ${ab.debt} coins` : `${fb} owes ${fa} ${ba.debt} coins`
        edges.push({ a, b, kind: 'debt', strength: Math.max(ab.debt, ba.debt) / 8, title: `${owes}\n${detail}` })
      }
    }
  }
  const order: Edge['kind'][] = ['kinship', 'affection', 'resentment', 'debt']
  edges.sort((x, y) => order.indexOf(x.kind) - order.indexOf(y.kind))

  const EDGE_STYLE: Record<Edge['kind'], { stroke: string; dash?: string }> = {
    kinship: { stroke: 'rgba(200, 200, 215, 0.25)' },
    affection: { stroke: 'rgba(110, 200, 130, 0.8)' },
    resentment: { stroke: 'rgba(228, 86, 74, 0.85)' },
    debt: { stroke: 'rgba(217, 164, 74, 0.9)', dash: '5 4' },
  }

  return (
    <div className="social-graph-pane">
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="social-graph">
        {edges.map((edge, i) => {
          const pa = pos.get(edge.a.id)!
          const pb = pos.get(edge.b.id)!
          const style = EDGE_STYLE[edge.kind]
          return (
            <line
              key={i}
              x1={pa.x}
              y1={pa.y}
              x2={pb.x}
              y2={pb.y}
              stroke={style.stroke}
              strokeWidth={edge.kind === 'kinship' ? 5 : 1 + Math.min(3, edge.strength * 3.2)}
              strokeDasharray={style.dash}
            >
              <title>{edge.title}</title>
            </line>
          )
        })}
        {agents.map((agent) => {
          const p = pos.get(agent.id)!
          const selected = agent.id === selectedAgentId
          return (
            <g
              key={agent.id}
              transform={`translate(${p.x}, ${p.y})`}
              className="social-node"
              onClick={() => onSelectAgent(agent.id)}
            >
              {selected && <circle r={16} fill="none" stroke="#fff" strokeDasharray="3 2" />}
              <circle r={11} fill={householdColor(world, agent.householdId)} stroke="#161310" strokeWidth={1.5} />
              <text y={26} textAnchor="middle" className="social-label">
                {agent.name.split(' ')[0]}
              </text>
              <title>{`${agent.name} · ${world.households[agent.householdId]?.name ?? ''}`}</title>
            </g>
          )
        })}
      </svg>
      <div className="social-legend">
        <span><i className="legend-line legend-kin" /> household</span>
        <span><i className="legend-line legend-aff" /> affection</span>
        <span><i className="legend-line legend-res" /> resentment</span>
        <span><i className="legend-line legend-debt" /> debt owed</span>
        <span className="dim-text">hover an edge for the numbers · click a villager to inspect</span>
      </div>
    </div>
  )
}
