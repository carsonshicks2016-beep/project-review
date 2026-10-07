import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Agent, Building, Place, SimEvent, World } from '../../sim'
import {
  TERRAIN_H,
  TERRAIN_W,
  WEAR_PATH,
  WEAR_ROAD,
  WEAR_TRAIL,
  currentPhase,
  idx,
} from '../../sim'
import type { Utterance } from '../../sim'
import {
  BIOME_SHADES,
  PHASE_TINT,
  SPEECH_COLORS,
  TILE,
  TREE_DARK,
  TREE_DARKER,
  WEAR_COLORS,
  householdColor,
} from '../palette'
import type { PhraseSelection } from './PhraseCard'

const W = TERRAIN_W * TILE
const H = TERRAIN_H * TILE

export type ZoomTier = 'region' | 'settlement' | 'human'
const tierFor = (zoom: number): ZoomTier =>
  zoom < 1.6 ? 'region' : zoom < 3.4 ? 'settlement' : 'human'
const TIER_LABEL: Record<ZoomTier, string> = {
  region: 'Region — biomes & claims',
  settlement: 'Settlement — buildings & roads',
  human: 'Street — people & talk',
}

interface Props {
  world: World
  version: number
  recentEvents: SimEvent[]
  selectedAgentId: string | null
  onSelectAgent: (id: string | null) => void
  onSelectPlace: (id: string | null) => void
  onSelectPhrase: (selection: PhraseSelection | null) => void
}

/** Deterministic per-tile texture jitter without touching the sim's rng. */
const jitter = (x: number, y: number, n: number): number =>
  Math.abs((x * 73856093) ^ (y * 19349663)) % n

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

// ---------------------------------------------------------------------------
// Canvas painters

function paintTerrain(ctx: CanvasRenderingContext2D, world: World): void {
  const { terrain, pathWear } = world
  for (let y = 0; y < TERRAIN_H; y++) {
    for (let x = 0; x < TERRAIN_W; x++) {
      const biome = terrain.biome[idx(x, y)]
      const shades = BIOME_SHADES[biome]
      ctx.fillStyle = shades[jitter(x, y, shades.length)]
      ctx.fillRect(x * TILE, y * TILE, TILE, TILE)
    }
  }
  // Worn ground under everything that stands.
  for (const key of Object.keys(pathWear)) {
    const wear = pathWear[key]
    if (wear < WEAR_TRAIL) continue
    const [x, y] = key.split(',').map(Number)
    if (wear >= WEAR_ROAD) {
      ctx.fillStyle = WEAR_COLORS.road
      ctx.fillRect(x * TILE, y * TILE, TILE, TILE)
      ctx.fillStyle = 'rgba(255,235,200,0.18)'
      ctx.fillRect(x * TILE + 1, y * TILE + 1, TILE - 2, 1)
    } else if (wear >= WEAR_PATH) {
      ctx.fillStyle = WEAR_COLORS.path
      ctx.fillRect(x * TILE + 1, y * TILE + 1, TILE - 2, TILE - 2)
    } else {
      ctx.fillStyle = WEAR_COLORS.trail
      ctx.fillRect(x * TILE + 2, y * TILE + 2, TILE - 4, TILE - 4)
    }
  }
  // Trees and peaks stamp over the ground.
  for (let y = 0; y < TERRAIN_H; y++) {
    for (let x = 0; x < TERRAIN_W; x++) {
      const biome = terrain.biome[idx(x, y)]
      const px = x * TILE
      const py = y * TILE
      if (biome === 'forest' || biome === 'dense_forest') {
        const j = jitter(x, y, 3) - 1
        stampTree(ctx, px + 3 + j, py + 2, biome === 'dense_forest')
        if (biome === 'dense_forest') stampTree(ctx, px + 6, py + 5, true)
      } else if (biome === 'mountain' || biome === 'snow') {
        stampPeak(ctx, px, py, biome === 'snow')
      } else if (biome === 'marsh') {
        ctx.fillStyle = '#87a05e'
        ctx.fillRect(px + 2 + jitter(x, y, 4), py + 5, 1, 4)
        ctx.fillRect(px + 6, py + 4 + jitter(y, x, 3), 1, 4)
      }
    }
  }
}

function stampTree(ctx: CanvasRenderingContext2D, px: number, py: number, dark: boolean): void {
  ctx.fillStyle = dark ? TREE_DARKER : TREE_DARK
  ctx.fillRect(px + 1, py, 3, 2)
  ctx.fillRect(px, py + 2, 5, 3)
  ctx.fillStyle = '#3a2a18'
  ctx.fillRect(px + 2, py + 5, 1, 2)
}

function stampPeak(ctx: CanvasRenderingContext2D, px: number, py: number, snow: boolean): void {
  ctx.fillStyle = snow ? '#c9d2da' : '#75757c'
  ctx.fillRect(px + 4, py + 1, 2, 2)
  ctx.fillRect(px + 3, py + 3, 4, 2)
  ctx.fillRect(px + 2, py + 5, 6, 2)
  if (snow) {
    ctx.fillStyle = '#f4f8fb'
    ctx.fillRect(px + 4, py + 1, 2, 2)
  }
}

function paintClaims(ctx: CanvasRenderingContext2D, world: World): void {
  ctx.clearRect(0, 0, W, H)
  const claims = world.claims
  if (!claims) return
  for (let y = 0; y < TERRAIN_H; y++) {
    for (let x = 0; x < TERRAIN_W; x++) {
      const i = idx(x, y)
      const owner = claims.owner[i]
      if (!owner) continue
      const color = householdColor(world, owner)
      ctx.fillStyle = hexToRgba(color, claims.disputed[i] ? 0.16 : 0.22)
      ctx.fillRect(x * TILE, y * TILE, TILE, TILE)
      if (claims.disputed[i]) {
        ctx.fillStyle = 'rgba(225, 60, 50, 0.55)'
        for (let s = 0; s < TILE; s += 4) {
          ctx.fillRect(x * TILE + s, y * TILE + s, 2, 2)
        }
      }
      // Hard edge where the claim meets different ground.
      ctx.fillStyle = hexToRgba(color, 0.85)
      if (x === 0 || claims.owner[idx(x - 1, y)] !== owner) ctx.fillRect(x * TILE, y * TILE, 1, TILE)
      if (x === TERRAIN_W - 1 || claims.owner[idx(x + 1, y)] !== owner)
        ctx.fillRect(x * TILE + TILE - 1, y * TILE, 1, TILE)
      if (y === 0 || claims.owner[idx(x, y - 1)] !== owner) ctx.fillRect(x * TILE, y * TILE, TILE, 1)
      if (y === TERRAIN_H - 1 || claims.owner[idx(x, y + 1)] !== owner)
        ctx.fillRect(x * TILE, y * TILE + TILE - 1, TILE, 1)
    }
  }
}

// ---------------------------------------------------------------------------
// Building sprites (pixel rect compositions)

interface BuildingSpec {
  w: number
  h: number
  body: string
  roof: string
}

const BUILDING_SPECS: Record<Building['kind'], BuildingSpec> = {
  house: { w: 16, h: 11, body: '#d8c9a8', roof: '#b5482f' },
  cottage: { w: 13, h: 9, body: '#c9b591', roof: '#c9a64e' },
  farmhouse: { w: 18, h: 11, body: '#b59a72', roof: '#c9a64e' },
  bakery: { w: 20, h: 12, body: '#d8c9a8', roof: '#9c5a3c' },
  market_stalls: { w: 18, h: 8, body: '#8a5a33', roof: '#d9a44a' },
  well: { w: 8, h: 7, body: '#9aa0a6', roof: '#8a5a33' },
  tavern: { w: 22, h: 13, body: '#a98a62', roof: '#7d4631' },
  granary: { w: 16, h: 14, body: '#b59a72', roof: '#8a8f6a' },
}

function BuildingSprite({
  world,
  building,
  place,
  litWindows,
  selected,
  onClick,
}: {
  world: World
  building: Building
  place: Place
  litWindows: boolean
  selected: boolean
  onClick: () => void
}) {
  const spec = BUILDING_SPECS[building.kind]
  const anchorX = place.tile.x * TILE + TILE / 2
  const anchorY = place.tile.y * TILE + TILE
  const x = -spec.w / 2
  const y = -spec.h
  const roofH = Math.max(4, Math.floor(spec.h * 0.5))
  const ownerColor = building.ownerHouseholdId
    ? householdColor(world, building.ownerHouseholdId)
    : null

  let bodyContent: JSX.Element
  if (building.state === 'planned') {
    bodyContent = (
      <g opacity={0.6}>
        <rect x={x} y={y} width={spec.w} height={spec.h} fill="none" stroke="#f6f1e3" strokeWidth={1} strokeDasharray="3 2" />
        <rect x={x + 2} y={y + spec.h - 3} width={2} height={3} fill="#8a5a33" />
        <rect x={x + 1} y={y + spec.h - 6} width={5} height={3} fill="#d8c9a8" stroke="#2e2118" strokeWidth={0.5} />
      </g>
    )
  } else if (building.state === 'under_construction') {
    bodyContent = (
      <g>
        <rect x={x} y={y + spec.h - 3} width={spec.w} height={3} fill="#9aa0a6" stroke="#2e2118" strokeWidth={0.6} />
        <rect x={x + 1} y={y} width={2} height={spec.h - 3} fill="#8a5a33" />
        <rect x={x + spec.w - 3} y={y} width={2} height={spec.h - 3} fill="#8a5a33" />
        <rect x={x} y={y + 2} width={spec.w} height={1.4} fill="#a87a48" />
        <rect x={x} y={y + spec.h * 0.55} width={spec.w} height={1.4} fill="#a87a48" />
        <rect x={x} y={y - 5} width={spec.w} height={2.6} fill="#2e2118" />
        <rect x={x + 0.4} y={y - 4.6} width={(spec.w - 0.8) * building.progress} height={1.8} fill="#d9a44a" />
      </g>
    )
  } else if (building.kind === 'well') {
    bodyContent = (
      <g>
        <rect x={x} y={y + 3} width={spec.w} height={4} fill="#9aa0a6" stroke="#2e2118" strokeWidth={0.6} />
        <rect x={x + 1} y={y} width={1.4} height={4} fill="#8a5a33" />
        <rect x={x + spec.w - 2.4} y={y} width={1.4} height={4} fill="#8a5a33" />
        <rect x={x + 0.5} y={y - 1.4} width={spec.w - 1} height={2} fill="#7d4631" />
      </g>
    )
  } else if (building.kind === 'market_stalls') {
    bodyContent = (
      <g>
        {[0, 10].map((off) => (
          <g key={off}>
            <rect x={x + off} y={y + 2} width={8} height={6} fill="none" stroke="#2e2118" strokeWidth={0.6} />
            <rect x={x + off + 0.6} y={y + 2.5} width={1.2} height={5} fill={spec.body} />
            <rect x={x + off + 6.2} y={y + 2.5} width={1.2} height={5} fill={spec.body} />
            <rect x={x + off - 0.6} y={y} width={9.2} height={2.6} fill={spec.roof} stroke="#2e2118" strokeWidth={0.6} />
          </g>
        ))}
      </g>
    )
  } else {
    bodyContent = (
      <g>
        <rect x={x} y={y + roofH - 1} width={spec.w} height={spec.h - roofH + 1} fill={spec.body} stroke="#2e2118" strokeWidth={0.7} />
        <polygon
          points={`${x - 1},${y + roofH} ${x + spec.w / 2},${y - 2} ${x + spec.w + 1},${y + roofH}`}
          fill={spec.roof}
          stroke="#2e2118"
          strokeWidth={0.7}
        />
        <rect x={x + spec.w / 2 - 1.6} y={y + spec.h - 5} width={3.2} height={5} fill="#5a3a22" />
        <rect
          x={x + 2}
          y={y + roofH + 1}
          width={3}
          height={3}
          fill={litWindows ? '#ffd87a' : '#3a3f4a'}
          stroke="#2e2118"
          strokeWidth={0.5}
        />
        {spec.w >= 16 && (
          <rect
            x={x + spec.w - 5}
            y={y + roofH + 1}
            width={3}
            height={3}
            fill={litWindows ? '#ffd87a' : '#3a3f4a'}
            stroke="#2e2118"
            strokeWidth={0.5}
          />
        )}
        {building.kind === 'bakery' && (
          <rect x={x + spec.w - 5} y={y - 4} width={2.4} height={5} fill="#9aa0a6" stroke="#2e2118" strokeWidth={0.5} />
        )}
      </g>
    )
  }

  return (
    <g
      className="wm-building"
      transform={`translate(${anchorX}, ${anchorY})`}
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
    >
      {selected && (
        <rect x={x - 2} y={y - 8} width={spec.w + 4} height={spec.h + 10} fill="none" stroke="#fff" strokeWidth={1} strokeDasharray="2 2" />
      )}
      {ownerColor && building.state === 'complete' && (
        <rect x={x} y={y - 4 < y ? y - 0.5 : y} width={2.2} height={2.2} fill={ownerColor} stroke="#2e2118" strokeWidth={0.4} />
      )}
      {bodyContent}
    </g>
  )
}

// ---------------------------------------------------------------------------
// Agent sprites

const YARD_SLOTS: { x: number; y: number }[] = [
  { x: -14, y: 13 },
  { x: 3, y: 17 },
  { x: 16, y: 11 },
  { x: -22, y: 4 },
  { x: 22, y: 3 },
  { x: -6, y: 21 },
  { x: 11, y: 23 },
  { x: -18, y: 19 },
  { x: 20, y: 17 },
  { x: 0, y: 8 },
]

function agentPosition(world: World, agent: Agent, slotIndex: number): { x: number; y: number } {
  const place = world.places[agent.locationId]
  const slot = YARD_SLOTS[slotIndex % YARD_SLOTS.length]
  return {
    x: place.tile.x * TILE + TILE / 2 + slot.x,
    y: place.tile.y * TILE + TILE + slot.y,
  }
}

function AgentSprite({
  world,
  agent,
  pos,
  tier,
  selected,
  onClick,
}: {
  world: World
  agent: Agent
  pos: { x: number; y: number }
  tier: ZoomTier
  selected: boolean
  onClick: () => void
}) {
  const color = householdColor(world, agent.householdId)
  return (
    <g
      className="wm-agent"
      style={{ transform: `translate(${pos.x}px, ${pos.y}px)` }}
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
    >
      {selected && <circle cx={0} cy={-5} r={8} fill="none" stroke="#fff" strokeWidth={1} strokeDasharray="2 2" />}
      {tier === 'human' ? (
        <g>
          <rect x={-1.5} y={-10} width={3} height={3} fill="#e8b58a" stroke="#2e2118" strokeWidth={0.4} />
          <rect x={-2.5} y={-7} width={5} height={5} fill={color} stroke="#2e2118" strokeWidth={0.4} />
          <rect x={-2} y={-2} width={1.6} height={2.6} fill="#4a3826" />
          <rect x={0.4} y={-2} width={1.6} height={2.6} fill="#4a3826" />
        </g>
      ) : (
        <rect x={-2} y={-4} width={4} height={4} fill={color} stroke="#2e2118" strokeWidth={0.5} />
      )}
    </g>
  )
}

// ---------------------------------------------------------------------------
// Speech / thought bubbles

interface BubbleData {
  key: string
  x: number
  y: number
  text: string
  category: string
  selection: PhraseSelection
}

function Bubble({ bubble, zoom, onSelect }: { bubble: BubbleData; zoom: number; onSelect: () => void }) {
  const width = 150
  const thought = bubble.category === 'thought'
  return (
    <g transform={`translate(${bubble.x}, ${bubble.y})`}>
      <g transform={`scale(${1 / zoom})`}>
        <foreignObject x={-width / 2} y={-64} width={width} height={60} style={{ overflow: 'visible' }}>
          <div
            className={`wm-bubble ${thought ? 'wm-bubble-thought' : ''}`}
            style={{ background: SPEECH_COLORS[bubble.category] ?? SPEECH_COLORS.chat }}
            onClick={(e) => {
              e.stopPropagation()
              onSelect()
            }}
          >
            {bubble.text}
          </div>
        </foreignObject>
      </g>
    </g>
  )
}

// ---------------------------------------------------------------------------
// The map

export function WorldMap({
  world,
  version,
  recentEvents,
  selectedAgentId,
  onSelectAgent,
  onSelectPlace,
  onSelectPhrase,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const terrainRef = useRef<HTMLCanvasElement>(null)
  const claimsRef = useRef<HTMLCanvasElement>(null)
  const [camera, setCamera] = useState({ x: 0, y: 0, zoom: 2.4 })
  const [rumorLens, setRumorLens] = useState(false)
  const dragRef = useRef<{ startX: number; startY: number; camX: number; camY: number; moved: boolean } | null>(null)

  const phase = currentPhase(world)
  const tier = tierFor(camera.zoom)
  const night = phase === 'evening' || phase === 'night'

  // Center the camera on the town whenever a new world arrives.
  useLayoutEffect(() => {
    const box = containerRef.current?.getBoundingClientRect()
    if (!box) return
    const zoom = 2.4
    const tc = world.terrain.townCenter
    setCamera({
      x: box.width / 2 - tc.x * TILE * zoom,
      y: box.height / 2 - tc.y * TILE * zoom,
      zoom,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [world.id])

  // Repaint canvases when the sim advances.
  useEffect(() => {
    const terrainCtx = terrainRef.current?.getContext('2d')
    if (terrainCtx) paintTerrain(terrainCtx, world)
    const claimsCtx = claimsRef.current?.getContext('2d')
    if (claimsCtx) paintClaims(claimsCtx, world)
  }, [world, version])

  // Wheel zoom anchored under the cursor (non-passive to prevent page scroll).
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const box = el.getBoundingClientRect()
      const cx = e.clientX - box.left
      const cy = e.clientY - box.top
      setCamera((cam) => {
        const zoom = Math.min(8, Math.max(0.7, cam.zoom * Math.exp(-e.deltaY * 0.0016)))
        const k = zoom / cam.zoom
        return { x: cx - (cx - cam.x) * k, y: cy - (cy - cam.y) * k, zoom }
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  /** Button zoom keeps whatever is mid-screen mid-screen. */
  const zoomAtCenter = (factor: number) => {
    const box = containerRef.current?.getBoundingClientRect()
    if (!box) return
    setCamera((cam) => {
      const zoom = Math.min(8, Math.max(0.7, cam.zoom * factor))
      const k = zoom / cam.zoom
      const cx = box.width / 2
      const cy = box.height / 2
      return { x: cx - (cx - cam.x) * k, y: cy - (cy - cam.y) * k, zoom }
    })
  }

  const onPointerDown = (e: React.PointerEvent) => {
    dragRef.current = { startX: e.clientX, startY: e.clientY, camX: camera.x, camY: camera.y, moved: false }
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current
    if (!drag) return
    const dx = e.clientX - drag.startX
    const dy = e.clientY - drag.startY
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true
    if (drag.moved) setCamera((cam) => ({ ...cam, x: drag.camX + dx, y: drag.camY + dy }))
  }
  const onPointerUp = () => {
    const moved = dragRef.current?.moved
    dragRef.current = null
    if (!moved) {
      onSelectAgent(null)
      onSelectPlace(null)
      onSelectPhrase(null)
    }
  }

  // Outdoor agents with stable yard slots; indoor headcounts per place.
  const outdoorByPlace = new Map<string, Agent[]>()
  const indoorCount = new Map<string, number>()
  for (const agent of Object.values(world.agents)) {
    if (agent.indoors) {
      indoorCount.set(agent.locationId, (indoorCount.get(agent.locationId) ?? 0) + 1)
    } else {
      const list = outdoorByPlace.get(agent.locationId) ?? []
      list.push(agent)
      outdoorByPlace.set(agent.locationId, list)
    }
  }
  const agentPos = new Map<string, { x: number; y: number }>()
  for (const [, list] of outdoorByPlace) {
    list.sort((a, b) => a.id.localeCompare(b.id))
    list.forEach((agent, i) => agentPos.set(agent.id, agentPosition(world, agent, i)))
  }

  // Bubbles: the latest spoken line per speaker from the freshest events.
  const bubbles: BubbleData[] = []
  if (tier === 'human') {
    const seen = new Set<string>()
    for (let i = recentEvents.length - 1; i >= 0 && bubbles.length < 8; i--) {
      const event = recentEvents[i]
      const speech = event.payload.speech as Utterance | undefined
      if (!speech) continue
      const speakerId = event.actorIds[0]
      if (!speakerId || seen.has(speakerId)) continue
      seen.add(speakerId)
      const speaker = world.agents[speakerId]
      if (!speaker) continue
      const pos =
        agentPos.get(speakerId) ??
        (() => {
          const t = world.places[speaker.locationId].tile
          return { x: t.x * TILE + TILE / 2, y: t.y * TILE - 4 }
        })()
      bubbles.push({
        key: event.id,
        x: pos.x,
        y: pos.y - 12,
        text: speech.text,
        category: speech.category,
        selection: { kind: 'event', eventId: event.id },
      })
    }
    // Thought bubbles for pressing needs (presentation-derived, never stored).
    for (const [id, pos] of agentPos) {
      if (bubbles.length >= 10) break
      if (bubbles.some((b) => b.selection.kind === 'event' && world.events.find((e) => e.id === (b.selection as { eventId: string }).eventId)?.actorIds[0] === id)) continue
      const agent = world.agents[id]
      if (agent.needs.hunger > 72) {
        const stocked = world.places['place_bakery'].foodStock > 0
        bubbles.push({
          key: `thought_${id}`,
          x: pos.x,
          y: pos.y - 12,
          text: stocked ? 'So hungry… the bakery smells cruel today.' : 'So hungry… and the shelves are bare.',
          category: 'thought',
          selection: { kind: 'thought', agentId: id },
        })
      } else if (agent.needs.fatigue > 82) {
        bubbles.push({
          key: `thought_${id}`,
          x: pos.x,
          y: pos.y - 12,
          text: 'Dead on my feet…',
          category: 'thought',
          selection: { kind: 'thought', agentId: id },
        })
      }
    }
  }

  const claimsOpacity = tier === 'region' ? 0.95 : tier === 'settlement' ? 0.35 : 0.15

  /** Current world position for any agent: yard slot outdoors, building anchor indoors. */
  const anyAgentPos = (agentId: string): { x: number; y: number } => {
    const fromYard = agentPos.get(agentId)
    if (fromYard) return fromYard
    const agent = world.agents[agentId]
    const tile = world.places[agent.locationId].tile
    return { x: tile.x * TILE + TILE / 2, y: tile.y * TILE + TILE / 2 }
  }

  // Rumor River: recent tellings rendered as glowing arcs with particles.
  interface RumorArc {
    key: string
    d: string
    mutated: boolean
    charge: number
  }
  const rumorArcs: RumorArc[] = []
  if (rumorLens && tier !== 'region') {
    const tellings = world.events
      .filter((e) => e.type === 'shared_rumor' && e.day >= world.day - 2)
      .slice(-14)
    for (const telling of tellings) {
      const from = telling.actorIds[0]
      const to = telling.targetIds[0]
      if (!from || !to || !world.agents[from] || !world.agents[to]) continue
      const a = anyAgentPos(from)
      const b = anyAgentPos(to)
      const mx = (a.x + b.x) / 2
      const my = (a.y + b.y) / 2 - Math.max(14, Math.hypot(b.x - a.x, b.y - a.y) * 0.3)
      rumorArcs.push({
        key: telling.id,
        d: `M ${a.x} ${a.y - 6} Q ${mx} ${my} ${b.x} ${b.y - 6}`,
        mutated: (telling.payload as { mutated?: boolean }).mutated === true,
        charge: world.rumors[(telling.payload as { rumorId?: string }).rumorId ?? '']?.emotionalCharge ?? 0.4,
      })
    }
  }

  return (
    <div
      ref={containerRef}
      className={`worldmap ${rumorLens ? 'rumor-lens-on' : ''}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <div
        className="worldmap-inner"
        style={{ transform: `translate(${camera.x}px, ${camera.y}px) scale(${camera.zoom})` }}
      >
        <canvas ref={terrainRef} width={W} height={H} className="wm-canvas" />
        <canvas ref={claimsRef} width={W} height={H} className="wm-canvas" style={{ opacity: claimsOpacity }} />
        <svg className="wm-svg" width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
          {/* Buildings */}
          {Object.values(world.buildings).map((building) => {
            const place = world.places[building.placeId]
            if (!place) return null
            return (
              <BuildingSprite
                key={building.id}
                world={world}
                building={building}
                place={place}
                litWindows={night && (indoorCount.get(place.id) ?? 0) > 0}
                selected={false}
                onClick={() => onSelectPlace(place.id)}
              />
            )
          })}

          {/* Indoor headcount chips */}
          {tier !== 'region' &&
            Object.values(world.places).map((place) => {
              const count = indoorCount.get(place.id) ?? 0
              if (count === 0) return null
              const x = place.tile.x * TILE + TILE / 2
              const y = place.tile.y * TILE - 8
              return (
                <g key={`chip_${place.id}`} transform={`translate(${x}, ${y})`}>
                  <g transform={`scale(${1 / camera.zoom})`}>
                    <g
                      className="wm-chip"
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelectPlace(place.id)
                      }}
                    >
                      <rect x={-16} y={-9} width={32} height={15} rx={4} fill="rgba(20,18,14,0.82)" stroke="#f6f1e3" strokeWidth={0.6} />
                      <text x={0} y={2.5} textAnchor="middle" fontSize={9} fill="#f6f1e3">
                        ⌂ {count}
                      </text>
                    </g>
                  </g>
                </g>
              )
            })}

          {/* Outdoor people */}
          {tier !== 'region' &&
            [...agentPos.entries()].map(([id, pos]) => (
              <AgentSprite
                key={id}
                world={world}
                agent={world.agents[id]}
                pos={pos}
                tier={tier}
                selected={selectedAgentId === id}
                onClick={() => onSelectAgent(id)}
              />
            ))}

          {/* Landmark names at region zoom */}
          {tier === 'region' &&
            world.terrain.landmarks.map((lm) => (
              <g key={`${lm.kind}_${lm.name}`} transform={`translate(${lm.x * TILE}, ${lm.y * TILE})`}>
                <g transform={`scale(${1 / camera.zoom})`}>
                  <text className="wm-landmark" textAnchor="middle">
                    {lm.name}
                  </text>
                </g>
              </g>
            ))}

          {/* Rumor River: who told whom, deceit glowing hotter */}
          {rumorArcs.map((arc) => {
            const color = arc.mutated ? '#e06ad9' : '#a98ae6'
            return (
              <g key={arc.key} className="rumor-arc">
                <path d={arc.d} fill="none" stroke={color} strokeOpacity={0.18} strokeWidth={3.5} />
                <path
                  d={arc.d}
                  fill="none"
                  stroke={color}
                  strokeOpacity={0.75}
                  strokeWidth={1}
                  strokeDasharray="3 4"
                />
                <circle r={1.8 + arc.charge * 1.6} fill={color}>
                  <animateMotion dur={`${2.6 - arc.charge}s`} repeatCount="indefinite" path={arc.d} />
                </circle>
              </g>
            )
          })}

          {/* Talk */}
          {bubbles.map((bubble) => (
            <Bubble key={bubble.key} bubble={bubble} zoom={camera.zoom} onSelect={() => onSelectPhrase(bubble.selection)} />
          ))}
        </svg>
      </div>

      <div className="wm-tint" style={{ background: PHASE_TINT[phase] }} />

      <div
        className="wm-hud"
        onPointerDown={(e) => e.stopPropagation()}
        onPointerUp={(e) => e.stopPropagation()}
      >
        <span className="wm-tier">{TIER_LABEL[tier]}</span>
        <button
          className={rumorLens ? 'lens-active' : ''}
          title="Rumor River: watch gossip travel (purple) and mutate (pink)"
          onClick={() => setRumorLens(!rumorLens)}
        >
          🗣
        </button>
        <button onClick={() => zoomAtCenter(1.45)}>＋</button>
        <button onClick={() => zoomAtCenter(1 / 1.45)}>－</button>
      </div>
    </div>
  )
}
