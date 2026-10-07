import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import {
  Activity,
  Brain,
  Download,
  Flag,
  Gauge,
  Pause,
  Play,
  RefreshCcw,
  Shuffle,
  Trophy,
  Upload,
} from 'lucide-react'
import './App.css'

interface Vec {
  x: number
  y: number
}

interface Segment {
  a: Vec
  b: Vec
  length: number
  cumulative: number
  angle: number
}

interface MovingObstacle {
  id: number
  progress: number
  laneOffset: number
  speed: number
  radius: number
  sway: number
  swaySpeed: number
  phase: number
  color: string
}

interface ObstacleState extends Vec {
  id: number
  progress: number
  laneOffset: number
  radius: number
  heading: number
  color: string
  speed: number
}

interface Track {
  center: Vec[]
  segments: Segment[]
  length: number
  miles: number
  width: number
  worldWidth: number
  worldHeight: number
  start: Vec
  startHeading: number
  seed: number
  name: string
  obstacles: MovingObstacle[]
}

interface ClosestPoint {
  point: Vec
  distance: number
  signedDistance: number
  progress: number
  angle: number
  segmentIndex?: number
}

interface Genome {
  id: number
  weights: number[]
  score: number
}

interface BrainState {
  inputs: number[]
  hidden: number[][]
  outputs: number[]
}

interface Car {
  id: number
  genome: Genome
  x: number
  y: number
  heading: number
  speed: number
  vx: number
  vy: number
  angularVelocity: number
  slip: number
  traction: number
  alive: boolean
  crashed: boolean
  fitness: number
  lap: number
  progress: number
  lastProgress: number
  bestFitness: number
  stagnantTicks: number
  age: number
  sensors: number[]
  brain: BrainState
  color: string
  streak: number
  lastSegmentIndex?: number
}

interface Settings {
  population: number
  trials: number
  saveTop: number
  simSpeed: number
  mutationRate: number
  mutationStrength: number
  randomTrackEachTrial: boolean
  infiniteTraining: boolean
  enableObstacles: boolean
}

interface Stats {
  generation: number
  alive: number
  bestScore: number
  bestEver: number
  averageScore: number
  championLap: number
  championSpeed: number
  championTraction: number
  championSlip: number
  trialComplete: boolean
  selectedCarId: number
  savedBrains: Genome[]
}

const WORLD_WIDTH = 980
const WORLD_HEIGHT = 660
const SENSOR_ANGLES = [-170, -140, -92, -58, -32, -12, 0, 12, 32, 58, 92, 140, 170, 180].map(
  (angle) => (angle * Math.PI) / 180,
)
const SENSOR_RANGE = 168
const LOOKAHEAD_DISTANCES = [90, 220, 420, 760]
const INPUT_COUNT = SENSOR_ANGLES.length + 22
const HIDDEN_LAYERS = [28, 22, 16]
const OUTPUT_COUNT = 3
const NETWORK_LAYERS = [INPUT_COUNT, ...HIDDEN_LAYERS, OUTPUT_COUNT]
const GENOME_LENGTH = NETWORK_LAYERS.slice(1).reduce(
  (total, layerSize, layerIndex) => total + (NETWORK_LAYERS[layerIndex] + 1) * layerSize,
  0,
)
const TARGET_TRACK_MILES = 5
const UNITS_PER_MILE = 1600
const TRACK_MARGIN = 360
const MAX_TICKS = 60 * 120
const MAX_SPEED = 5.7
const MIN_SPEED = -1.1
const MAX_VISUAL_SPEED = MAX_SPEED * 1.16
const OBSTACLE_COLORS = ['#ec6b56', '#2f80aa', '#6c5ce7', '#d99b2b', '#2e9d79']
const CAR_COLORS = [
  '#f25f5c',
  '#247ba0',
  '#70c1b3',
  '#f5b841',
  '#b56576',
  '#4d908e',
  '#f3722c',
  '#577590',
  '#43aa8b',
  '#9d4edd',
]
const STORAGE_KEY = 'neuro-racer-top-brains'

const DEFAULT_SETTINGS: Settings = {
  population: 20,
  trials: 80,
  saveTop: 5,
  simSpeed: 3,
  mutationRate: 0.09,
  mutationStrength: 0.42,
  randomTrackEachTrial: false,
  infiniteTraining: false,
  enableObstacles: true,
}

const INITIAL_STATS: Stats = {
  generation: 1,
  alive: 0,
  bestScore: 0,
  bestEver: 0,
  averageScore: 0,
  championLap: 0,
  championSpeed: 0,
  championTraction: 1,
  championSlip: 0,
  trialComplete: false,
  selectedCarId: 1,
  savedBrains: [],
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function lerp(start: number, end: number, amount: number) {
  return start + (end - start) * amount
}

function wrapAngle(angle: number) {
  let wrapped = angle

  while (wrapped > Math.PI) {
    wrapped -= Math.PI * 2
  }

  while (wrapped < -Math.PI) {
    wrapped += Math.PI * 2
  }

  return wrapped
}

function mulberry32(seed: number) {
  let value = seed >>> 0

  return () => {
    value += 0x6d2b79f5
    let next = value
    next = Math.imul(next ^ (next >>> 15), next | 1)
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61)
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296
  }
}

function randomSeed() {
  return Math.floor(Math.random() * 1_000_000_000)
}

function randomNormal(random = Math.random) {
  const a = Math.max(random(), Number.EPSILON)
  const b = Math.max(random(), Number.EPSILON)
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(Math.PI * 2 * b)
}

function distanceSquared(a: Vec, b: Vec) {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return dx * dx + dy * dy
}

function createSegments(points: Vec[]) {
  const segments: Segment[] = []
  let cumulative = 0

  for (let index = 0; index < points.length; index += 1) {
    const a = points[index]
    const b = points[(index + 1) % points.length]
    const dx = b.x - a.x
    const dy = b.y - a.y
    const length = Math.hypot(dx, dy)

    segments.push({
      a,
      b,
      length,
      cumulative,
      angle: Math.atan2(dy, dx),
    })
    cumulative += length
  }

  return { segments, length: cumulative }
}

function smoothPoints(points: Vec[], passes: number) {
  let smoothed = points

  for (let pass = 0; pass < passes; pass += 1) {
    smoothed = smoothed.map((point, index) => {
      const previous = smoothed[(index - 1 + smoothed.length) % smoothed.length]
      const next = smoothed[(index + 1) % smoothed.length]

      return {
        x: point.x * 0.56 + previous.x * 0.22 + next.x * 0.22,
        y: point.y * 0.56 + previous.y * 0.22 + next.y * 0.22,
      }
    })
  }

  return smoothed
}

function scaleTrackToLength(points: Vec[], targetLength: number) {
  const { length } = createSegments(points)
  const scale = targetLength / Math.max(length, 1)

  return points.map((point) => ({
    x: point.x * scale,
    y: point.y * scale,
  }))
}

function moveTrackIntoWorld(points: Vec[]) {
  const minX = Math.min(...points.map((point) => point.x))
  const maxX = Math.max(...points.map((point) => point.x))
  const minY = Math.min(...points.map((point) => point.y))
  const maxY = Math.max(...points.map((point) => point.y))

  return {
    center: points.map((point) => ({
      x: point.x - minX + TRACK_MARGIN,
      y: point.y - minY + TRACK_MARGIN,
    })),
    worldWidth: maxX - minX + TRACK_MARGIN * 2,
    worldHeight: maxY - minY + TRACK_MARGIN * 2,
  }
}

function createObstacles(random: () => number, randomize: boolean): MovingObstacle[] {
  const count = randomize ? 18 + Math.floor(random() * 10) : 20

  return Array.from({ length: count }, (_, index) => {
    const cluster = Math.floor(index / 4)
    const clusterOffset = (index % 4) * 0.012
    const baseProgress =
      index < 4
        ? 0.16 + index * 0.055
        : (cluster * 0.137 + clusterOffset + random() * 0.045) % 1

    return {
      id: index + 1,
      progress: baseProgress,
      laneOffset: (random() - 0.5) * 58,
      speed: 0.65 + random() * (randomize ? 1.7 : 1.35),
      radius: 12 + random() * 8,
      sway: 10 + random() * (randomize ? 44 : 34),
      swaySpeed: 0.006 + random() * 0.016,
      phase: random() * Math.PI * 2,
      color: OBSTACLE_COLORS[index % OBSTACLE_COLORS.length],
    }
  })
}

function createTrack(seed = 140_529, randomize = false): Track {
  const random = mulberry32(seed)
  const points: Vec[] = []
  const count = 380
  const targetLength = TARGET_TRACK_MILES * UNITS_PER_MILE
  const width = randomize ? 82 + random() * 24 : 94
  const rx = randomize ? 1020 + random() * 390 : 1240
  const ry = randomize ? 660 + random() * 290 : 780
  const wobbleA = randomize ? random() * Math.PI * 2 : 0.7
  const wobbleB = randomize ? random() * Math.PI * 2 : 2.2
  const wobbleC = randomize ? random() * Math.PI * 2 : 4.1
  const technicalPhase = randomize ? random() * Math.PI * 2 : 1.35

  for (let index = 0; index < count; index += 1) {
    const angle = (index / count) * Math.PI * 2
    const bend =
      1 +
      Math.sin(angle * 2 + wobbleA) * (randomize ? 0.14 : 0.10) +
      Math.cos(angle * 3 + wobbleB) * (randomize ? 0.10 : 0.07) +
      Math.sin(angle * 5 + wobbleC) * (randomize ? 0.03 : 0.02)
    const chicane =
      Math.sin(angle * 7 + wobbleB) * (randomize ? 25 : 18)
    const localRx = rx * bend
    const localRy =
      ry *
      (1 +
        Math.cos(angle * 2 + wobbleB) * (randomize ? 0.16 : 0.10) +
        Math.sin(angle * 4 + wobbleA) * (randomize ? 0.08 : 0.05))

    points.push({
      x: Math.cos(angle) * (localRx + chicane),
      y: Math.sin(angle) * (localRy + chicane),
    })
  }

  const longCourse = scaleTrackToLength(smoothPoints(points, randomize ? 4 : 6), targetLength)
  const { center, worldWidth, worldHeight } = moveTrackIntoWorld(longCourse)
  const { segments, length } = createSegments(center)
  const start = center[0]
  const startHeading = segments[0].angle
  const obstacles = createObstacles(random, randomize)

  return {
    center,
    segments,
    length,
    miles: length / UNITS_PER_MILE,
    width,
    worldWidth,
    worldHeight,
    start,
    startHeading,
    seed,
    name: randomize
      ? `Technical 5-mile seed ${seed.toString().slice(-6)}`
      : 'Technical five-mile loop',
    obstacles,
  }
}

function closestPointOnTrack(track: Track, point: Vec, hintIndex?: number): ClosestPoint {
  let bestDistanceSquared = Number.POSITIVE_INFINITY
  let bestPoint = track.start
  let bestProgress = 0
  let bestAngle = track.startHeading
  let bestSignedDistance = 0
  let bestIndex = 0

  const segmentsCount = track.segments.length
  let startIndex = 0
  let endIndex = segmentsCount
  let useLocalSearch = false

  if (hintIndex !== undefined && hintIndex >= 0 && hintIndex < segmentsCount) {
    useLocalSearch = true
    const windowSize = 30
    startIndex = (hintIndex - windowSize + segmentsCount) % segmentsCount
    endIndex = (hintIndex + windowSize) % segmentsCount
  }

  const checkSegment = (idx: number) => {
    const segment = track.segments[idx]
    const abx = segment.b.x - segment.a.x
    const aby = segment.b.y - segment.a.y
    const apx = point.x - segment.a.x
    const apy = point.y - segment.a.y
    const denominator = segment.length * segment.length || 1
    const t = clamp((apx * abx + apy * aby) / denominator, 0, 1)
    const projection = {
      x: segment.a.x + abx * t,
      y: segment.a.y + aby * t,
    }
    const candidateDistanceSquared = distanceSquared(point, projection)

    if (candidateDistanceSquared < bestDistanceSquared) {
      const cross = abx * (point.y - projection.y) - aby * (point.x - projection.x)
      const sign = cross >= 0 ? 1 : -1
      bestDistanceSquared = candidateDistanceSquared
      bestPoint = projection
      bestProgress = (segment.cumulative + t * segment.length) / track.length
      bestAngle = segment.angle
      bestSignedDistance = Math.sqrt(candidateDistanceSquared) * sign
      bestIndex = idx
    }
  }

  if (useLocalSearch) {
    let curr = startIndex
    while (curr !== endIndex) {
      checkSegment(curr)
      curr = (curr + 1) % segmentsCount
    }
    checkSegment(endIndex)
  } else {
    for (let i = 0; i < segmentsCount; i++) {
      checkSegment(i)
    }
  }

  return {
    point: bestPoint,
    distance: Math.sqrt(bestDistanceSquared),
    signedDistance: bestSignedDistance,
    progress: bestProgress,
    angle: bestAngle,
    segmentIndex: bestIndex,
  }
}

function isPointOnTrack(track: Track, point: Vec, hintIndex?: number) {
  return closestPointOnTrack(track, point, hintIndex).distance <= track.width * 0.5 - 5
}

function lerpAngle(a: number, b: number, t: number) {
  let difference = b - a
  while (difference < -Math.PI) difference += Math.PI * 2
  while (difference > Math.PI) difference -= Math.PI * 2
  return a + difference * t
}

function getVertexAngle(track: Track, idx: number) {
  const count = track.segments.length
  const currentAngle = track.segments[idx].angle
  const prevAngle = track.segments[(idx - 1 + count) % count].angle
  return lerpAngle(prevAngle, currentAngle, 0.5)
}

function sampleTrackAtProgress(track: Track, progress: number) {
  const normalized = ((progress % 1) + 1) % 1
  const targetDistance = normalized * track.length
  let selectedIndex = track.segments.length - 1

  for (let i = 0; i < track.segments.length; i += 1) {
    const segment = track.segments[i]
    if (segment.cumulative + segment.length >= targetDistance) {
      selectedIndex = i
      break
    }
  }

  const selected = track.segments[selectedIndex]
  const localDistance = targetDistance - selected.cumulative
  const t = clamp(localDistance / Math.max(selected.length, 1), 0, 1)

  const startAngle = getVertexAngle(track, selectedIndex)
  const endAngle = getVertexAngle(track, (selectedIndex + 1) % track.segments.length)
  const smoothAngle = lerpAngle(startAngle, endAngle, t)

  return {
    point: {
      x: lerp(selected.a.x, selected.b.x, t),
      y: lerp(selected.a.y, selected.b.y, t),
    },
    angle: smoothAngle,
  }
}

function getObstacleState(track: Track, obstacle: MovingObstacle, tick: number): ObstacleState {
  const progress = (obstacle.progress + (tick * obstacle.speed) / track.length) % 1
  const sample = sampleTrackAtProgress(track, progress)
  const normal = {
    x: Math.cos(sample.angle + Math.PI / 2),
    y: Math.sin(sample.angle + Math.PI / 2),
  }
  const laneLimit = track.width * 0.5 - obstacle.radius - 8
  const laneOffset = clamp(
    obstacle.laneOffset + Math.sin(tick * obstacle.swaySpeed + obstacle.phase) * obstacle.sway,
    -laneLimit,
    laneLimit,
  )

  return {
    id: obstacle.id,
    x: sample.point.x + normal.x * laneOffset,
    y: sample.point.y + normal.y * laneOffset,
    progress,
    laneOffset,
    radius: obstacle.radius,
    heading: sample.angle,
    color: obstacle.color,
    speed: obstacle.speed,
  }
}

function getObstacleStates(track: Track, tick: number, enabled: boolean = true) {
  if (!enabled) return []
  return track.obstacles.map((obstacle) => getObstacleState(track, obstacle, tick))
}

function obstacleProgressDelta(from: number, to: number) {
  return ((to - from + 1.5) % 1) - 0.5
}

function pointHitsObstacle(point: Vec, obstacles: ObstacleState[]) {
  return obstacles.some((obstacle) => distanceSquared(point, obstacle) <= obstacle.radius ** 2)
}

function getNearestObstacleInputs(
  track: Track,
  car: Car,
  closest: ClosestPoint,
  obstacles: ObstacleState[],
) {
  const forward = {
    x: Math.cos(car.heading),
    y: Math.sin(car.heading),
  }
  const right = {
    x: Math.cos(car.heading + Math.PI / 2),
    y: Math.sin(car.heading + Math.PI / 2),
  }
  let nearest: ObstacleState | undefined
  let nearestScore = Number.POSITIVE_INFINITY

  for (const obstacle of obstacles) {
    const progressDelta = (obstacle.progress - closest.progress + 1) % 1
    const distanceAhead = progressDelta * track.length
    const dx = obstacle.x - car.x
    const dy = obstacle.y - car.y
    const forwardDistance = dx * forward.x + dy * forward.y

    if (distanceAhead < 900 && forwardDistance > -80) {
      const lateralDistance = Math.abs(dx * right.x + dy * right.y)
      const score = distanceAhead + lateralDistance * 1.5

      if (score < nearestScore) {
        nearest = obstacle
        nearestScore = score
      }
    }
  }

  if (!nearest) {
    return [1, 0, 1, 0]
  }

  const dx = nearest.x - car.x
  const dy = nearest.y - car.y
  const forwardDistance = dx * forward.x + dy * forward.y
  const lateralDistance = dx * right.x + dy * right.y
  const bearing = Math.atan2(lateralDistance, Math.max(forwardDistance, 1))
  const progressDelta = obstacleProgressDelta(closest.progress, nearest.progress)

  return [
    clamp(forwardDistance / 900, 0, 1) * 2 - 1,
    Math.sin(bearing),
    Math.cos(bearing),
    clamp(progressDelta * 8, -1, 1),
  ]
}

function getRearObstacleInputs(
  track: Track,
  car: Car,
  closest: ClosestPoint,
  obstacles: ObstacleState[],
  longitudinalVelocity: number,
) {
  const forward = {
    x: Math.cos(car.heading),
    y: Math.sin(car.heading),
  }
  const right = {
    x: Math.cos(car.heading + Math.PI / 2),
    y: Math.sin(car.heading + Math.PI / 2),
  }
  let nearest: ObstacleState | undefined
  let nearestScore = Number.POSITIVE_INFINITY

  for (const obstacle of obstacles) {
    const progressDelta = obstacleProgressDelta(closest.progress, obstacle.progress)
    const distanceBehind = -progressDelta * track.length
    const dx = obstacle.x - car.x
    const dy = obstacle.y - car.y
    const forwardDistance = dx * forward.x + dy * forward.y

    if (distanceBehind > 0 && distanceBehind < 760 && forwardDistance < 90) {
      const lateralDistance = Math.abs(dx * right.x + dy * right.y)
      const score = distanceBehind + lateralDistance * 1.4

      if (score < nearestScore) {
        nearest = obstacle
        nearestScore = score
      }
    }
  }

  if (!nearest) {
    return [1, 0, 1, -1]
  }

  const dx = nearest.x - car.x
  const dy = nearest.y - car.y
  const rearDistance = Math.max(-(dx * forward.x + dy * forward.y), 1)
  const lateralDistance = dx * right.x + dy * right.y
  const bearing = Math.atan2(lateralDistance, rearDistance)
  const closingSpeed = nearest.speed - Math.max(0, longitudinalVelocity)

  return [
    clamp(rearDistance / 760, 0, 1) * 2 - 1,
    Math.sin(bearing),
    Math.cos(bearing),
    clamp(closingSpeed / 4.2, -1, 1),
  ]
}

function createGenome(id: number, random = Math.random): Genome {
  return {
    id,
    score: 0,
    weights: Array.from({ length: GENOME_LENGTH }, () => randomNormal(random) * 0.36),
  }
}

function runBrain(weights: number[], inputs: number[]): BrainState {
  const hidden: number[][] = []
  let activations = inputs
  let cursor = 0

  for (const layerSize of HIDDEN_LAYERS) {
    const nextActivations: number[] = []

    for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex += 1) {
      let sum = weights[cursor]
      cursor += 1

      for (const activation of activations) {
        sum += activation * weights[cursor]
        cursor += 1
      }

      nextActivations.push(Math.tanh(sum))
    }

    hidden.push(nextActivations)
    activations = nextActivations
  }

  const outputs: number[] = []

  for (let outputIndex = 0; outputIndex < OUTPUT_COUNT; outputIndex += 1) {
    let sum = weights[cursor]
    cursor += 1

    for (const activation of activations) {
      sum += activation * weights[cursor]
      cursor += 1
    }

    outputs.push(Math.tanh(sum))
  }

  return { inputs, hidden, outputs }
}

function crossover(
  parentA: Genome,
  parentB: Genome,
  id: number,
  settings: Settings,
): Genome {
  return {
    id,
    score: 0,
    weights: parentA.weights.map((weight, index) => {
      let next = Math.random() < 0.5 ? weight : parentB.weights[index]

      if (Math.random() < settings.mutationRate) {
        next += randomNormal() * settings.mutationStrength
      }

      if (Math.random() < settings.mutationRate * 0.08) {
        next = randomNormal() * 0.75
      }

      return clamp(next, -4.2, 4.2)
    }),
  }
}

function cloneGenome(source: Genome, id: number): Genome {
  return {
    id,
    score: source.score,
    weights: [...source.weights],
  }
}

function getStartGridPosition(index: number, track: Track) {
  const laneLimit = Math.max(0, track.width * 0.5 - 22)
  const columnSpacing = 18
  const columns = clamp(Math.floor((laneLimit * 2) / columnSpacing) + 1, 1, 5)
  const column = index % columns
  const row = Math.floor(index / columns)
  const desiredLateralOffset =
    columns === 1 ? 0 : lerp(-laneLimit, laneLimit, column / (columns - 1)) * 0.82
  const rowDistance = 28 + row * 26
  const progress = (rowDistance / track.length) % 1
  const sample = sampleTrackAtProgress(track, progress)
  const normal = {
    x: Math.cos(sample.angle + Math.PI / 2),
    y: Math.sin(sample.angle + Math.PI / 2),
  }
  const candidateOffsets = [
    desiredLateralOffset,
    desiredLateralOffset * 0.72,
    desiredLateralOffset * 0.48,
    desiredLateralOffset * 0.24,
    0,
  ]
  let lateralOffset = 0

  for (const candidateOffset of candidateOffsets) {
    const candidatePoint = {
      x: sample.point.x + normal.x * candidateOffset,
      y: sample.point.y + normal.y * candidateOffset,
    }
    const clearance = closestPointOnTrack(track, candidatePoint).distance

    if (clearance <= track.width * 0.5 - 18) {
      lateralOffset = candidateOffset
      break
    }
  }

  return {
    x: sample.point.x + normal.x * lateralOffset,
    y: sample.point.y + normal.y * lateralOffset,
    heading: sample.angle,
    progress,
  }
}

function createCar(genome: Genome, index: number, track: Track): Car {
  const start = getStartGridPosition(index, track)

  return {
    id: genome.id,
    genome,
    x: start.x,
    y: start.y,
    heading: start.heading + (Math.random() - 0.5) * 0.045,
    speed: 0,
    vx: 0,
    vy: 0,
    angularVelocity: 0,
    slip: 0,
    traction: 1,
    alive: true,
    crashed: false,
    fitness: 0,
    lap: 0,
    progress: start.progress,
    lastProgress: start.progress,
    bestFitness: 0,
    stagnantTicks: 0,
    age: 0,
    sensors: SENSOR_ANGLES.map(() => 1),
    brain: {
      inputs: Array.from({ length: INPUT_COUNT }, () => 0),
      hidden: HIDDEN_LAYERS.map((size) => Array.from({ length: size }, () => 0)),
      outputs: [0, 0, 0],
    },
    color: CAR_COLORS[index % CAR_COLORS.length],
    streak: 0,
    lastSegmentIndex: 0,
  }
}

function castSensor(
  track: Track,
  car: Car,
  relativeAngle: number,
  obstacles: ObstacleState[],
  hintIndex?: number,
) {
  const angle = car.heading + relativeAngle
  const step = 5

  for (let distance = step; distance <= SENSOR_RANGE; distance += step) {
    const point = {
      x: car.x + Math.cos(angle) * distance,
      y: car.y + Math.sin(angle) * distance,
    }

    if (!isPointOnTrack(track, point, hintIndex) || pointHitsObstacle(point, obstacles)) {
      return distance / SENSOR_RANGE
    }
  }

  return 1
}

function ensureCarPhysics(car: Car) {
  if (!Number.isFinite(car.vx)) {
    car.vx = Math.cos(car.heading) * (Number.isFinite(car.speed) ? car.speed : 0)
  }

  if (!Number.isFinite(car.vy)) {
    car.vy = Math.sin(car.heading) * (Number.isFinite(car.speed) ? car.speed : 0)
  }

  if (!Number.isFinite(car.angularVelocity)) {
    car.angularVelocity = 0
  }

  if (!Number.isFinite(car.slip)) {
    car.slip = 0
  }

  if (!Number.isFinite(car.traction)) {
    car.traction = 1
  }

  if (car.genome.weights.length !== GENOME_LENGTH) {
    car.genome.weights = createGenome(car.genome.id).weights
  }
}

function applyDrivingPhysics(car: Car, steer: number, throttle: number, brake: number) {
  const speedMagnitude = Math.hypot(car.vx, car.vy)
  const rightBefore = {
    x: Math.cos(car.heading + Math.PI / 2),
    y: Math.sin(car.heading + Math.PI / 2),
  }
  const lateralBefore = car.vx * rightBefore.x + car.vy * rightBefore.y
  const slipRatio = clamp(Math.abs(lateralBefore) / 3.6, 0, 1)
  const steeringDemand = Math.abs(steer) * clamp(speedMagnitude / MAX_SPEED, 0, 1)
  const throttleDemand = throttle * clamp(speedMagnitude / MAX_SPEED, 0, 1)
  const brakeDemand = brake * clamp(speedMagnitude / MAX_SPEED, 0, 1)
  const traction = clamp(
    1 - steeringDemand * 0.58 - slipRatio * 0.34 - throttleDemand * 0.12 - brakeDemand * 0.2,
    0.16,
    1,
  )
  const turnAuthority = 0.018 + speedMagnitude * 0.0054
  const angularTarget = steer * turnAuthority * (0.42 + traction * 0.78)

  car.angularVelocity = car.angularVelocity * 0.74 + angularTarget
  car.heading = wrapAngle(car.heading + car.angularVelocity)

  const forward = {
    x: Math.cos(car.heading),
    y: Math.sin(car.heading),
  }
  const right = {
    x: Math.cos(car.heading + Math.PI / 2),
    y: Math.sin(car.heading + Math.PI / 2),
  }
  const driveForce = throttle * (0.08 + traction * 0.13)

  car.vx += forward.x * driveForce
  car.vy += forward.y * driveForce

  let longitudinal = car.vx * forward.x + car.vy * forward.y
  let lateral = car.vx * right.x + car.vy * right.y
  const lateralRetention = lerp(0.58, 0.96, 1 - traction)
  const rollingDrag = 0.993 - clamp(speedMagnitude / MAX_SPEED, 0, 1) * 0.012

  if (brake > 0) {
    const brakeForce = brake * (0.18 + traction * 0.18)

    if (longitudinal > 0) {
      longitudinal = Math.max(0, longitudinal - brakeForce)
    } else {
      longitudinal = Math.min(0, longitudinal + brakeForce * 0.45)
    }
  }

  longitudinal = clamp(longitudinal, MIN_SPEED, MAX_VISUAL_SPEED) * rollingDrag
  lateral *= lateralRetention

  car.vx = forward.x * longitudinal + right.x * lateral
  car.vy = forward.y * longitudinal + right.y * lateral

  const finalSpeed = Math.hypot(car.vx, car.vy)

  if (finalSpeed > MAX_VISUAL_SPEED) {
    const scale = MAX_VISUAL_SPEED / finalSpeed
    car.vx *= scale
    car.vy *= scale
  }

  car.x += car.vx
  car.y += car.vy
  car.speed = longitudinal
  car.slip = clamp(lateral / 3.6, -1, 1)
  car.traction = traction
}

function updateCar(car: Car, track: Track, obstacles: ObstacleState[]) {
  if (!car.alive) {
    return
  }

  ensureCarPhysics(car)

  const closest = closestPointOnTrack(track, car, car.lastSegmentIndex)
  car.lastSegmentIndex = closest.segmentIndex

  if (closest.distance > track.width * 0.5 - 4) {
    car.alive = false
    car.crashed = true
    car.fitness -= 35
    car.genome.score = car.fitness
    return
  }

  car.sensors = SENSOR_ANGLES.map((angle) => castSensor(track, car, angle, obstacles, car.lastSegmentIndex))

  const headingError = wrapAngle(closest.angle - car.heading)
  const laneOffset = clamp(closest.signedDistance / (track.width * 0.5), -1, 1)
  const speedMagnitude = Math.hypot(car.vx, car.vy)
  const forward = {
    x: Math.cos(car.heading),
    y: Math.sin(car.heading),
  }
  const right = {
    x: Math.cos(car.heading + Math.PI / 2),
    y: Math.sin(car.heading + Math.PI / 2),
  }
  const longitudinalVelocity = car.vx * forward.x + car.vy * forward.y
  const lateralVelocity = car.vx * right.x + car.vy * right.y
  const leftClearance = clamp((track.width * 0.5 - closest.signedDistance) / track.width, 0, 1)
  const rightClearance = clamp((track.width * 0.5 + closest.signedDistance) / track.width, 0, 1)
  const lookahead = LOOKAHEAD_DISTANCES.map((distance) => {
    const sample = sampleTrackAtProgress(track, closest.progress + distance / track.length)
    return Math.sin(wrapAngle(sample.angle - closest.angle))
  })
  const obstacleInputs = getNearestObstacleInputs(track, car, closest, obstacles)
  const rearObstacleInputs = getRearObstacleInputs(
    track,
    car,
    closest,
    obstacles,
    longitudinalVelocity,
  )
  const inputs = [
    ...car.sensors.map((value) => value * 2 - 1),
    clamp(speedMagnitude / MAX_VISUAL_SPEED, 0, 1) * 2 - 1,
    clamp(longitudinalVelocity / MAX_SPEED, -1, 1),
    clamp(lateralVelocity / 3.6, -1, 1),
    clamp(car.angularVelocity / 0.12, -1, 1),
    Math.sin(headingError),
    Math.cos(headingError),
    laneOffset,
    leftClearance * 2 - 1,
    rightClearance * 2 - 1,
    car.traction * 2 - 1,
    ...lookahead,
    ...obstacleInputs,
    ...rearObstacleInputs,
  ]

  const brain = runBrain(car.genome.weights, inputs)
  const steer = brain.outputs[0]
  const throttle = clamp((brain.outputs[1] + 1) / 2, 0, 1)
  const brake = clamp((brain.outputs[2] + 1) / 2, 0, 1)

  car.brain = brain
  applyDrivingPhysics(car, steer, throttle, brake)
  car.age += 1

  const nextClosest = closestPointOnTrack(track, car, car.lastSegmentIndex)
  car.lastSegmentIndex = nextClosest.segmentIndex

  for (const obstacle of obstacles) {
    if (distanceSquared(car, obstacle) < (obstacle.radius + 11) ** 2) {
      car.alive = false
      car.crashed = true
      car.fitness -= 90
      car.genome.score = car.fitness
      return
    }
  }

  let progressDelta = nextClosest.progress - car.lastProgress

  if (progressDelta < -0.55) {
    progressDelta += 1
    car.lap += 1
  } else if (progressDelta > 0.55) {
    progressDelta -= 1
    car.lap = Math.max(0, car.lap - 1)
  }

  car.progress = nextClosest.progress
  car.lastProgress = nextClosest.progress

  const speedScore = clamp(speedMagnitude / MAX_SPEED, 0, 1.35)
  const centerDiscipline = 1 - Math.abs(laneOffset) ** 1.7
  const alignment = Math.cos(headingError)
  const isMovingForward = progressDelta > 0.0001
  const isAligned = alignment > 0.75
  const isWaiting = Math.abs(longitudinalVelocity) < 0.1

  if (isMovingForward && isAligned) {
    car.streak = Math.min(car.streak + 0.012, 3.0)
  } else if (!isWaiting) {
    car.streak *= 0.92
  }

  const streakBonus = 1 + car.streak
  const racingReward = (1.2 + (speedScore ** 2) * 1.8 + centerDiscipline * 0.12) * streakBonus

  if (progressDelta > 0) {
    car.fitness += progressDelta * track.length * racingReward
    car.stagnantTicks = 0
  } else {
    car.fitness += progressDelta * track.length * 0.45
    car.stagnantTicks += 1
  }

  // Penalty for spinning to prevent gaming the physics
  car.fitness -= Math.abs(car.angularVelocity) * 1.5
  car.fitness -= Math.abs(car.slip) * (0.05 + speedScore * 0.12)
  car.fitness -= Math.abs(laneOffset) ** 1.55 * 0.13
  car.fitness -= throttle * brake * 0.1
  car.fitness -= Math.max(0, 0.42 - (obstacleInputs[0] + 1) * 0.5) * speedScore * 0.18
  car.fitness -=
    Math.max(0, 0.38 - (rearObstacleInputs[0] + 1) * 0.5) *
    Math.max(0, rearObstacleInputs[3]) *
    brake *
    0.26
  car.bestFitness = Math.max(car.bestFitness, car.fitness)
  car.genome.score = car.fitness

  if (car.stagnantTicks > 260 || car.age > MAX_TICKS) {
    car.alive = false
    car.genome.score = car.fitness
  }
}

function setupCanvas(canvas: HTMLCanvasElement, width: number, height: number) {
  const context = canvas.getContext('2d')

  if (!context) {
    return null
  }

  const ratio = window.devicePixelRatio || 1
  canvas.width = width * ratio
  canvas.height = height * ratio
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  return context
}

function drawTrack(context: CanvasRenderingContext2D, track: Track) {
  context.save()
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.beginPath()
  track.center.forEach((point, index) => {
    if (index === 0) {
      context.moveTo(point.x, point.y)
    } else {
      context.lineTo(point.x, point.y)
    }
  })
  context.closePath()
  context.strokeStyle = '#1f2428'
  context.lineWidth = track.width + 18
  context.stroke()
  context.strokeStyle = '#333b40'
  context.lineWidth = track.width
  context.stroke()
  context.setLineDash([22, 18])
  context.strokeStyle = 'rgba(245, 241, 231, 0.72)'
  context.lineWidth = 4
  context.stroke()
  context.setLineDash([])
  context.restore()

  for (const marker of [0.2, 0.4, 0.6, 0.8]) {
    const segment = track.segments[Math.floor(track.segments.length * marker)]
    const nx = Math.cos(segment.angle + Math.PI / 2)
    const ny = Math.sin(segment.angle + Math.PI / 2)
    const mid = segment.a

    context.strokeStyle = 'rgba(245, 184, 65, 0.56)'
    context.lineWidth = 3
    context.beginPath()
    context.moveTo(mid.x - nx * (track.width * 0.42), mid.y - ny * (track.width * 0.42))
    context.lineTo(mid.x + nx * (track.width * 0.42), mid.y + ny * (track.width * 0.42))
    context.stroke()
    context.save()
    context.translate(mid.x + nx * (track.width * 0.6), mid.y + ny * (track.width * 0.6))
    context.rotate(segment.angle)
    context.fillStyle = '#f7f3ea'
    context.font = '700 24px Inter, system-ui, sans-serif'
    context.textAlign = 'center'
    context.fillText(`${Math.round(marker * TARGET_TRACK_MILES)} mi`, 0, -8)
    context.restore()
  }

  const startSegment = track.segments[0]
  const nx = Math.cos(startSegment.angle + Math.PI / 2)
  const ny = Math.sin(startSegment.angle + Math.PI / 2)

  context.strokeStyle = '#f7f3ea'
  context.lineWidth = 7
  context.beginPath()
  context.moveTo(
    track.start.x - nx * (track.width * 0.48),
    track.start.y - ny * (track.width * 0.48),
  )
  context.lineTo(
    track.start.x + nx * (track.width * 0.48),
    track.start.y + ny * (track.width * 0.48),
  )
  context.stroke()
}

function getCamera(track: Track, target: Vec) {
  return {
    x: clamp(target.x - WORLD_WIDTH / 2, 0, Math.max(0, track.worldWidth - WORLD_WIDTH)),
    y: clamp(target.y - WORLD_HEIGHT / 2, 0, Math.max(0, track.worldHeight - WORLD_HEIGHT)),
  }
}

function drawGround(context: CanvasRenderingContext2D, camera: Vec) {
  context.clearRect(0, 0, WORLD_WIDTH, WORLD_HEIGHT)
  context.fillStyle = '#ded8cc'
  context.fillRect(0, 0, WORLD_WIDTH, WORLD_HEIGHT)
  context.save()
  context.strokeStyle = 'rgba(32, 39, 44, 0.08)'
  context.lineWidth = 1

  const grid = 100
  const startX = -(camera.x % grid)
  const startY = -(camera.y % grid)

  for (let x = startX; x < WORLD_WIDTH; x += grid) {
    context.beginPath()
    context.moveTo(x, 0)
    context.lineTo(x, WORLD_HEIGHT)
    context.stroke()
  }

  for (let y = startY; y < WORLD_HEIGHT; y += grid) {
    context.beginPath()
    context.moveTo(0, y)
    context.lineTo(WORLD_WIDTH, y)
    context.stroke()
  }

  context.restore()
}

function drawMiniMap(
  context: CanvasRenderingContext2D,
  track: Track,
  champion: Car | undefined,
  camera: Vec,
  obstacles: ObstacleState[],
) {
  const width = 178
  const height = 132
  const x = WORLD_WIDTH - width - 18
  const y = WORLD_HEIGHT - height - 18
  const scale = Math.min((width - 24) / track.worldWidth, (height - 24) / track.worldHeight)

  context.save()
  context.fillStyle = 'rgba(24, 32, 39, 0.84)'
  context.strokeStyle = 'rgba(247, 243, 234, 0.24)'
  context.lineWidth = 1
  context.beginPath()
  context.roundRect(x, y, width, height, 6)
  context.fill()
  context.stroke()
  context.translate(x + 12, y + 14)
  context.scale(scale, scale)
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.strokeStyle = 'rgba(247, 243, 234, 0.42)'
  context.lineWidth = Math.max(5 / scale, 1)
  context.beginPath()
  track.center.forEach((point, index) => {
    if (index === 0) {
      context.moveTo(point.x, point.y)
    } else {
      context.lineTo(point.x, point.y)
    }
  })
  context.closePath()
  context.stroke()

  context.strokeStyle = 'rgba(112, 193, 179, 0.9)'
  context.lineWidth = Math.max(2 / scale, 1)
  context.strokeRect(camera.x, camera.y, WORLD_WIDTH, WORLD_HEIGHT)

  if (champion) {
    context.fillStyle = '#f5b841'
    context.beginPath()
    context.arc(champion.x, champion.y, Math.max(7 / scale, 2), 0, Math.PI * 2)
    context.fill()
  }

  for (const obstacle of obstacles) {
    context.fillStyle = obstacle.color
    context.beginPath()
    context.arc(obstacle.x, obstacle.y, Math.max(4 / scale, 1.5), 0, Math.PI * 2)
    context.fill()
  }

  context.restore()
}

function drawObstacle(context: CanvasRenderingContext2D, obstacle: ObstacleState) {
  context.save()
  context.translate(obstacle.x, obstacle.y)
  context.rotate(obstacle.heading)
  context.fillStyle = obstacle.color
  context.strokeStyle = '#111820'
  context.lineWidth = 2
  context.beginPath()
  context.roundRect(-obstacle.radius * 1.25, -obstacle.radius * 0.85, obstacle.radius * 2.5, obstacle.radius * 1.7, 5)
  context.fill()
  context.stroke()
  context.fillStyle = 'rgba(247, 243, 234, 0.88)'
  context.beginPath()
  context.arc(obstacle.radius * 0.55, 0, obstacle.radius * 0.22, 0, Math.PI * 2)
  context.fill()
  context.restore()
}

function drawCar(
  context: CanvasRenderingContext2D,
  car: Car,
  isChampion: boolean,
  track: Track,
) {
  if (car.alive && Math.abs(car.slip) > 0.18) {
    const slipAlpha = clamp(Math.abs(car.slip), 0.18, 1)

    context.save()
    context.globalAlpha = isChampion ? 0.24 + slipAlpha * 0.26 : 0.14 + slipAlpha * 0.18
    context.strokeStyle = isChampion ? '#f7f3ea' : '#15191d'
    context.lineWidth = isChampion ? 3 : 2
    context.lineCap = 'round'

    for (const side of [-1, 1]) {
      const wheelX =
        car.x -
        Math.cos(car.heading) * 9 +
        Math.cos(car.heading + Math.PI / 2) * side * 5
      const wheelY =
        car.y -
        Math.sin(car.heading) * 9 +
        Math.sin(car.heading + Math.PI / 2) * side * 5

      context.beginPath()
      context.moveTo(wheelX, wheelY)
      context.lineTo(
        wheelX - car.vx * (3.6 + slipAlpha * 5),
        wheelY - car.vy * (3.6 + slipAlpha * 5),
      )
      context.stroke()
    }

    context.restore()
  }

  context.save()
  context.translate(car.x, car.y)
  context.rotate(car.heading)
  context.globalAlpha = car.alive ? 1 : 0.28
  context.fillStyle = isChampion ? '#f5b841' : car.color
  context.strokeStyle = isChampion ? '#fff6d8' : '#101316'
  context.lineWidth = isChampion ? 2.5 : 1.5
  context.beginPath()
  context.roundRect(-11, -6.5, 22, 13, 4)
  context.fill()
  context.stroke()
  context.fillStyle = '#f7f3ea'
  context.beginPath()
  context.moveTo(8, 0)
  context.lineTo(1, -4)
  context.lineTo(1, 4)
  context.closePath()
  context.fill()
  context.restore()

  if (!isChampion || !car.alive) {
    return
  }

  context.save()
  context.lineWidth = 1.5
  SENSOR_ANGLES.forEach((relativeAngle, index) => {
    const angle = car.heading + relativeAngle
    const length = car.sensors[index] * SENSOR_RANGE
    const end = {
      x: car.x + Math.cos(angle) * length,
      y: car.y + Math.sin(angle) * length,
    }

    context.strokeStyle = 'rgba(112, 193, 179, 0.55)'
    context.beginPath()
    context.moveTo(car.x, car.y)
    context.lineTo(end.x, end.y)
    context.stroke()
    context.fillStyle = 'rgba(112, 193, 179, 0.85)'
    context.beginPath()
    context.arc(end.x, end.y, 2.5, 0, Math.PI * 2)
    context.fill()
  })

  const closest = closestPointOnTrack(track, car)
  context.strokeStyle = 'rgba(247, 243, 234, 0.65)'
  context.lineWidth = 1
  context.beginPath()
  context.moveTo(car.x, car.y)
  context.lineTo(closest.point.x, closest.point.y)
  context.stroke()
  context.restore()
}

function drawWorld(
  canvas: HTMLCanvasElement | null,
  track: Track,
  cars: Car[],
  champion: Car | undefined,
  obstacles: ObstacleState[],
) {
  if (!canvas) {
    return
  }

  const context = setupCanvas(canvas, WORLD_WIDTH, WORLD_HEIGHT)

  if (!context) {
    return
  }

  const camera = getCamera(track, champion ?? track.start)
  drawGround(context, camera)
  context.save()
  context.translate(-camera.x, -camera.y)
  drawTrack(context, track)

  for (const obstacle of obstacles) {
    drawObstacle(context, obstacle)
  }

  for (const car of cars) {
    if (car.id !== champion?.id) {
      drawCar(context, car, false, track)
    }
  }

  if (champion) {
    drawCar(context, champion, true, track)
  }

  context.restore()
  drawMiniMap(context, track, champion, camera, obstacles)
}

function drawBrain(canvas: HTMLCanvasElement | null, car: Car | undefined) {
  if (!canvas) {
    return
  }

  const width = 520
  const height = 420
  const context = setupCanvas(canvas, width, height)

  if (!context) {
    return
  }

  context.clearRect(0, 0, width, height)
  context.fillStyle = '#f4f1e9'
  context.fillRect(0, 0, width, height)

  if (!car) {
    context.fillStyle = '#59636b'
    context.font = '16px Inter, system-ui, sans-serif'
    context.fillText('Start training to wake up a neural net.', 122, 206)
    return
  }

  const rawHidden = car.brain.hidden as unknown
  const hiddenLayers =
    Array.isArray(rawHidden) && Array.isArray(rawHidden[0])
      ? (rawHidden as number[][])
      : HIDDEN_LAYERS.map((size) => Array.from({ length: size }, () => 0))
  const inputLayer =
    car.brain.inputs.length === INPUT_COUNT
      ? car.brain.inputs
      : Array.from({ length: INPUT_COUNT }, () => 0)
  const outputLayer =
    car.brain.outputs.length === OUTPUT_COUNT
      ? car.brain.outputs
      : Array.from({ length: OUTPUT_COUNT }, () => 0)
  const layers = [inputLayer, ...hiddenLayers, outputLayer]
  const labels = layers.map((_, index) => {
    if (index === 0) {
      return 'Inputs'
    }

    if (index === layers.length - 1) {
      return 'Drive'
    }

    return `H${index}`
  })
  const xPositions = layers.map((_, index) => lerp(48, width - 48, index / (layers.length - 1)))
  const nodePositions = layers.map((layer, layerIndex) =>
    layer.map((_, index) => ({
      x: xPositions[layerIndex],
      y: lerp(46, height - 46, layer.length === 1 ? 0.5 : index / (layer.length - 1)),
    })),
  )

  let cursor = 0

  for (let toLayerIndex = 1; toLayerIndex < layers.length; toLayerIndex += 1) {
    const fromLayerIndex = toLayerIndex - 1
    const fromLayer = layers[fromLayerIndex]
    const toLayer = layers[toLayerIndex]

    for (let toIndex = 0; toIndex < toLayer.length; toIndex += 1) {
      cursor += 1

      for (let fromIndex = 0; fromIndex < fromLayer.length; fromIndex += 1) {
        const weight = car.genome.weights[cursor]
        const from = nodePositions[fromLayerIndex][fromIndex]
        const to = nodePositions[toLayerIndex][toIndex]
        const opacity = clamp(Math.abs(weight) / 4.2, 0.025, toLayerIndex === layers.length - 1 ? 0.62 : 0.34)

        context.strokeStyle =
          weight >= 0
            ? `rgba(36, 123, 160, ${opacity})`
            : `rgba(242, 95, 92, ${opacity})`
        context.lineWidth = clamp(Math.abs(weight) * 0.48, 0.25, 2.1)
        context.beginPath()
        context.moveTo(from.x, from.y)
        context.lineTo(to.x, to.y)
        context.stroke()
        cursor += 1
      }
    }
  }

  layers.forEach((layer, layerIndex) => {
    context.fillStyle = '#20272c'
    context.font = '600 12px Inter, system-ui, sans-serif'
    context.textAlign = 'center'
    context.fillText(labels[layerIndex], xPositions[layerIndex], 24)

    layer.forEach((activation, index) => {
      const position = nodePositions[layerIndex][index]
      const intensity = clamp((activation + 1) / 2, 0, 1)
      const radius = layerIndex === layers.length - 1 ? 12 : layer.length > 24 ? 6 : 7.5

      context.fillStyle = `rgb(${Math.round(242 - intensity * 62)}, ${Math.round(
        241 - intensity * 88,
      )}, ${Math.round(233 - intensity * 122)})`
      context.strokeStyle = layerIndex === layers.length - 1 ? '#20272c' : '#59636b'
      context.lineWidth = layerIndex === layers.length - 1 ? 2 : 1.2
      context.beginPath()
      context.arc(position.x, position.y, radius, 0, Math.PI * 2)
      context.fill()
      context.stroke()

      if (layerIndex === layers.length - 1) {
        const outputLabels = ['Steer', 'Gas', 'Brake']
        context.fillStyle = '#20272c'
        context.font = '600 11px Inter, system-ui, sans-serif'
        context.fillText(outputLabels[index], position.x, position.y + 28)
      }
    })
  })
}

function loadSavedBrains(): Genome[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)

    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw) as Genome[]
    return parsed
      .filter((brain) => Array.isArray(brain.weights) && brain.weights.length === GENOME_LENGTH)
      .slice(0, 5)
  } catch {
    return []
  }
}

function saveBrains(brains: Genome[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(brains.slice(0, 5)))
}

function formatScore(score: number) {
  return Math.round(score).toLocaleString()
}

function App() {
  const worldCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const brainCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const trackRef = useRef<Track>(createTrack())
  const carsRef = useRef<Car[]>([])
  const genomesRef = useRef<Genome[]>([])
  const savedBrainsRef = useRef<Genome[]>([])
  const settingsRef = useRef<Settings>(DEFAULT_SETTINGS)
  const generationRef = useRef(1)
  const tickRef = useRef(0)
  const runningRef = useRef(false)
  const animationRef = useRef(0)
  const nextGenomeIdRef = useRef(1)
  const bestEverRef = useRef(0)
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)
  const [stats, setStats] = useState<Stats>(INITIAL_STATS)
  const [isRunning, setIsRunning] = useState(false)

  if (!trackRef.current.worldWidth || !trackRef.current.miles || !trackRef.current.obstacles) {
    trackRef.current = createTrack()
  }

  const trackMiles = trackRef.current.miles || trackRef.current.length / UNITS_PER_MILE
  const trackLabel = `${trackRef.current.name} · ${trackMiles.toFixed(1)} mi`

  function pushStats(forceComplete = false) {
    const cars = carsRef.current
    const alive = cars.filter((car) => car.alive).length
    const sorted = [...cars].sort((a, b) => b.fitness - a.fitness)
    const champion = sorted[0]
    const averageScore =
      cars.reduce((total, car) => total + car.fitness, 0) / Math.max(cars.length, 1)

    bestEverRef.current = Math.max(bestEverRef.current, champion?.fitness ?? 0)
    setStats({
      generation: generationRef.current,
      alive,
      bestScore: champion?.fitness ?? 0,
      bestEver: bestEverRef.current,
      averageScore,
      championLap: champion?.lap ?? 0,
      championSpeed: champion ? Math.hypot(champion.vx, champion.vy) : 0,
      championTraction: champion?.traction ?? 1,
      championSlip: champion?.slip ?? 0,
      trialComplete:
        forceComplete ||
        (!settingsRef.current.infiniteTraining &&
          generationRef.current > settingsRef.current.trials),
      selectedCarId: champion?.id ?? stats.selectedCarId,
      savedBrains: savedBrainsRef.current,
    })
  }

  function createPopulation(sourceGenomes?: Genome[]) {
    const population = settingsRef.current.population
    const genomes =
      sourceGenomes ??
      Array.from({ length: population }, () => {
        const genome = createGenome(nextGenomeIdRef.current)
        nextGenomeIdRef.current += 1
        return genome
      })

    genomesRef.current = genomes.slice(0, population)
    carsRef.current = genomesRef.current.map((genome, index) =>
      createCar(genome, index, trackRef.current),
    )
    tickRef.current = 0
  }

  function startFresh(nextTrack = trackRef.current) {
    generationRef.current = 1
    tickRef.current = 0
    bestEverRef.current = savedBrainsRef.current.length > 0
      ? Math.max(...savedBrainsRef.current.map((b) => b.score))
      : 0
    nextGenomeIdRef.current = 1
    trackRef.current = nextTrack
    createPopulation()
    pushStats()
  }

  function finishGeneration() {
    const sorted = [...genomesRef.current].sort((a, b) => b.score - a.score)
    const savedCount = clamp(settingsRef.current.saveTop, 1, 12)
    
    // Merge existing saved brains with the current generation
    const combined = [...savedBrainsRef.current, ...genomesRef.current]
    
    // Sort combined brains by score in descending order
    const sortedCombined = combined.sort((a, b) => b.score - a.score)
    
    // De-duplicate based on score to ensure unique brains in the list
    const uniqueCombined: Genome[] = []
    const seenScores = new Set<number>()
    
    for (const genome of sortedCombined) {
      if (genome.score > 0 && !seenScores.has(genome.score)) {
        seenScores.add(genome.score)
        uniqueCombined.push(genome)
      }
    }
    
    const finalTop = uniqueCombined.length > 0 ? uniqueCombined : sortedCombined
    
    // Keep the top 5 historical best brains
    savedBrainsRef.current = finalTop.slice(0, 5).map((genome, index) => ({
      ...cloneGenome(genome, genome.id),
      id: index + 1,
    }))
    
    saveBrains(savedBrainsRef.current)

    if (
      !settingsRef.current.infiniteTraining &&
      generationRef.current >= settingsRef.current.trials
    ) {
      runningRef.current = false
      setIsRunning(false)
      pushStats(true)
      return
    }

    generationRef.current += 1

    if (settingsRef.current.randomTrackEachTrial) {
      trackRef.current = createTrack(randomSeed(), true)
    }

    const eliteCount = Math.min(savedCount, sorted.length, settingsRef.current.population)
    const parents = sorted.slice(0, Math.max(eliteCount, Math.ceil(sorted.length * 0.28)))
    const nextGenomes: Genome[] = []

    for (let index = 0; index < eliteCount; index += 1) {
      const elite = cloneGenome(sorted[index], nextGenomeIdRef.current)
      elite.score = 0
      nextGenomeIdRef.current += 1
      nextGenomes.push(elite)
    }

    while (nextGenomes.length < settingsRef.current.population) {
      const parentA = parents[Math.floor(Math.random() * parents.length)]
      const parentB = parents[Math.floor(Math.random() * parents.length)]
      nextGenomes.push(
        crossover(parentA, parentB, nextGenomeIdRef.current, settingsRef.current),
      )
      nextGenomeIdRef.current += 1
    }

    createPopulation(nextGenomes)
    pushStats()
  }

  function stepSimulation() {
    const cars = carsRef.current

    for (let step = 0; step < settingsRef.current.simSpeed; step += 1) {
      if (!runningRef.current) {
        return
      }

      tickRef.current += 1
      const obstacles = getObstacleStates(trackRef.current, tickRef.current, settingsRef.current.enableObstacles)

      for (const car of cars) {
        updateCar(car, trackRef.current, obstacles)
      }

      if (cars.every((car) => !car.alive) || tickRef.current > MAX_TICKS) {
        finishGeneration()
        break
      }
    }
  }

  function animate() {
    if (runningRef.current) {
      stepSimulation()
    }

    const champion = [...carsRef.current].sort((a, b) => b.fitness - a.fitness)[0]
    drawWorld(
      worldCanvasRef.current,
      trackRef.current,
      carsRef.current,
      champion,
      getObstacleStates(trackRef.current, tickRef.current, settingsRef.current.enableObstacles),
    )
    drawBrain(brainCanvasRef.current, champion)
    animationRef.current = window.requestAnimationFrame(animate)
  }

  useEffect(() => {
    savedBrainsRef.current = loadSavedBrains()
    settingsRef.current = settings
    startFresh(trackRef.current)
    animationRef.current = window.requestAnimationFrame(animate)

    const statTimer = window.setInterval(() => pushStats(), 180)

    return () => {
      window.cancelAnimationFrame(animationRef.current)
      window.clearInterval(statTimer)
    }
    // The animation loop reads mutable refs so React does not rebind it every frame.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    settingsRef.current = settings
  }, [settings])

  function updateNumberSetting(
    event: ChangeEvent<HTMLInputElement>,
    key: keyof Pick<
      Settings,
      'population' | 'trials' | 'saveTop' | 'simSpeed' | 'mutationRate' | 'mutationStrength'
    >,
  ) {
    const value = Number(event.target.value)
    const nextValue = Number.isFinite(value) ? value : DEFAULT_SETTINGS[key]

    setSettings((current) => ({
      ...current,
      [key]:
        key === 'population'
          ? clamp(Math.round(nextValue), 2, 120)
          : key === 'trials'
            ? clamp(Math.round(nextValue), 1, 100_000)
            : key === 'saveTop'
              ? clamp(Math.round(nextValue), 1, 12)
              : key === 'simSpeed'
                ? clamp(Math.round(nextValue), 1, 120)
                : key === 'mutationRate'
                  ? clamp(nextValue, 0.01, 0.5)
                  : clamp(nextValue, 0.05, 1.5),
    }))
  }

  function toggleTraining() {
    if (carsRef.current.length === 0) {
      startFresh(trackRef.current)
    }

    runningRef.current = !runningRef.current
    setIsRunning(runningRef.current)
  }

  function resetTraining() {
    runningRef.current = false
    setIsRunning(false)
    startFresh(trackRef.current)
  }

  function makeRandomTrack() {
    runningRef.current = false
    setIsRunning(false)
    startFresh(createTrack(randomSeed(), true))
  }

  function exportBrains() {
    const payload = {
      exportedAt: new Date().toISOString(),
      track: {
        seed: trackRef.current.seed,
        name: trackRef.current.name,
        miles: trackRef.current.miles,
      },
      inputCount: INPUT_COUNT,
      hiddenLayers: HIDDEN_LAYERS,
      outputCount: OUTPUT_COUNT,
      brains: savedBrainsRef.current,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'neuro-racer-top-5.json'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  function handleImportFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string
        const data = JSON.parse(content)
        
        let importedList: Genome[] = []
        if (Array.isArray(data.brains)) {
          importedList = data.brains
        } else if (Array.isArray(data)) {
          importedList = data
        } else {
          alert('Invalid format. File must contain an array of brains or a brains field.')
          return
        }

        // Validate imported genomes
        const validBrains = importedList.filter((brain) => {
          return (
            brain &&
            typeof brain === 'object' &&
            Array.isArray(brain.weights) &&
            brain.weights.length === GENOME_LENGTH &&
            typeof brain.score === 'number'
          )
        })

        if (validBrains.length === 0) {
          alert('No valid brains found in the JSON file. Ensure they match the network architecture.')
          return
        }

        const topBrains = validBrains.slice(0, 5)
        savedBrainsRef.current = topBrains
        saveBrains(topBrains)
        
        setStats((current) => ({
          ...current,
          savedBrains: topBrains,
        }))

        alert(`Successfully imported ${topBrains.length} brains! Click "Seed population with these brains" to run them.`)
      } catch (error) {
        console.error(error)
        alert('Failed to parse JSON file.')
      }
    }
    reader.readAsText(file)
    event.target.value = ''
  }

  function importBrains() {
    document.getElementById('import-brains-file-input')?.click()
  }

  function seedFromSaved() {
    const imported = savedBrainsRef.current
    if (imported.length === 0) return

    const population = settingsRef.current.population
    const nextGenomes: Genome[] = []

    // 1. Add the saved/imported brains themselves directly
    imported.forEach((brain) => {
      if (nextGenomes.length < population) {
        const clone = cloneGenome(brain, nextGenomeIdRef.current)
        clone.score = 0 // Reset score for new trial
        nextGenomeIdRef.current += 1
        nextGenomes.push(clone)
      }
    })

    // 2. Fill the remaining slots with crossover/mutations of the imported brains
    while (nextGenomes.length < population) {
      const parentA = imported[Math.floor(Math.random() * imported.length)]
      const parentB = imported[Math.floor(Math.random() * imported.length)]
      nextGenomes.push(
        crossover(parentA, parentB, nextGenomeIdRef.current, settingsRef.current),
      )
      nextGenomeIdRef.current += 1
    }

    runningRef.current = false
    setIsRunning(false)
    generationRef.current = 1
    tickRef.current = 0
    bestEverRef.current = Math.max(...imported.map((b) => b.score))
    
    createPopulation(nextGenomes)
    pushStats()
    
    alert('Population successfully seeded from your imported brains! Click "Train" to run the simulation.')
  }

  const savedBrains = stats.savedBrains
  const progress =
    settings.infiniteTraining || settings.trials <= 0
      ? 0
      : clamp(stats.generation / settings.trials, 0, 1)

  return (
    <main className="app-shell">
      <section className="simulator-panel">
        <div className="canvas-head">
          <div>
            <p className="eyebrow">Neuroevolution racing lab</p>
            <h1>Self-training cars</h1>
          </div>
          <div className="head-actions" aria-label="Training controls">
            <button className="icon-button primary" onClick={toggleTraining} type="button">
              {isRunning ? <Pause size={18} /> : <Play size={18} />}
              <span>{isRunning ? 'Pause' : 'Train'}</span>
            </button>
            <button className="icon-button" onClick={resetTraining} type="button">
              <RefreshCcw size={18} />
              <span>Reset</span>
            </button>
            <button className="icon-button" onClick={makeRandomTrack} type="button">
              <Shuffle size={18} />
              <span>Random track</span>
            </button>
          </div>
        </div>

        <div className="track-wrap">
          <canvas
            aria-label="Live race course with learning cars"
            className="world-canvas"
            height={WORLD_HEIGHT}
            ref={worldCanvasRef}
            width={WORLD_WIDTH}
          />
          <div className="track-hud" aria-live="polite">
            <span>{trackLabel}</span>
            <strong>
              Trial {settings.infiniteTraining ? stats.generation : Math.min(stats.generation, settings.trials)}
              {settings.infiniteTraining ? '' : ` / ${settings.trials}`}
            </strong>
          </div>
        </div>
      </section>

      <aside className="side-panel">
        <section className="metric-grid" aria-label="Training statistics">
          <div className="metric">
            <Trophy size={18} />
            <span>Best</span>
            <strong>{formatScore(stats.bestScore)}</strong>
          </div>
          <div className="metric">
            <Activity size={18} />
            <span>Alive</span>
            <strong>
              {stats.alive}/{settings.population}
            </strong>
          </div>
          <div className="metric">
            <Flag size={18} />
            <span>Lap</span>
            <strong>{stats.championLap}</strong>
          </div>
          <div className="metric">
            <Gauge size={18} />
            <span>Speed</span>
            <strong>{stats.championSpeed.toFixed(1)}</strong>
          </div>
          <div className="metric">
            <Gauge size={18} />
            <span>Traction</span>
            <strong>{Math.round(stats.championTraction * 100)}%</strong>
          </div>
          <div className="metric">
            <Activity size={18} />
            <span>Slip</span>
            <strong>{Math.round(Math.abs(stats.championSlip) * 100)}%</strong>
          </div>
        </section>

        <section className="brain-panel">
          <div className="section-title">
            <Brain size={18} />
            <h2>Live brain</h2>
          </div>
          <canvas
            aria-label="Neural network visualization"
            className="brain-canvas"
            height={420}
            ref={brainCanvasRef}
            width={520}
          />
          <div className="brain-legend" aria-label="Brain legend">
            <span className="positive">Positive weights</span>
            <span className="negative">Negative weights</span>
          </div>
        </section>

        <section className="controls-panel">
          <div className="section-title">
            <Flag size={18} />
            <h2>Training setup</h2>
          </div>

          <label className="field">
            <span>Population</span>
            <input
              min={2}
              max={120}
              onChange={(event) => updateNumberSetting(event, 'population')}
              type="number"
              value={settings.population}
            />
          </label>

          <label className="field">
            <span>Trials</span>
            <input
              disabled={settings.infiniteTraining}
              min={1}
              max={100000}
              onChange={(event) => updateNumberSetting(event, 'trials')}
              type="number"
              value={settings.trials}
            />
          </label>

          <label className="field">
            <span>Save top</span>
            <input
              min={1}
              max={12}
              onChange={(event) => updateNumberSetting(event, 'saveTop')}
              type="number"
              value={settings.saveTop}
            />
          </label>

          <label className="field">
            <span>Simulation speed</span>
            <input
              min={1}
              max={120}
              onChange={(event) => updateNumberSetting(event, 'simSpeed')}
              type="range"
              value={settings.simSpeed}
            />
            <b>{settings.simSpeed}x</b>
          </label>

          <label className="field">
            <span>Mutation rate</span>
            <input
              max={0.5}
              min={0.01}
              onChange={(event) => updateNumberSetting(event, 'mutationRate')}
              step={0.01}
              type="range"
              value={settings.mutationRate}
            />
            <b>{Math.round(settings.mutationRate * 100)}%</b>
          </label>

          <label className="field">
            <span>Mutation strength</span>
            <input
              max={1.5}
              min={0.05}
              onChange={(event) => updateNumberSetting(event, 'mutationStrength')}
              step={0.05}
              type="range"
              value={settings.mutationStrength}
            />
            <b>{settings.mutationStrength.toFixed(2)}</b>
          </label>

          <div className="toggle-row">
            <label>
              <input
                checked={settings.randomTrackEachTrial}
                onChange={(event) =>
                  setSettings((current) => ({
                    ...current,
                    randomTrackEachTrial: event.target.checked,
                  }))
                }
                type="checkbox"
              />
              New random track every trial
            </label>
            <label>
              <input
                checked={settings.infiniteTraining}
                onChange={(event) =>
                  setSettings((current) => ({
                    ...current,
                    infiniteTraining: event.target.checked,
                  }))
                }
                type="checkbox"
              />
              Train forever
            </label>
            <label>
              <input
                checked={settings.enableObstacles}
                onChange={(event) =>
                  setSettings((current) => ({
                    ...current,
                    enableObstacles: event.target.checked,
                  }))
                }
                type="checkbox"
              />
              Enable obstacles
            </label>
          </div>

          <div className="progress-bar" aria-label="Trial progress">
            <span style={{ width: `${progress * 100}%` }} />
          </div>
        </section>

        <section className="saved-panel">
          <div className="section-title split">
            <div>
              <Trophy size={18} />
              <h2>Saved top 5</h2>
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                aria-label="Import saved brains"
                className="small-icon"
                onClick={importBrains}
                title="Import saved brains"
                type="button"
              >
                <Upload size={16} />
              </button>
              <button
                aria-label="Export saved brains"
                className="small-icon"
                disabled={savedBrains.length === 0}
                onClick={exportBrains}
                title="Export saved brains"
                type="button"
              >
                <Download size={16} />
              </button>
            </div>
          </div>

          <ol className="brain-list">
            {Array.from({ length: 5 }, (_, index) => {
              const brain = savedBrains[index]

              return (
                <li key={index}>
                  <span>#{index + 1}</span>
                  <strong>{brain ? formatScore(brain.score) : 'Waiting'}</strong>
                </li>
              )
            })}
          </ol>

          {savedBrains.length > 0 && (
            <button
              className="icon-button"
              onClick={seedFromSaved}
              type="button"
              style={{ marginTop: '12px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', padding: '8px 12px', borderRadius: '6px', fontSize: '12px', fontWeight: 600 }}
            >
              <Brain size={14} />
              <span>Seed population with these brains</span>
            </button>
          )}
        </section>
      </aside>
      <input
        id="import-brains-file-input"
        type="file"
        accept="application/json"
        onChange={handleImportFile}
        style={{ display: 'none' }}
      />
    </main>
  )
}

export default App
