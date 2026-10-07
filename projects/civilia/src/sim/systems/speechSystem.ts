import { rngFor } from '../rng'
import type { Agent, Belief, Memory, Relationship, SimEvent, World } from '../types'
import { relationshipWith } from '../relationships'
import { peekMemories } from './memorySystem'
import { damagingBeliefsAbout } from './beliefSystem'

/**
 * Speech v0 — every social interaction produces an actual line, composed
 * from preset templates mixed with live context: names, prices, rumors,
 * landmarks, buildings, relationship state. The line and a motive analysis
 * are stored on the emitting event (`payload.speech`), so history replays
 * with the same words and every phrase can answer "why did they say that?"
 *
 * Speech is flavor over canonical events; it never changes world state.
 * When the Ollama gateway lands (stage 29), interesting lines get an LLM
 * pass — these templates remain the deterministic fallback.
 */

export type SpeechCategory = 'chat' | 'rumor' | 'hostile' | 'kind' | 'status' | 'alarm'

export type MotiveKind =
  | 'genuine'
  | 'bonding'
  | 'premeditated'
  | 'deceitful'
  | 'performative'
  | 'desperate'
  | 'venting'

export interface Utterance {
  text: string
  category: SpeechCategory
  motive: {
    kind: MotiveKind
    summary: string
    reasons: string[]
  }
  /**
   * The specific memory this line draws on, snapshotted at speaking time —
   * the memory itself may decay and be forgotten, but the receipt stays.
   */
  citedMemory?: { id: string; summary: string; day: number }
  /** The specific belief this line draws on, snapshotted at speaking time. */
  citedBelief?: { id: string; claim: string; confidence: number; source: string }
}

interface SpeechContext {
  world: World
  event: SimEvent
  speaker: Agent
  listener?: Agent
  rel?: Relationship
  pick: <T>(arr: readonly T[]) => T
}

const first = (name: string): string => name.split(' ')[0]

function landmarkName(world: World, kind?: string): string | null {
  const pool = kind
    ? world.terrain.landmarks.filter((l) => l.kind === kind)
    : world.terrain.landmarks
  return pool.length > 0 ? pool[0].name : null
}

// ---------------------------------------------------------------------------
// Per-event-type composers

function talkLine(ctx: SpeechContext): { text: string; category: SpeechCategory } {
  const { world, speaker, listener, event, pick } = ctx
  const topic = (event.payload as { topic?: string }).topic ?? 'the weather turning'
  const bakery = world.places['place_bakery']
  const lake = landmarkName(world, 'lake')
  const forest = landmarkName(world, 'forest')
  const peak = landmarkName(world, 'mountain')

  const byTopic: Record<string, string[]> = {
    'bread prices': [
      `${bakery.foodPrice} coins a loaf now. ${bakery.foodPrice >= 3 ? 'Somebody is getting fat on this.' : 'Could be worse, I suppose.'}`,
      `Did you pay ${bakery.foodPrice} at the bakery too? My purse felt that.`,
    ],
    'the harvest': [
      `The fields look thin this year. We'll feel it by winter.`,
      `Grain's coming in slow. ${first(listener?.name ?? '')}, you've seen the farm — am I wrong?`,
    ],
    'the weather turning': [
      peak ? `Clouds are sitting on ${peak} again. Rain by evening, mark me.` : `Sky's turning. Rain by evening, mark me.`,
      `Cold wind off the water today.`,
    ],
    'old stories': [
      forest ? `My mother swore ${forest} was full of voices at night. I half believe her.` : `My mother told stories about these woods. I half believe them.`,
      lake ? `They say a wedding ring sits at the bottom of ${lake}. Someone's, anyway.` : `Every town has its drowned secrets.`,
    ],
    'the state of the well': [
      `The well rope is fraying again. Someone will fall in before anyone fixes it.`,
    ],
    'tavern songs': [`They sang the one about the drowned miller again last night. Off key.`],
    'family matters': [
      `${first(speaker.name)}s look after their own. That's all I'll say.`,
      `The little ones grow faster than the grain does.`,
    ],
    'who was seen with whom': [
      `I saw who lingered by the well this morning. Not my business, of course.`,
    ],
  }
  const lines = byTopic[topic] ?? [`So… ${topic}. Strange days.`]
  return { text: pick(lines), category: 'chat' }
}

function rumorLine(ctx: SpeechContext): { text: string; category: SpeechCategory } {
  const { event, pick } = ctx
  const claim = (event.payload as { claim?: string }).claim ?? 'something is going on'
  const openers = [
    `Keep this close — ${claim}.`,
    `You didn't hear it from me, but ${claim}.`,
    `Everyone half knows it already: ${claim}.`,
    `I'd swear on the well: ${claim}.`,
  ]
  return { text: pick(openers), category: 'rumor' }
}

interface ComposedLine {
  text: string
  category: SpeechCategory
  citedMemory?: { id: string; summary: string; day: number }
  citedBelief?: { id: string; claim: string; confidence: number; source: string }
}

/** "three days back", "yesterday" — memories surface with their age attached. */
function daysAgo(world: World, day: number): string {
  const n = world.day - day
  if (n <= 0) return 'this very day'
  if (n === 1) return 'yesterday'
  return `${n} days back`
}

/**
 * The dialogue engine's core move: query the speaker's memory graph for the
 * strongest edge to the listener and say something about *that*, not a
 * generic bark. Falls back to belief citations, then stock lines.
 */
function insultLine(ctx: SpeechContext): ComposedLine {
  const { world, speaker, listener, pick } = ctx
  const job = listener?.jobId

  // Strongest wound this speaker carries about this listener.
  const wound: Memory | undefined = listener
    ? peekMemories(world, speaker, {
        aboutAgentId: listener.id,
        tones: ['angry', 'ashamed', 'bitter'],
        limit: 1,
      })[0]
    : undefined
  if (wound) {
    const when = daysAgo(world, wound.createdDay)
    const lines: string[] = []
    if (wound.tags.includes('crime')) {
      lines.push(
        `I saw you with my own eyes ${when}, ${first(listener!.name)}. Once a thief, always a thief.`,
        `Steal anything ${when === 'this very day' ? 'else today' : 'lately'}? The whole town counts its loaves around you.`,
      )
    } else if (wound.summary.includes('insulted me')) {
      lines.push(
        `${when[0].toUpperCase()}${when.slice(1)} you made the town laugh at me. See how it sits on your side of the fence.`,
        `I owed you this one since ${when}, and I always pay my debts.`,
      )
    } else if (wound.tags.includes('border')) {
      lines.push(`Boundary stones don't walk, ${first(listener!.name)} — and I haven't forgotten ${when}.`)
    } else {
      lines.push(`I haven't forgotten what you did ${when}. Not a day of it.`)
    }
    return {
      text: pick(lines),
      category: 'hostile',
      citedMemory: { id: wound.id, summary: wound.summary, day: wound.createdDay },
    }
  }

  // No personal wound: lean on what they believe about this person.
  const conviction: Belief | undefined = listener
    ? damagingBeliefsAbout(world, speaker, listener.id)[0]
    : undefined
  if (conviction) {
    return {
      text: pick([
        `Everyone whispers it, so I'll say it plain: ${conviction.claim}.`,
        `Deny it all you like — ${conviction.claim}, and half the town knows.`,
      ]),
      category: 'hostile',
      citedBelief: {
        id: conviction.id,
        claim: conviction.claim,
        confidence: conviction.confidence,
        source: conviction.source,
      },
    }
  }

  const lines = [
    `Everyone smiles at you, ${first(listener?.name ?? '')}, and talks behind your back. I'm just doing it to your face.`,
    job
      ? `Call yourself a ${job}? This town deserves better.`
      : `At least the rest of us work for our bread.`,
    `Your whole house walks around like it owns the square.`,
  ]
  return { text: pick(lines), category: 'hostile' }
}

function apologyLine(ctx: SpeechContext): { text: string; category: SpeechCategory } {
  const { event, listener, pick } = ctx
  const restitution = (event.payload as { restitution?: number }).restitution ?? 0
  const lines = restitution
    ? [
        `I was small about it, ${first(listener?.name ?? '')}. Take these — and the apology with them.`,
        `What I did wasn't right. Here, two coins, and my word it won't happen again.`,
      ]
    : [
        `I was wrong the other day, ${first(listener?.name ?? '')}. It's been sitting heavy on me.`,
        `Say what you like back — I owed you the words. I'm sorry.`,
        `No excuses. I shouldn't have done it.`,
      ]
  return { text: pick(lines), category: 'kind' }
}

function repayLine(ctx: SpeechContext): { text: string; category: SpeechCategory } {
  const { event, pick } = ctx
  const amount = (event.payload as { amount?: number }).amount ?? 0
  return {
    text: pick([
      `${amount} coins, paid in full. We're square now.`,
      `I don't like owing, you know that. Count it.`,
      `Take what's yours before I find a way to spend it.`,
    ]),
    category: 'status',
  }
}

function helpLine(ctx: SpeechContext): ComposedLine {
  const { world, speaker, event, listener, pick } = ctx
  const gift = (event.payload as { gift?: string }).gift ?? ''

  // Kindness remembers kindness: cite the debt of warmth if one exists.
  const warmth: Memory | undefined = listener
    ? peekMemories(world, speaker, { aboutAgentId: listener.id, tones: ['warm'], limit: 1 })[0]
    : undefined
  if (warmth) {
    return {
      text: pick([
        `After what you did for me ${daysAgo(world, warmth.createdDay)}, this is the least of it. Eat.`,
        `You fed me once when you didn't have to. Take it, ${first(listener!.name)}.`,
      ]),
      category: 'kind',
      citedMemory: { id: warmth.id, summary: warmth.summary, day: warmth.createdDay },
    }
  }

  const lines = gift.includes('bread')
    ? [
        `Here — eat. You look hollow, ${first(listener?.name ?? '')}.`,
        `Take the loaf. You'd do the same.`,
      ]
    : [
        `Take the coins, settle up when the work comes back.`,
        `It's two coins, not a kingdom. Go eat.`,
      ]
  return { text: pick(lines), category: 'kind' }
}

function caughtLine(ctx: SpeechContext): { text: string; category: SpeechCategory } {
  const { pick } = ctx
  return {
    text: pick([
      `It's not what it looks like—`,
      `I was going to pay. I was.`,
      `A loaf. One loaf. Look how they feed us and tell me I'm the thief.`,
    ]),
    category: 'alarm',
  }
}

function statusLine(ctx: SpeechContext): { text: string; category: SpeechCategory } | null {
  const { event, world, pick } = ctx
  const payload = event.payload as { kind?: string; total?: number; progress?: number }
  switch (event.type) {
    case 'gathered_materials':
      return {
        text: pick([
          `Timber's heavy and the day's long.`,
          `Good stone wants finding. Found some.`,
          `Another load for the ${payload.kind ?? 'site'}.`,
        ]),
        category: 'status',
      }
    case 'construction_progress':
      return {
        text: pick([
          `She's coming up straight and true.`,
          `Walls by week's end, roof if the weather holds.`,
        ]),
        category: 'status',
      }
    case 'building_completed': {
      const placeName = event.locationId ? world.places[event.locationId]?.name : 'it'
      return {
        text: pick([
          `Done! ${placeName} stands!`,
          `Last nail's in. Somebody fetch the fiddle.`,
        ]),
        category: 'status',
      }
    }
    case 'border_dispute':
      return {
        text: pick([
          `That fence line moved in the night, and fences don't walk.`,
          `Tell your kin to keep to their side of the stones.`,
        ]),
        category: 'hostile',
      }
    default:
      return null
  }
}

// ---------------------------------------------------------------------------
// Motive analysis — why they said it, from both people's state

function motiveFor(ctx: SpeechContext): Utterance['motive'] {
  const { event, speaker, listener, rel } = ctx
  const reasons: string[] = []
  const t = speaker.traits

  switch (event.type) {
    case 'insulted': {
      const resentment = rel?.resentment ?? 0
      reasons.push(`resentment toward ${first(listener?.name ?? 'them')}: ${(resentment * 100).toFixed(0)}%`)
      reasons.push(`pride ${(t.pride * 100).toFixed(0)}%, empathy ${(t.empathy * 100).toFixed(0)}%`)
      if ((event.payload as { publicly?: boolean }).publicly) reasons.push('said in front of witnesses')
      if (resentment >= 0.5 && t.pride >= 0.55) {
        return { kind: 'premeditated', summary: 'A grudge nursed for days, delivered on purpose where people could hear it.', reasons }
      }
      return { kind: 'venting', summary: 'Old resentment boiling over in the moment rather than a plan.', reasons }
    }
    case 'helped': {
      const witnesses = event.witnessIds.length
      reasons.push(`empathy ${(t.empathy * 100).toFixed(0)}%, pride ${(t.pride * 100).toFixed(0)}%`)
      if (rel?.kinship === 'household') {
        reasons.push('they are family')
        return { kind: 'genuine', summary: 'Family looks after family; no calculation in it.', reasons }
      }
      if (t.pride >= 0.65 && witnesses > 0) {
        reasons.push(`${witnesses} onlooker(s) saw the gift`)
        return { kind: 'performative', summary: 'Kind, but chosen for the audience — charity is also reputation.', reasons }
      }
      return { kind: 'genuine', summary: 'Plain kindness: they saw hunger and could spare something.', reasons }
    }
    case 'shared_rumor': {
      const payload = event.payload as { mutated?: boolean }
      const trust = rel?.trust ?? 0
      reasons.push(`trust in listener ${(trust * 100).toFixed(0)}%`)
      reasons.push(`sociability ${(t.sociability * 100).toFixed(0)}%, paranoia ${(t.paranoia * 100).toFixed(0)}%`)
      if (payload.mutated) {
        reasons.push('the story changed in the telling')
        return { kind: 'deceitful', summary: 'The retelling bent the story — embellished to land harder or serve the teller.', reasons }
      }
      if (trust >= 0.45) {
        return { kind: 'bonding', summary: 'Sharing a secret to pull the listener closer; gossip is currency.', reasons }
      }
      return { kind: 'genuine', summary: 'Passing on what they believe to be true, more warning than weapon.', reasons }
    }
    case 'apologized': {
      const payload = event.payload as { accepted?: boolean; publicly?: boolean; restitution?: number }
      reasons.push(`empathy ${(t.empathy * 100).toFixed(0)}%, pride ${(t.pride * 100).toFixed(0)}%`)
      if (payload.restitution) reasons.push(`offered ${payload.restitution} coins in restitution`)
      reasons.push(payload.accepted ? 'the apology was accepted' : 'the apology was rebuffed')
      if (listener && speaker.status + 8 < listener.status) {
        reasons.push(`${first(listener.name)} outranks them in standing`)
        return {
          kind: 'premeditated',
          summary: 'Fence-mending with someone who matters — contrition with an eye on the ledger.',
          reasons,
        }
      }
      if (t.pride >= 0.6 && payload.publicly) {
        reasons.push('delivered in front of witnesses')
        return {
          kind: 'performative',
          summary: 'The town needed to see this apology as much as the wronged party needed to hear it.',
          reasons,
        }
      }
      if (speaker.needs.belonging < 40) {
        reasons.push(`belonging ${speaker.needs.belonging.toFixed(0)}/100`)
        return { kind: 'desperate', summary: 'Too lonely to afford an enemy — repair at any price.', reasons }
      }
      return { kind: 'genuine', summary: 'Remorse that would not sit quietly; the words cost pride and were paid anyway.', reasons }
    }
    case 'debt_repaid': {
      reasons.push(`pride ${(t.pride * 100).toFixed(0)}%`)
      reasons.push(`trust in ${first(listener?.name ?? 'them')}: ${((rel?.trust ?? 0) * 100).toFixed(0)}%`)
      if (t.pride >= 0.55) {
        return { kind: 'genuine', summary: 'Pride could not carry the debt another day.', reasons }
      }
      return { kind: 'bonding', summary: 'Settling up to keep the friendship unencumbered.', reasons }
    }
    case 'talked': {
      reasons.push(`belonging ${speaker.needs.belonging.toFixed(0)}/100`)
      reasons.push(`affection for ${first(listener?.name ?? 'them')}: ${((rel?.affection ?? 0) * 100).toFixed(0)}%`)
      if (speaker.needs.belonging < 45) {
        return { kind: 'desperate', summary: 'Lonely and starved for company; almost anyone would have done.', reasons }
      }
      return { kind: 'bonding', summary: 'Ordinary warmth — keeping a friendship in good repair.', reasons }
    }
    case 'stole_item': {
      reasons.push(`hunger ${speaker.needs.hunger.toFixed(0)}/100, coins ${speaker.money}`)
      return { kind: 'desperate', summary: 'Caught red-handed and talking fast — the denial is hunger speaking.', reasons }
    }
    case 'border_dispute': {
      return { kind: 'premeditated', summary: 'A boundary challenge, rehearsed on the walk over.', reasons: ['disputed ground between the households'] }
    }
    default:
      reasons.push('routine work talk')
      return { kind: 'genuine', summary: 'Nothing hidden — narrating the day.', reasons }
  }
}

// ---------------------------------------------------------------------------
// Entry point

const SPEAKING_EVENTS = new Set([
  'talked',
  'shared_rumor',
  'insulted',
  'helped',
  'apologized',
  'debt_repaid',
  'gathered_materials',
  'construction_progress',
  'building_completed',
  'border_dispute',
])

/** Attach generated speech + motive to freshly emitted events that warrant it. */
export function decorateWithSpeech(world: World, eventIds: string[]): void {
  for (const id of eventIds) {
    const event = world.events.find((e) => e.id === id)
    if (!event) continue
    // A caught thief blurts a denial; an unseen one says nothing.
    const speaks =
      SPEAKING_EVENTS.has(event.type) ||
      (event.type === 'stole_item' && (event.payload as { caught?: boolean }).caught === true)
    if (!speaks) continue
    const utterance = composeUtterance(world, event)
    if (utterance) event.payload.speech = utterance
  }
}

export function composeUtterance(world: World, event: SimEvent): Utterance | null {
  const speaker = world.agents[event.actorIds[0]]
  if (!speaker) return null
  const listener = event.targetIds[0] ? world.agents[event.targetIds[0]] : undefined
  const rng = rngFor(world.seed, 'speech', event.id)
  const ctx: SpeechContext = {
    world,
    event,
    speaker,
    listener,
    rel: listener ? relationshipWith(speaker, listener.id) : undefined,
    pick: (arr) => rng.pick(arr),
  }

  let line: ComposedLine | null = null
  switch (event.type) {
    case 'talked':
      line = talkLine(ctx)
      break
    case 'shared_rumor':
      line = rumorLine(ctx)
      break
    case 'insulted':
      line = insultLine(ctx)
      break
    case 'helped':
      line = helpLine(ctx)
      break
    case 'apologized':
      line = apologyLine(ctx)
      break
    case 'debt_repaid':
      line = repayLine(ctx)
      break
    case 'stole_item':
      line = caughtLine(ctx)
      break
    default:
      line = statusLine(ctx)
  }
  if (!line) return null
  const motive = motiveFor(ctx)
  // Historical context: when the line cites a memory or belief, the motive
  // carries the receipt — the reasoning tree renders it as a link.
  if (line.citedMemory) {
    motive.reasons.push(`drawing on day ${line.citedMemory.day}: "${line.citedMemory.summary}"`)
  }
  if (line.citedBelief) {
    motive.reasons.push(
      `convinced (${Math.round(line.citedBelief.confidence * 100)}%): "${line.citedBelief.claim}"`,
    )
  }
  return { ...line, motive }
}
