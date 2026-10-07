/**
 * Deterministic seeded randomness for Civilia.
 *
 * Every random decision in the simulation draws from an Rng stream derived
 * from a label like `${worldSeed}:day:phase:agentId`. Streams are independent
 * of iteration order, so the same world seed always replays the same history.
 */

/** xmur3 string hash — turns an arbitrary string into a 32-bit seed. */
function xmur3(str: string): () => number {
  let h = 1779033703 ^ str.length
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353)
    h = (h << 13) | (h >>> 19)
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507)
    h = Math.imul(h ^ (h >>> 13), 3266489909)
    h ^= h >>> 16
    return h >>> 0
  }
}

/** mulberry32 PRNG — small, fast, good enough for a town. */
function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export class Rng {
  readonly label: string
  private gen: () => number

  constructor(label: string) {
    this.label = label
    const hash = xmur3(label)
    this.gen = mulberry32(hash())
  }

  /** Uniform float in [0, 1). */
  next(): number {
    return this.gen()
  }

  /** Uniform float in [min, max). */
  range(min: number, max: number): number {
    return min + (max - min) * this.next()
  }

  /** Uniform integer in [min, max] inclusive. */
  int(min: number, max: number): number {
    return min + Math.floor(this.next() * (max - min + 1))
  }

  /** True with probability p. */
  chance(p: number): boolean {
    return this.next() < p
  }

  /** Pick one element uniformly. Throws on empty array. */
  pick<T>(arr: readonly T[]): T {
    if (arr.length === 0) throw new Error(`Rng.pick on empty array (${this.label})`)
    return arr[Math.floor(this.next() * arr.length)]
  }

  /** Weighted pick. Weights must be non-negative with a positive sum. */
  weightedPick<T>(items: readonly T[], weights: readonly number[]): T {
    let total = 0
    for (const w of weights) total += w
    if (total <= 0) throw new Error(`Rng.weightedPick with non-positive total (${this.label})`)
    let roll = this.next() * total
    for (let i = 0; i < items.length; i++) {
      roll -= weights[i]
      if (roll <= 0) return items[i]
    }
    return items[items.length - 1]
  }
}

/** Derive an order-independent random stream from labeled parts. */
export function rngFor(...parts: (string | number)[]): Rng {
  return new Rng(parts.join(':'))
}
