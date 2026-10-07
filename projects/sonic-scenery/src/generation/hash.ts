/**
 * Stable string hashing for the generation core.
 *
 * cyrb53 is a fast, well-distributed 53-bit hash. We fold it down to an
 * unsigned 32-bit integer so it can seed the PRNG / simplex noise and be
 * stored in TerrainParams.seed. Pure and deterministic — no randomness, no I/O.
 */

/** cyrb53 — 53-bit hash of a string. Deterministic for a given (str, seed). */
export function cyrb53(str: string, seed = 0): number {
  let h1 = 0xdeadbeef ^ seed;
  let h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}

/**
 * Deterministic unsigned 32-bit seed from a track id. Stable across runs and
 * platforms (relies only on charCodeAt + Math.imul).
 */
export function seedFromTrackId(trackId: string): number {
  // Lower 32 bits of cyrb53, forced unsigned.
  return cyrb53(trackId) >>> 0;
}
