/**
 * Album-art → Palette extraction using node-vibrant (v4).
 *
 * We download the album art to a Buffer and feed it to Vibrant, then map its
 * named swatches onto the contract's {primary, secondary, accent, bg} shape.
 * Extraction is best-effort: if the art can't be fetched or analyzed we fall
 * back to a neutral palette so a track change is never blocked.
 */
import { Vibrant } from "node-vibrant/node";
import type { Palette } from "../contracts";

/** Neutral fallback used when art is missing or extraction fails. */
const FALLBACK_PALETTE: Palette = {
  primary: "#6b7280",
  secondary: "#9ca3af",
  accent: "#d1d5db",
  bg: "#1f2937",
};

interface SwatchLike {
  hex: string;
}

type VibrantPalette = Record<string, SwatchLike | null>;

/**
 * Picks the first available hex from an ordered list of swatch names,
 * falling back to the provided default.
 */
function pick(palette: VibrantPalette, names: string[], fallback: string): string {
  for (const name of names) {
    const swatch = palette[name];
    if (swatch && typeof swatch.hex === "string") return swatch.hex;
  }
  return fallback;
}

/**
 * Downloads album art at `artUrl` and extracts a Palette. Never throws —
 * returns a neutral fallback on any failure.
 */
export async function extractPalette(artUrl: string): Promise<Palette> {
  if (!artUrl) return { ...FALLBACK_PALETTE };
  try {
    const res = await fetch(artUrl);
    if (!res.ok) return { ...FALLBACK_PALETTE };
    const arrayBuf = await res.arrayBuffer();
    const buffer = Buffer.from(arrayBuf);

    const swatches = (await Vibrant.from(buffer).getPalette()) as VibrantPalette;

    return {
      primary: pick(swatches, ["Vibrant", "DarkVibrant", "Muted"], FALLBACK_PALETTE.primary),
      secondary: pick(
        swatches,
        ["LightVibrant", "Muted", "Vibrant"],
        FALLBACK_PALETTE.secondary,
      ),
      accent: pick(swatches, ["LightMuted", "DarkVibrant", "Vibrant"], FALLBACK_PALETTE.accent),
      bg: pick(swatches, ["DarkMuted", "DarkVibrant", "Muted"], FALLBACK_PALETTE.bg),
    };
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn(`[spotify] Palette extraction failed: ${(err as Error).message}`);
    return { ...FALLBACK_PALETTE };
  }
}
