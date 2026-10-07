using UnityEngine;

namespace EditorScripts
{
    /// <summary>
    /// The world's textures, generated rather than authored: gravel for the road,
    /// scrub for the terrain, bark and needles for the trees, stone for the rocks.
    ///
    /// The stage has always been four flat colours. Flat colour is not actually the
    /// period look it was standing in for — 90s console games leaned hard on texture
    /// precisely because geometry was scarce, and Rally/PS1_Lit has sampled _MainTex
    /// from the day it was written. Nothing was ever assigned to it.
    ///
    /// Two rules run through everything here:
    ///
    ///   * Every feature wraps. These are tiled across hundreds of metres of ground,
    ///     so any structure that does not wrap — a gradient, an edge-weighted wash —
    ///     repeats into a hard seam on a fixed grid and reads instantly as a texture
    ///     tile rather than as ground. All noise is sampled on a wrapping lattice.
    ///
    ///   * Point filtering and small dimensions. These are meant to be crunchy at
    ///     close range; that is the same call the car's livery makes.
    /// </summary>
    public static class WorldTextureBuilder
    {
        public const int Size = 256;

        /// <summary>Gravel: grey-brown, loose, with stones sitting proud of the surface.</summary>
        public static Texture2D BuildGravel()
        {
            var px = new Color[Size * Size];

            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float u = (x + 0.5f) / Size, v = (y + 0.5f) / Size;

                float coarse = Fbm(u, v, 5, 3, 601);
                float fine   = Fbm(u, v, 22, 2, 607);

                // A loose surface is not one colour: the fines are paler and dustier
                // than the stone they sit between.
                Color dust  = new Color(0.50f, 0.46f, 0.39f);
                Color stone = new Color(0.34f, 0.32f, 0.30f);
                Color c = Color.Lerp(stone, dust, Mathf.SmoothStep(0.35f, 0.72f, coarse));

                c *= 0.88f + fine * 0.26f;
                px[y * Size + x] = c;
            }

            Pebbles(px, 190, 613);
            Blur(px, 1);
            Grain(px, 0.055f, 619);
            return Pack("World_Gravel", px, TextureWrapMode.Repeat);
        }

        /// <summary>Dry scrub: olive green worn through to dirt in patches.</summary>
        public static Texture2D BuildScrub()
        {
            var px = new Color[Size * Size];

            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float u = (x + 0.5f) / Size, v = (y + 0.5f) / Size;

                float patch = Fbm(u, v, 4, 3, 701);
                float blade = Fbm(u * 1.0f, v * 1.0f, 26, 2, 709);

                Color grass = new Color(0.27f, 0.33f, 0.18f);
                Color dead  = new Color(0.41f, 0.39f, 0.22f);
                Color dirt  = new Color(0.34f, 0.29f, 0.21f);

                Color c = Color.Lerp(grass, dead, Mathf.SmoothStep(0.40f, 0.70f, patch));
                c = Color.Lerp(c, dirt, Mathf.SmoothStep(0.72f, 0.92f, patch));

                c *= 0.84f + blade * 0.34f;
                px[y * Size + x] = c;
            }

            Pebbles(px, 70, 719);
            Blur(px, 1);
            Grain(px, 0.05f, 727);
            return Pack("World_Scrub", px, TextureWrapMode.Repeat);
        }

        /// <summary>
        /// Two materials in one texture: bark on the left half, needles on the right.
        /// The trees are merged into a single mesh with a single material, so a trunk
        /// and a canopy have to come out of the same image. Clamped rather than
        /// repeated, and the halves are kept clear of the seam, because a wrapped
        /// fetch here would put needles on the trunk.
        /// </summary>
        public static Texture2D BuildFoliage()
        {
            var px = new Color[Size * Size];
            int half = Size / 2;

            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float v = (y + 0.5f) / Size;
                Color c;

                if (x < half)
                {
                    // Bark: vertical striations, because a trunk is fibrous and
                    // because it tells you which way is up at a glance.
                    float u = (x + 0.5f) / half;
                    float fibre = Fbm(u, v * 0.22f, 14, 2, 811);
                    float rough = Fbm(u, v, 7, 3, 817);

                    c = Color.Lerp(new Color(0.20f, 0.15f, 0.11f),
                                   new Color(0.38f, 0.30f, 0.22f), fibre);
                    c *= 0.86f + rough * 0.28f;
                }
                else
                {
                    // Needles: clumped, darker in the gaps between clumps.
                    float u = (x - half + 0.5f) / half;
                    float clump = Fbm(u, v, 6, 3, 823);
                    float fleck = Fbm(u, v, 24, 2, 827);

                    c = Color.Lerp(new Color(0.09f, 0.15f, 0.09f),
                                   new Color(0.22f, 0.33f, 0.16f), Mathf.SmoothStep(0.30f, 0.75f, clump));
                    c *= 0.82f + fleck * 0.36f;
                }

                px[y * Size + x] = c;
            }

            Blur(px, 1);
            Grain(px, 0.045f, 829);
            return Pack("World_Foliage", px, TextureWrapMode.Clamp);
        }

        /// <summary>Stone: grey, cracked, with brighter exposed faces.</summary>
        public static Texture2D BuildStone()
        {
            var px = new Color[Size * Size];

            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float u = (x + 0.5f) / Size, v = (y + 0.5f) / Size;

                float body  = Fbm(u, v, 5, 3, 907);
                float grit  = Fbm(u, v, 20, 2, 911);

                Color c = Color.Lerp(new Color(0.30f, 0.29f, 0.28f),
                                     new Color(0.50f, 0.49f, 0.46f), body);
                c *= 0.88f + grit * 0.24f;

                // Cracks: the thin dark band where the noise field crosses a level.
                float crack = Mathf.Abs(Fbm(u, v, 8, 2, 919) - 0.5f);
                c *= Mathf.Lerp(0.55f, 1f, Mathf.SmoothStep(0f, 0.045f, crack));

                px[y * Size + x] = c;
            }

            Blur(px, 1);
            Grain(px, 0.05f, 929);
            return Pack("World_Stone", px, TextureWrapMode.Repeat);
        }

        /// <summary>
        /// The distant ridge line. Almost featureless on purpose: it sits 400 m out
        /// under heavy fog, so anything detailed would be thrown away, and a vertical
        /// ramp is what actually sells aerial perspective — hills pale toward their base
        /// where there is more air between you and them.
        /// </summary>
        public static Texture2D BuildBackdrop()
        {
            var px = new Color[Size * Size];

            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float u = (x + 0.5f) / Size, v = (y + 0.5f) / Size;

                float texture = Fbm(u, v, 6, 2, 1013);
                Color low  = new Color(0.55f, 0.60f, 0.64f);
                Color high = new Color(0.31f, 0.36f, 0.42f);

                Color c = Color.Lerp(low, high, Mathf.SmoothStep(0.1f, 0.85f, v));
                c *= 0.93f + texture * 0.14f;
                px[y * Size + x] = c;
            }

            Blur(px, 2);
            return Pack("World_Backdrop", px, TextureWrapMode.Clamp);
        }

        // ══════════════════════════════════════════════════════════════
        //  PLUMBING
        // ══════════════════════════════════════════════════════════════
        static Texture2D Pack(string name, Color[] buf, TextureWrapMode wrap)
        {
            // Mipmaps on, point filtering kept. Those are not in tension: point sampling is
            // what keeps the surface crunchy where the car actually is, and the mip chain
            // only takes over once a texel is smaller than a pixel.
            //
            // Without the chain, ground seen at a grazing angle asks for a dozen texels per
            // pixel and gets whichever one it lands on, which crawls and fizzes as the car
            // moves. Real hardware of the period had no mipmaps and did exactly that — but
            // it also ran at 320x240, where the artefact is a few pixels wide. At this
            // resolution it does not read as period, it reads as broken.
            var tex = new Texture2D(Size, Size, TextureFormat.RGBA32, true)
            {
                name = name,
                filterMode = FilterMode.Trilinear,
                wrapMode = wrap,
                // The ground is the whole reason: a road stretching to the horizon is the
                // worst case for mip selection, and without anisotropy it goes to mush
                // several car lengths ahead.
                anisoLevel = 8
            };

            var px = new Color32[buf.Length];
            for (int i = 0; i < buf.Length; i++)
            {
                Color c = buf[i];
                px[i] = new Color32(
                    (byte)Mathf.Round(Mathf.Clamp01(c.r) * 255f),
                    (byte)Mathf.Round(Mathf.Clamp01(c.g) * 255f),
                    (byte)Mathf.Round(Mathf.Clamp01(c.b) * 255f),
                    255);
            }

            tex.SetPixels32(px);
            tex.Apply(true, false);   // true: actually generate the mip chain
            return tex;
        }

        /// <summary>Individual stones sitting proud of the surface: a lit top, a shadow under it.</summary>
        static void Pebbles(Color[] buf, int count, int seed)
        {
            for (int p = 0; p < count; p++)
            {
                int cx = Mathf.FloorToInt(Hash(p, 1, seed) * Size);
                int cy = Mathf.FloorToInt(Hash(p, 2, seed) * Size);
                float r = 0.9f + Hash(p, 3, seed) * 2.1f;
                float lit = 0.22f + Hash(p, 4, seed) * 0.42f;

                int ri = Mathf.CeilToInt(r) + 1;
                for (int dy = -ri; dy <= ri; dy++)
                for (int dx = -ri; dx <= ri; dx++)
                {
                    float d = Mathf.Sqrt(dx * dx + dy * dy);
                    if (d > r) continue;

                    int i = Wrap(cy + dy, Size) * Size + Wrap(cx + dx, Size);
                    // Brighter on the upper side, darker on the lower — one light
                    // direction, baked, so a flat polygon still reads as gravel.
                    float shade = dy > 0 ? lit : -lit * 0.7f;
                    buf[i] = new Color(buf[i].r + shade, buf[i].g + shade, buf[i].b + shade * 0.9f);
                }
            }
        }

        static void Blur(Color[] buf, int passes)
        {
            var tmp = new Color[buf.Length];

            for (int p = 0; p < passes; p++)
            {
                for (int y = 0; y < Size; y++)
                for (int x = 0; x < Size; x++)
                {
                    Color acc = new Color(0f, 0f, 0f, 0f);
                    for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        acc += buf[Wrap(y + dy, Size) * Size + Wrap(x + dx, Size)];

                    tmp[y * Size + x] = acc / 9f;
                }
                System.Array.Copy(tmp, buf, buf.Length);
            }
        }

        static void Grain(Color[] buf, float amount, int seed)
        {
            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float g = (Hash(x, y, seed) - 0.5f) * amount;
                int i = y * Size + x;
                buf[i] = new Color(buf[i].r + g, buf[i].g + g, buf[i].b + g, buf[i].a);
            }
        }

        // ─── Noise ───────────────────────────────────────────────────
        static int Wrap(int v, int p) => ((v % p) + p) % p;

        internal static float Hash(int x, int y, int seed)
        {
            unchecked
            {
                int h = x * 374761393 + y * 668265263 + seed * 1274126177;
                h = (h ^ (h >> 13)) * 1274126177;
                h ^= h >> 16;
                return (h & 0x7FFFFFFF) / 2147483647f;
            }
        }

        /// <summary>Value noise on a wrapping lattice, so every texture tiles against itself.</summary>
        static float Noise(float x, float y, int period, int seed)
        {
            int x0 = Mathf.FloorToInt(x), y0 = Mathf.FloorToInt(y);
            float fx = x - x0, fy = y - y0;
            fx = fx * fx * (3f - 2f * fx);
            fy = fy * fy * (3f - 2f * fy);

            int xa = Wrap(x0, period), xb = Wrap(x0 + 1, period);
            int ya = Wrap(y0, period), yb = Wrap(y0 + 1, period);

            float n00 = Hash(xa, ya, seed), n10 = Hash(xb, ya, seed);
            float n01 = Hash(xa, yb, seed), n11 = Hash(xb, yb, seed);

            return Mathf.Lerp(Mathf.Lerp(n00, n10, fx), Mathf.Lerp(n01, n11, fx), fy);
        }

        internal static float Fbm(float u, float v, int cells, int octaves, int seed)
        {
            float sum = 0f, amp = 1f, norm = 0f;
            int p = Mathf.Max(1, cells);

            for (int o = 0; o < octaves; o++)
            {
                sum  += amp * Noise(u * p, v * p, p, seed + o * 101);
                norm += amp;
                amp  *= 0.5f;
                p    *= 2;
            }

            return sum / norm;
        }
    }
}
