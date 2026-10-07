using UnityEngine;

namespace EditorScripts
{
    /// <summary>
    /// The two road surfaces the road shader blends across the width of the stage: the
    /// PACKED gravel the wheels have compacted into tracks, and the LOOSE gravel thrown up
    /// onto the crown and the edges. A forest gravel road is never one surface — it is both,
    /// in stripes, and those stripes are what make it read as a road that gets driven.
    ///
    /// 512 rather than the world textures' 256: the road fills the bottom half of every
    /// frame, and this is the PS2 step — more texels, perspective-correct, no swim.
    ///
    /// Colour follows the reference: warm tan-brown, not grey. The old gravel was a cool
    /// grey with bright speckles, which reads as concrete under a forest canopy.
    /// </summary>
    public static class RoadTextureBuilder
    {
        public const int Size = 512;

        /// <summary>
        /// Compacted gravel. Smooth, fine and a little darker, with the odd stone pressed
        /// flush, and faint streaks along the direction of travel. Alpha carries a
        /// low-frequency field the shader uses to wobble the tracks and break up the
        /// loose patches, so the road does not look ruled.
        /// </summary>
        public static Texture2D BuildPacked()
        {
            var px = new Color[Size * Size];
            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float u = (x + 0.5f) / Size, v = (y + 0.5f) / Size;
                float coarse = WorldTextureBuilder.Fbm(u, v, 4, 3, 1201);
                float fine = WorldTextureBuilder.Fbm(u, v, 48, 2, 1207);
                // Streaks run along v, the direction of travel: stretched noise, many
                // cells across and few along.
                float streak = WorldTextureBuilder.Fbm(u * 1f, v * 0.08f, 40, 2, 1213);

                Color dark = new Color(0.32f, 0.25f, 0.18f);
                Color pale = new Color(0.46f, 0.37f, 0.27f);
                Color c = Color.Lerp(dark, pale, Mathf.SmoothStep(0.30f, 0.75f, coarse));
                c *= 0.90f + fine * 0.16f + (streak - 0.5f) * 0.10f;
                c.a = WorldTextureBuilder.Fbm(u, v, 2, 3, 1217);
                px[y * Size + x] = c;
            }
            Stones(px, 900, 0.6f, 1.6f, 0.10f, 1223);
            Grain(px, 0.04f, 1229);
            return Pack("World_RoadPacked", px);
        }

        /// <summary>
        /// Loose gravel: a bed of stones in mixed tones — tan, grey, a rusty one now and
        /// then — with dark gaps between them. Higher contrast than the packed surface,
        /// which is what lets the eye pick out the crown from a distance.
        /// </summary>
        public static Texture2D BuildLoose()
        {
            var px = new Color[Size * Size];
            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float u = (x + 0.5f) / Size, v = (y + 0.5f) / Size;
                float coarse = WorldTextureBuilder.Fbm(u, v, 6, 3, 1301);
                Color c = Color.Lerp(new Color(0.34f, 0.30f, 0.25f), new Color(0.50f, 0.45f, 0.37f), coarse);
                c.a = 1f;
                px[y * Size + x] = c;
            }
            Stones(px, 5200, 1.2f, 3.6f, 0.30f, 1307);
            Grain(px, 0.05f, 1319);
            return Pack("World_RoadLoose", px);
        }

        /// <summary>
        /// The verge between the road and the trees: short grass, greener than the old scrub,
        /// worn through to dirt in patches, with small stones kicked off the road. Blade
        /// streaks run in v so the grass has a direction.
        /// </summary>
        public static Texture2D BuildVerge()
        {
            var px = new Color[Size * Size];
            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float u = (x + 0.5f) / Size, v = (y + 0.5f) / Size;
                float patch = WorldTextureBuilder.Fbm(u, v, 5, 3, 1401);
                float blade = WorldTextureBuilder.Fbm(u, v * 0.35f, 90, 2, 1409);
                float tuft = WorldTextureBuilder.Fbm(u, v, 22, 2, 1417);

                Color grass = new Color(0.25f, 0.33f, 0.15f);
                Color dry = new Color(0.42f, 0.40f, 0.22f);
                Color dirt = new Color(0.33f, 0.27f, 0.19f);
                Color c = Color.Lerp(grass, dry, Mathf.SmoothStep(0.35f, 0.75f, patch));
                c = Color.Lerp(c, dirt, Mathf.SmoothStep(0.70f, 0.90f, tuft * 0.6f + patch * 0.5f));
                c *= 0.74f + blade * 0.48f;
                c.a = 1f;
                px[y * Size + x] = c;
            }
            Stones(px, 260, 0.8f, 2.2f, 0.25f, 1423);
            Grain(px, 0.05f, 1427);
            return Pack("World_Verge", px);
        }

        /// <summary>
        /// Forest floor: needle litter and moss under the canopy. Dark, brown and soft —
        /// what the gaps between the trunks should show at their foot.
        /// </summary>
        public static Texture2D BuildForestFloor()
        {
            var px = new Color[Size * Size];
            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float u = (x + 0.5f) / Size, v = (y + 0.5f) / Size;
                float moss = WorldTextureBuilder.Fbm(u, v, 4, 3, 1501);
                float needles = WorldTextureBuilder.Fbm(u * 1.3f, v * 0.7f, 120, 2, 1507);
                Color litter = new Color(0.25f, 0.18f, 0.12f);
                Color green = new Color(0.18f, 0.24f, 0.12f);
                Color c = Color.Lerp(litter, green, Mathf.SmoothStep(0.55f, 0.80f, moss));
                c *= 0.70f + needles * 0.55f;
                c.a = 1f;
                px[y * Size + x] = c;
            }
            Grain(px, 0.06f, 1511);
            return Pack("World_ForestFloor", px);
        }

        /// <summary>
        /// Grass and fern, side by side: a tuft of blades on the left half and one fern
        /// frond on the right. Shape lives in alpha, for the cut-out shader; colour is mid
        /// green with the variation left to vertex tint, so one atlas makes a whole verge.
        ///
        /// The mip chain is built by hand. Averaged the usual way, a thin blade's alpha falls
        /// below the cut-out threshold a couple of levels down and the grass simply vanishes
        /// a few metres from the camera. Here each level keeps its coverage instead.
        /// </summary>
        public static Texture2D BuildVergeFoliage()
        {
            const int W = 512, H = 512, half = W / 2;
            var px = new Color[W * H];
            var fill = new Color(0.27f, 0.33f, 0.13f, 0f);   // what transparent texels bleed
            for (int i = 0; i < px.Length; i++) px[i] = fill;

            // ── Tuft: tapered blades fanning out from a narrow base, curving as they rise ──
            for (int b = 0; b < 80; b++)
            {
                float h = Mathf.Lerp(0.40f, 0.97f, Mathf.Sqrt(WorldTextureBuilder.Hash(b, 1, 1701))) * H;
                float x0 = half * Mathf.Lerp(0.30f, 0.70f, WorldTextureBuilder.Hash(b, 2, 1701));
                float lean = (x0 - half * 0.5f) * 1.6f;
                float bend = (WorldTextureBuilder.Hash(b, 3, 1701) - 0.5f) * 90f + lean;
                float w0 = Mathf.Lerp(2.5f, 5.5f, WorldTextureBuilder.Hash(b, 4, 1701));
                float tone = WorldTextureBuilder.Hash(b, 5, 1701);
                Color tip = tone < 0.7f ? Color.Lerp(new Color(0.42f, 0.52f, 0.20f), new Color(0.56f, 0.60f, 0.26f), tone / 0.7f)
                                        : new Color(0.66f, 0.60f, 0.34f);   // dry
                Color root = new Color(0.14f, 0.20f, 0.07f);
                for (float t = 0f; t <= 1f; t += 0.5f / h)
                {
                    float x = x0 + bend * t * t, y = t * h;
                    float r = w0 * Mathf.Pow(1f - t, 0.8f) * 0.5f + 0.35f;
                    Disc(px, W, H, 0, half, x, y, r, Color.Lerp(root, tip, Mathf.Pow(t, 0.7f)));
                }
            }

            // ── Fern: a curving stem with alternating leaflets, longest in the middle ──
            {
                float len = H * 0.96f;
                for (float t = 0f; t <= 1f; t += 0.4f / len)
                {
                    float x = half + half * 0.5f + 18f * t * t, y = 6f + t * len;
                    Disc(px, W, H, half, W, x, y, 2.2f * (1f - t) + 0.6f, new Color(0.20f, 0.28f, 0.10f));
                }
                for (int k = 0; k < 34; k++)
                {
                    float t = 0.06f + k / 34f * 0.9f;
                    float sx = half + half * 0.5f + 18f * t * t, sy = 6f + t * len;
                    float side = k % 2 == 0 ? -1f : 1f;
                    float leaf = 92f * Mathf.Pow(Mathf.Sin(Mathf.PI * Mathf.Min(1f, t * 1.15f)), 0.7f) * (1f - 0.35f * t);
                    float wLeaf = 7f * (1f - 0.4f * t);
                    Color green = Color.Lerp(new Color(0.22f, 0.38f, 0.10f), new Color(0.38f, 0.52f, 0.16f), WorldTextureBuilder.Hash(k, 7, 1709));
                    for (float u = 0f; u <= 1f; u += 0.5f / Mathf.Max(8f, leaf))
                    {
                        // Leaflets sweep up toward the tip.
                        float lx = sx + side * u * leaf * 0.86f, ly = sy + u * leaf * 0.5f;
                        float r = wLeaf * Mathf.Sin(Mathf.PI * Mathf.Min(1f, u * 1.05f + 0.05f)) * 0.5f + 0.4f;
                        Disc(px, W, H, half, W, lx, ly, r, green * (0.85f + 0.3f * u));
                    }
                }
            }

            var tex = new Texture2D(W, H, TextureFormat.RGBA32, true)
            {
                name = "World_VergeFoliage", wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Trilinear, anisoLevel = 4
            };
            Color[] level = px;
            int lw = W, lh = H;
            for (int m = 0; m < tex.mipmapCount; m++)
            {
                tex.SetPixels(level, m);
                if (lw == 1 && lh == 1) break;
                int nw = Mathf.Max(1, lw / 2), nh = Mathf.Max(1, lh / 2);
                var next = new Color[nw * nh];
                for (int y = 0; y < nh; y++)
                for (int x = 0; x < nw; x++)
                {
                    Color acc = Color.clear; float a = 0f;
                    for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                    {
                        Color c = level[Mathf.Min(lh - 1, y * 2 + dy) * lw + Mathf.Min(lw - 1, x * 2 + dx)];
                        acc += new Color(c.r * c.a, c.g * c.a, c.b * c.a, 0f);
                        a += c.a;
                    }
                    Color rgb = a > 0.001f ? acc / a : fill;
                    // Coverage-preserving: average alpha, boosted, so blades thin but survive.
                    next[y * nw + x] = new Color(rgb.r, rgb.g, rgb.b, Mathf.Clamp01(a / 4f * 1.55f));
                }
                level = next; lw = nw; lh = nh;
            }
            tex.Apply(false, false);
            return tex;
        }

        /// <summary>Paint a filled disc into one column band of the atlas.</summary>
        static void Disc(Color[] px, int w, int h, int x0, int x1, float cx, float cy, float r, Color c)
        {
            int ri = Mathf.CeilToInt(r);
            for (int y = Mathf.FloorToInt(cy) - ri; y <= Mathf.FloorToInt(cy) + ri; y++)
            for (int x = Mathf.FloorToInt(cx) - ri; x <= Mathf.FloorToInt(cx) + ri; x++)
            {
                if (x < x0 + 1 || x >= x1 - 1 || y < 0 || y >= h) continue;
                float dx = x + 0.5f - cx, dy = y + 0.5f - cy;
                float cover = Mathf.Clamp01(r + 0.5f - Mathf.Sqrt(dx * dx + dy * dy));
                if (cover <= 0f) continue;
                int i = y * w + x;
                Color under = px[i];
                float a = Mathf.Max(under.a, cover);
                Color mixed = under.a > 0f ? Color.Lerp(under, c, cover) : c;
                px[i] = new Color(mixed.r, mixed.g, mixed.b, a);
            }
        }

        /// <summary>
        /// Stones: each a rounded blob in its own tone, lit from above and shadowed below
        /// (one baked light direction, so a flat polygon still reads as lumpy), with a dark
        /// rim where it meets the ground.
        /// </summary>
        static void Stones(Color[] buf, int count, float minR, float maxR, float relief, int seed)
        {
            for (int p = 0; p < count; p++)
            {
                float cx = WorldTextureBuilder.Hash(p, 1, seed) * Size;
                float cy = WorldTextureBuilder.Hash(p, 2, seed) * Size;
                float r = Mathf.Lerp(minR, maxR, Mathf.Pow(WorldTextureBuilder.Hash(p, 3, seed), 1.8f));
                float tone = WorldTextureBuilder.Hash(p, 4, seed);
                float squash = Mathf.Lerp(0.7f, 1.0f, WorldTextureBuilder.Hash(p, 5, seed));

                Color stone = tone < 0.55f ? Color.Lerp(new Color(0.46f, 0.40f, 0.32f), new Color(0.60f, 0.54f, 0.44f), tone / 0.55f)
                            : tone < 0.9f ? Color.Lerp(new Color(0.42f, 0.42f, 0.40f), new Color(0.58f, 0.57f, 0.54f), (tone - 0.55f) / 0.35f)
                                          : new Color(0.52f, 0.36f, 0.24f);   // the occasional rusty one

                int ri = Mathf.CeilToInt(r) + 2;
                for (int dy = -ri; dy <= ri; dy++)
                for (int dx = -ri; dx <= ri; dx++)
                {
                    float ex = dx / r, ey = dy / (r * squash);
                    float d = Mathf.Sqrt(ex * ex + ey * ey);
                    int i = Wrap(Mathf.FloorToInt(cy) + dy, Size) * Size + Wrap(Mathf.FloorToInt(cx) + dx, Size);
                    Color under = buf[i];
                    if (d > 1.0f && d < 1.35f)
                    {
                        // Contact shadow, heavier on the lower side.
                        float k = (1.35f - d) / 0.35f * (dy < 0 ? 0.35f : 0.15f);
                        buf[i] = new Color(under.r * (1 - k), under.g * (1 - k), under.b * (1 - k), under.a);
                        continue;
                    }
                    if (d > 1.0f) continue;
                    float dome = Mathf.Sqrt(1f - d * d);
                    float light = 1f + relief * (ey * 0.9f + dome * 0.4f - 0.3f);
                    Color c = stone * light;
                    float edge = Mathf.SmoothStep(1.0f, 0.75f, d);
                    buf[i] = new Color(Mathf.Lerp(under.r, c.r, edge), Mathf.Lerp(under.g, c.g, edge),
                                       Mathf.Lerp(under.b, c.b, edge), under.a);
                }
            }
        }

        static void Grain(Color[] buf, float amount, int seed)
        {
            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                float g = (WorldTextureBuilder.Hash(x, y, seed) - 0.5f) * amount;
                int i = y * Size + x;
                buf[i] = new Color(buf[i].r + g, buf[i].g + g, buf[i].b + g, buf[i].a);
            }
        }

        static int Wrap(int v, int p) => ((v % p) + p) % p;

        static Texture2D Pack(string name, Color[] buf)
        {
            var tex = new Texture2D(Size, Size, TextureFormat.RGBA32, true)
            {
                name = name,
                filterMode = FilterMode.Trilinear,
                wrapMode = TextureWrapMode.Repeat,
                anisoLevel = 8
            };
            var px = new Color32[buf.Length];
            for (int i = 0; i < buf.Length; i++)
            {
                Color c = buf[i];
                px[i] = new Color32((byte)Mathf.Round(Mathf.Clamp01(c.r) * 255f), (byte)Mathf.Round(Mathf.Clamp01(c.g) * 255f),
                                    (byte)Mathf.Round(Mathf.Clamp01(c.b) * 255f), (byte)Mathf.Round(Mathf.Clamp01(c.a) * 255f));
            }
            tex.SetPixels32(px);
            tex.Apply(true, false);
            return tex;
        }
    }
}
