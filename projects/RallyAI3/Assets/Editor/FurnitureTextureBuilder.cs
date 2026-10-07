using UnityEngine;

namespace EditorScripts
{
    /// <summary>
    /// One atlas for all the rally furniture, so every tape, stake, board, bale and banner on
    /// a stage draws with one material. Layout, in UV:
    ///
    ///   v 0.00-0.25   plain white | tape stripes | straw | wood        (u quarters)
    ///   v 0.25-0.50   chevron board (left half) | plain red (right half)
    ///   v 0.50-0.625  banner: STAGE START
    ///   v 0.625-0.75  banner: FLYING FINISH
    ///   v 0.75-0.875  banner: FOREST RALLY
    ///   v 0.875-1.00  banner: plain dark blue, for backs and pillars
    ///
    /// The lettering is a 5x7 block font drawn here rather than a font asset: the words are
    /// fixed and few, and pixel-block letters suit the look. The names are invented on
    /// purpose — no real sponsors or events.
    /// </summary>
    public static class FurnitureTextureBuilder
    {
        const int Size = 512;

        public static Texture2D Build()
        {
            var px = new Color[Size * Size];
            for (int i = 0; i < px.Length; i++) px[i] = Color.white;

            // ── Row 0: white | tape | straw | wood ──
            Fill(px, 0, 0, 128, 128, (x, y) => Color.white);
            Fill(px, 128, 0, 256, 128, (x, y) => ((x + y) / 16) % 2 == 0 ? new Color(0.85f, 0.10f, 0.08f) : new Color(0.97f, 0.97f, 0.95f));
            Fill(px, 256, 0, 384, 128, (x, y) =>
            {
                float u = x / 128f, v = y / 128f;
                float strand = WorldTextureBuilder.Fbm(u * 0.3f, v, 40, 2, 1801);
                float clump = WorldTextureBuilder.Fbm(u, v, 5, 3, 1807);
                return Color.Lerp(new Color(0.62f, 0.50f, 0.24f), new Color(0.90f, 0.78f, 0.44f), strand * 0.7f + clump * 0.3f);
            });
            Fill(px, 384, 0, 512, 128, (x, y) =>
            {
                float u = x / 128f, v = y / 128f;
                float grain = WorldTextureBuilder.Fbm(u * 0.15f, v, 30, 2, 1811);
                return Color.Lerp(new Color(0.36f, 0.25f, 0.15f), new Color(0.58f, 0.43f, 0.28f), grain);
            });

            // ── Row 1: chevron board, pointing toward +u (mirrored by UV for the other way) ──
            Fill(px, 0, 128, 256, 256, (x, y) =>
            {
                int lx = x, ly = y - 128;
                bool border = lx < 6 || lx >= 250 || ly < 6 || ly >= 122;
                if (border) return new Color(0.97f, 0.97f, 0.95f);
                // Three chevrons: |y - centre| tilted along x, repeating.
                float c = Mathf.Abs(ly - 64f) * 0.9f + lx;
                bool white = (Mathf.FloorToInt(c / 40f) % 2) == 1 && lx > 20 && lx < 236;
                return white ? new Color(0.97f, 0.97f, 0.95f) : new Color(0.82f, 0.10f, 0.08f);
            });
            Fill(px, 256, 128, 512, 256, (x, y) => new Color(0.82f, 0.10f, 0.08f));

            // ── Banners ──
            Banner(px, 256, "STAGE START", new Color(0.10f, 0.20f, 0.55f), Color.white);
            Banner(px, 320, "FLYING FINISH", new Color(0.82f, 0.10f, 0.08f), Color.white);
            Banner(px, 384, "FOREST RALLY", new Color(0.96f, 0.80f, 0.10f), new Color(0.10f, 0.10f, 0.12f));
            Fill(px, 0, 448, 512, 512, (x, y) => new Color(0.10f, 0.20f, 0.55f));

            var tex = new Texture2D(Size, Size, TextureFormat.RGBA32, true)
            {
                name = "World_Furniture", wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Trilinear, anisoLevel = 4
            };
            tex.SetPixels(px);
            tex.Apply(true, false);
            return tex;
        }

        static void Fill(Color[] px, int x0, int y0, int x1, int y1, System.Func<int, int, Color> f)
        {
            for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
                px[y * Size + x] = f(x, y);
        }

        /// <summary>A 512 x 64 band: background, a light inner border, centred block lettering.</summary>
        static void Banner(Color[] px, int y0, string text, Color background, Color ink)
        {
            Fill(px, 0, y0, Size, y0 + 64, (x, y) =>
            {
                int ly = y - y0;
                bool edge = ly < 4 || ly >= 60 || x < 4 || x >= Size - 4;
                return edge ? Color.Lerp(background, Color.white, 0.6f) : background;
            });

            const int scale = 6;                       // each font pixel is 6 x 6
            int advance = 6 * scale;                   // 5 wide + 1 gap
            int width = text.Length * advance - scale;
            int left = (Size - width) / 2, bottom = y0 + (64 - 7 * scale) / 2;
            for (int c = 0; c < text.Length; c++)
            {
                string[] glyph = Glyph(text[c]);
                if (glyph == null) continue;
                for (int row = 0; row < 7; row++)
                for (int col = 0; col < 5; col++)
                {
                    if (glyph[row][col] != '#') continue;
                    int gx = left + c * advance + col * scale;
                    int gy = bottom + (6 - row) * scale;   // row 0 is the top of the glyph
                    for (int dy = 0; dy < scale; dy++)
                    for (int dx = 0; dx < scale; dx++)
                        px[(gy + dy) * Size + gx + dx] = ink;
                }
            }
        }

        static string[] Glyph(char c)
        {
            switch (c)
            {
                case 'A': return new[] { ".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#" };
                case 'E': return new[] { "#####", "#....", "#....", "####.", "#....", "#....", "#####" };
                case 'F': return new[] { "#####", "#....", "#....", "####.", "#....", "#....", "#...." };
                case 'G': return new[] { ".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###." };
                case 'H': return new[] { "#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#" };
                case 'I': return new[] { "#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####" };
                case 'L': return new[] { "#....", "#....", "#....", "#....", "#....", "#....", "#####" };
                case 'N': return new[] { "#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#" };
                case 'O': return new[] { ".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###." };
                case 'R': return new[] { "####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#" };
                case 'S': return new[] { ".####", "#....", "#....", ".###.", "....#", "....#", "####." };
                case 'T': return new[] { "#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.." };
                case 'Y': return new[] { "#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.." };
                default: return null;   // space
            }
        }
    }
}
