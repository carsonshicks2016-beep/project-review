using UnityEngine;

namespace EditorScripts
{
    public static class AuthoredCarLivery
    {
        public static Texture2D Build()
        {
            const int size = 1024;
            var texture = new Texture2D(size,size,TextureFormat.RGBA32,false) {name="Rally Sedan Livery"};
            Color32 blue = new Color32(12,48,137,255), gold = new Color32(233,193,50,255);
            Color32 dark = new Color32(17,23,32,255), white = new Color32(225,229,229,255);
            Color32[] swatches = {blue,dark,gold,white,new Color32(181,144,56,255),new Color32(29,30,31,255),new Color32(222,124,36,255),new Color32(166,22,32,255)};
            var pixels = new Color32[size*size];
            for (int y=0;y<size;y++) for (int x=0;x<size;x++)
            {
                float u=(x+.5f)/size,v=(y+.5f)/size,z=u*4.34f-2.21f;
                Color32 color=blue;
                if (v<.30f) color=swatches[Mathf.Min(7,x*8/size)];
                else if (v<.59f)
                {
                    float lateral=(v-.31f)/.275f*1.90f-.95f;
                    if (z>.83f && z<1.73f && Mathf.Abs(lateral)>.37f && Mathf.Abs(lateral)<.40f) color=gold;
                }
                else if (v>=.60f)
                {
                    float height=(v-.60f)/.40f*1.5f;
                    if (height<.34f) color=dark;
                    // Two tapered gold sweeps leave the sculpted wheel arches legible.
                    float sweep=.44f+.09f*(z+1.6f);
                    if (z>-.95f && z<1.65f && Mathf.Abs(height-sweep)<.023f) color=gold;
                    if (z>-1.8f && z<-.63f && Mathf.Abs(height-(.56f-.16f*(z+1.8f)))<.042f) color=gold;
                    if (z>-.20f && z<.36f && height>.57f && height<.79f) color=white;
                    if (z>-.15f && z<.31f && height>.62f && height<.745f)
                    {
                        float a=(z+.15f)/.46f,b=(height-.62f)/.125f;
                        bool digit=(a<.18f && b>.44f) || b>.80f || (b>.44f && b<.60f) || (a>.68f && b<.52f) || b<.15f;
                        if (digit) color=dark;
                    }
                }
                pixels[y*size+x]=color;
            }
            texture.SetPixels32(pixels);
            texture.Apply();
            return texture;
        }
    }
}
