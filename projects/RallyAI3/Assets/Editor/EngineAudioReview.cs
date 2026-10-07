using System;
using System.IO;
using UnityEngine;
using Audio;

namespace EditorScripts
{
    public static class EngineAudioReview
    {
        public static void Execute()
        {
            const int rate=48000, seconds=16;
            var voice=new RallyEngineVoice();
            voice.Reset(rate);
            string folder=".rally/visual-review/audio";
            Directory.CreateDirectory(folder);
            float peak=0; double square=0,mean=0;
            using (var stream=new BinaryWriter(File.Create(folder+"/rally-engine-preview.wav")))
            {
                int bytes=rate*seconds*2;
                stream.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); stream.Write(36+bytes);
                stream.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); stream.Write(16);
                stream.Write((short)1); stream.Write((short)1); stream.Write(rate); stream.Write(rate*2);
                stream.Write((short)2); stream.Write((short)16);
                stream.Write(System.Text.Encoding.ASCII.GetBytes("data")); stream.Write(bytes);
                for (int i=0;i<rate*seconds;i++)
                {
                    if (i%256==0)
                    {
                        float t=(float)i/rate;
                        int gear=t<2 ? 0 : t<5 ? 1 : t<8 ? 2 : 3;
                        float pull=t<2 ? 0 : t<11 ? 1 : 0;
                        bool shift=(t>=5 && t<5.14f)||(t>=8 && t<8.14f);
                        float rpm=t<2 ? 1400 : t<5 ? Mathf.Lerp(1800,6500,(t-2)/3) :
                            t<8 ? Mathf.Lerp(4300,6600,(t-5)/3) :
                            t<11 ? Mathf.Lerp(4500,6800,(t-8)/3) : Mathf.Lerp(6800,2200,(t-11)/5);
                        voice.SetState(rpm,pull,pull*.8f,shift,gear,true,.06f,256);
                    }
                    float sample=voice.Next()*.95f;
                    if (float.IsNaN(sample)||float.IsInfinity(sample)) throw new InvalidDataException("Non-finite audio");
                    peak=Mathf.Max(peak,Mathf.Abs(sample)); square+=sample*sample; mean+=sample;
                    stream.Write((short)(Mathf.Clamp(sample,-1,1)*32767));
                }
            }
            if (peak>.98f || peak<.02f) throw new InvalidDataException("Engine preview level outside review envelope");
            File.WriteAllText(folder+"/levels.txt",$"Peak {peak:F6}\nRMS {Math.Sqrt(square/(rate*seconds)):F6}\nDC {mean/(rate*seconds):F6}\n");
            Debug.Log("Engine audio preview exported; finite and bounded.");
        }
    }
}
