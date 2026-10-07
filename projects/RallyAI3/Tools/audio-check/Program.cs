// Offline render harness for the procedural audio. Drives the REAL synth classes through
// their real lifecycle against the shims in Shim.cs, writes WAVs, and measures the things
// that can only be checked numerically: level, headroom, NaN, and whether the spectrum
// actually contains what the comments claim it does.
using System;
using System.IO;
using UnityEngine;
using Unity.MLAgents;
using Core.Physics;
using Audio;

static class Probe
{
    public const int Fs = 48000;
    public const int Buf = 1024;

    class EngineP : EngineSynth
    {
        public void Boot() { this.AddSibling<AudioSource>(); OnEnable(); }
        public void Tick() { ReadState(); }
        public void Fill(float[] b) { Render(b); }
        public void FixedTick() { ReadFixedState(); }
        public void Pump(float[] b)
        {
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            typeof(ProceduralAudio).GetField("gateTarget", flags).SetValue(this, 1f);
            typeof(ProceduralAudio).GetMethod("OnAudioRead", flags).Invoke(this, new object[] { b });
        }
        public void GatedPump(float[] b)
        {
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            typeof(ProceduralAudio).GetMethod("Update", flags).Invoke(this, null);
            typeof(ProceduralAudio).GetMethod("OnAudioRead", flags).Invoke(this, new object[] { b });
        }
    }
    class TyreP : TyreSynth
    {
        public void Boot() { this.AddSibling<AudioSource>(); OnEnable(); }
        public void Tick() { ReadState(); }
        public void Fill(float[] b) { Render(b); }
    }
    class WindP : WindSynth
    {
        public void Boot() { this.AddSibling<AudioSource>(); OnEnable(); }
        public void Tick() { ReadState(); }
        public void Fill(float[] b) { Render(b); }
    }
    class ImpactP
    {
        readonly ImpactSynth voice = new ImpactSynth();
        public VehicleController vehicle { set { voice.vehicle = value; } }
        public void Boot() { voice.AddSibling<AudioSource>(); Call(voice, "OnEnable"); }
        public void Contact(Collision contact) { voice.Contact(contact); }
        public void FixedTick() { Call(voice, "ReadFixedState"); }
        public void Fill(float[] b) { Call(voice, "Render", b); }
    }
    class AmbienceP
    {
        readonly ForestAmbience voice = new ForestAmbience();
        public void Boot() { voice.AddSibling<AudioSource>(); Call(voice, "OnEnable"); }
        public void Fill(float[] b) { Call(voice, "Render", b); }
    }
    static void Call(ProceduralAudio voice, string method, float[] data = null)
    {
        typeof(ProceduralAudio).GetMethod(method, System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic).Invoke(voice, data == null ? null : new object[] { data });
    }

    static string outDir;

    static int Main()
    {
        outDir = Path.Combine(Directory.GetCurrentDirectory(), "out");
        Directory.CreateDirectory(outDir);
        AudioSettings.outputSampleRate = Fs;

        int fails = 0;
        fails += GateTest();
        fails += EventAndPerspectiveChecks();
        fails += RuntimeEventChecks();
        fails += ImpactAndRateChecks();
        fails += EngineSweep();
        fails += BoxerSpectrum();
        fails += VoiceCharacter();
        fails += RevsGlide();
        fails += TyreScenarios();
        fails += GravelDetail();
        fails += WindLadder();
        fails += MixBalance();
        fails += DriveDemo();

        Console.WriteLine();
        Console.WriteLine(fails == 0 ? "ALL CHECKS PASSED" : $"{fails} CHECK(S) FAILED");
        return fails == 0 ? 0 : 1;
    }

    // ─────────────────────────────────────────────────────────────────
    static int EventAndPerspectiveChecks()
    {
        Console.WriteLine("== Event handoff, perspectives and Doppler ==");
        int fails = 0;
        var queue = new AudioEventQueue();
        queue.Enqueue(VehicleSoundEvent.Shift, 1, 3);
        queue.Enqueue(VehicleSoundEvent.TurboRelease, .7f, 3);
        SoundEvent ev;
        fails += Expect("short shift retained before audio buffer", queue.TryDequeue(out ev) && ev.kind == VehicleSoundEvent.Shift, true);
        fails += Expect("short lift retained before audio buffer", queue.TryDequeue(out ev) && ev.kind == VehicleSoundEvent.TurboRelease, true);
        fails += Expect("events consumed once", queue.TryDequeue(out ev), false);
        for (int i = 0; i < 100; i++) queue.Enqueue(VehicleSoundEvent.Collision, i, 4);
        int received = 0;
        while (queue.TryDequeue(out ev)) received++;
        fails += Expect("bounded overflow reports dropped events", received == 31 && queue.Dropped == 69, true);
        fails += Expect("queue recovers after overflow", queue.Enqueue(VehicleSoundEvent.Shift, 1, 5) && queue.TryDequeue(out ev) && ev.generation == 5, true);
        fails += Expect("approach raises pitch", AudioMixProfile.Doppler(-30, 0) > 1, true);
        fails += Expect("recession lowers pitch", AudioMixProfile.Doppler(30, 0) < 1, true);
        fails += Expect("co-moving listener has unity pitch", Math.Abs(AudioMixProfile.Doppler(30, 30) - 1) < 1e-6, true);
        foreach (ListeningPerspective p in Enum.GetValues(typeof(ListeningPerspective)))
        {
            var mix = AudioMixProfile.For(p);
            bool external = p == ListeningPerspective.Trackside || p == ListeningPerspective.Helicopter;
            fails += Expect(p + " Doppler policy", mix.doppler, external);
            if (external) fails += Expect(p + " excludes onboard wind", mix.wind == 0 && mix.spatial == 1, true);
        }
        var voice = new RallyEngineVoice(); voice.Reset(Fs);
        voice.SetState(5000, 1, .8f, false, 2, true, .06f, 1024, false);
        voice.Trigger(VehicleSoundEvent.Shift, 1);
        voice.SetState(4500, 1, .8f, true, 3, true, .06f, 1024, false);
        fails += Expect("queued shift not duplicated by snapshot", voice.ShiftEvents == 1, true);
        voice.Reset(Fs);
        fails += Expect("reset clears engine events", voice.ShiftEvents == 0 && voice.ReleaseEvents == 0, true);
        foreach (int rate in new[] { 44100, 48000 })
        foreach (int size in new[] { 256, 1024, 4096 })
        {
            voice.Reset(rate); double peak = 0; bool finite = true;
            for (int offset = 0; offset < rate * 2; offset += size)
            {
                voice.SetState(1400 + 6000f * offset / (rate * 2), 1, .8f, false, 2, true, .06f, size);
                for (int i = 0; i < size; i++)
                {
                    float sample = voice.Next();
                    finite &= !float.IsNaN(sample) && !float.IsInfinity(sample);
                    peak = Math.Max(peak, Math.Abs(sample));
                }
            }
            fails += Expect("finite bounded engine " + rate + " / " + size, finite && peak < .98, true);
        }
        return fails;
    }

    static int RuntimeEventChecks()
    {
        Console.WriteLine("== Real engine handoff and reset fade ==");
        int fails = 0;
        var car = new VehicleController { currentRpm = 5000, throttleInput = 1, currentBoost = .8f, currentGear = 2 };
        var engine = new EngineP { vehicle = car }; engine.Boot(); engine.FixedTick();
        car.currentGear = 3; car.isShifting = true; engine.FixedTick();
        car.isShifting = false; car.throttleInput = 0; engine.FixedTick();
        car.throttleInput = 1; engine.FixedTick();
        float[] buffer = new float[4096]; engine.Pump(buffer);
        fails += Expect("sub-buffer shift captured once", engine.ShiftEvents == 1, true);
        fails += Expect("sub-buffer lift captured once", engine.ReleaseEvents == 1, true);
        engine.Pump(buffer);
        fails += Expect("steady snapshot does not repeat events", engine.ShiftEvents == 1 && engine.ReleaseEvents == 1, true);
        for (int i = 0; i < 8; i++)
        {
            engine.ResetPlayback();
            car.currentGear = 1; engine.FixedTick();
            engine.Pump(buffer); engine.Pump(buffer); engine.Pump(buffer);
            bool finite = true; double peak = 0;
            foreach (float sample in buffer) { finite &= !float.IsNaN(sample) && !float.IsInfinity(sample); peak = Math.Max(peak, Math.Abs(sample)); }
            fails += Expect("reset " + i + " clears transients and stays bounded",
                finite && peak < 1 && engine.ShiftEvents == 0 && engine.ReleaseEvents == 0, true);
        }
        car.currentGear = 2; engine.FixedTick();
        engine.ResetPlayback();
        car.currentGear = 1; engine.FixedTick();
        car.currentGear = 2; engine.FixedTick();
        engine.Pump(buffer); engine.Pump(buffer);
        fails += Expect("reset drops old events but retains new-generation shift", engine.ShiftEvents == 1, true);
        Time.timeScale = 0;
        for (int i = 0; i < 8; i++) engine.GatedPump(buffer);
        double pausedPeak = 0;
        foreach (float sample in buffer) pausedPeak = Math.Max(pausedPeak, Math.Abs(sample));
        fails += Expect("pause fades to silence", pausedPeak < 1e-4, true);
        Time.timeScale = 1;
        engine.GatedPump(buffer);
        fails += Expect("resume produces finite audio", Stats(buffer).bad == 0, true);
        Core.ML.LabRuntime.Enabled = true; Core.ML.LabRuntime.Config.viewer = false;
        var silent = new EngineP { vehicle = car }; silent.Boot();
        fails += Expect("managed non-viewer does not allocate audio playback", !silent.IsLive && silent.Source == null, true);
        Core.ML.LabRuntime.Enabled = false;
        return fails;
    }

    static int ImpactAndRateChecks()
    {
        Console.WriteLine("== Impacts, ambience and multi-rate combined output ==");
        int fails = 0;
        var landingCar = new VehicleController();
        var landingWheel = new DynamicSuspension { normalLoad = landingCar.mass * 9.81f };
        landingCar.suspensions = new[] { landingWheel };
        var body = landingCar.AddSibling<Rigidbody>();
        var landing = new ImpactP { vehicle = landingCar }; landing.Boot(); landing.FixedTick();
        landingWheel.isGrounded = false; body.linearVelocity = new Vector3(0, -5, 0);
        for (int i = 0; i < 8; i++) landing.FixedTick();
        landingWheel.isGrounded = true; landing.FixedTick();
        var landingBuffer = new float[1024]; landing.Fill(landingBuffer);
        fails += Expect("ground reacquisition after a fall excites landing sound", Stats(landingBuffer).rms > 0, true);
        for (int i = 0; i < 30; i++) { landing.FixedTick(); landing.Fill(landingBuffer); }
        fails += Expect("steady ground does not retrigger landings", Stats(landingBuffer).rms < 1e-5, true);
        foreach (int rate in new[] { 44100, 48000 })
        foreach (int size in new[] { 256, 1024, 4096 })
        {
            AudioSettings.outputSampleRate = rate;
            var car = new VehicleController { currentRpm = 6500, throttleInput = 1, currentBoost = .8f, currentSpeedKmh = 120 };
            var corner = new DynamicSuspension { normalLoad = car.mass * 9.81f, isGrounded = true };
            corner.AddSibling<PacejkaTireModel>().angularVelocity = 35f / corner.wheelRadius;
            car.suspensions = new[] { corner };
            var engine = new EngineP { vehicle = car }; engine.Boot(); engine.Tick();
            var tyres = new TyreP { vehicle = car }; tyres.Boot(); tyres.Tick();
            var wind = new WindP { vehicle = car }; wind.Boot(); wind.Tick();
            var impact = new ImpactP { vehicle = car }; impact.Boot();
            var ambient = new AmbienceP(); ambient.Boot();
            float[] e = new float[size], t = new float[size], w = new float[size], p = new float[size], a = new float[size];
            Time.unscaledTime += 1;
            impact.Contact(new Collision { relativeVelocity = new Vector3(0, 10, 0) });
            bool finite = true; double peak = 0, impactEnergy = 0;
            var clock = System.Diagnostics.Stopwatch.StartNew();
            for (int block = 0; block < rate / size; block++)
            {
                engine.Fill(e); tyres.Fill(t); wind.Fill(w); impact.Fill(p); ambient.Fill(a);
                for (int i = 0; i < size; i++)
                {
                    float mix = (e[i] * EngineSynth.DefaultVolume + t[i] * TyreSynth.DefaultVolume +
                        w[i] * WindSynth.DefaultVolume + p[i] * AudioMixProfile.ImpactVolume +
                        a[i] * AudioMixProfile.AmbienceVolume) * .7079458f;
                    finite &= !float.IsNaN(mix) && !float.IsInfinity(mix);
                    peak = Math.Max(peak, Math.Abs(mix)); impactEnergy += p[i] * p[i];
                }
            }
            clock.Stop();
            Console.WriteLine("   synthesis CPU " + rate + " / " + size + ": " + clock.Elapsed.TotalMilliseconds.ToString("F2") + " ms per ~1 s audio (offline, includes analysis)");
            fails += Expect("all five layers bounded " + rate + " / " + size, finite && peak < .98 && impactEnergy > 0, true);
        }
        AudioSettings.outputSampleRate = Fs;
        Time.deltaTime = Buf / (float)Fs;
        return fails;
    }

    static int GateTest()
    {
        Console.WriteLine("== AudioGate ==");
        int f = 0;
        Time.timeScale = 1f; Application.isBatchMode = false; Academy.IsInitialized = false;
        f += Expect("plain play mode runs", AudioGate.ShouldRun, true);

        Time.timeScale = 20f;
        f += Expect("time scale 20x silenced", AudioGate.ShouldRun, false);

        Time.timeScale = 0f;
        f += Expect("paused silenced", AudioGate.ShouldRun, false);

        Time.timeScale = 1f; Academy.IsInitialized = true; Academy.Instance.IsCommunicatorOn = true;
        f += Expect("trainer attached silenced", AudioGate.ShouldRun, false);

        Academy.IsInitialized = false; Application.isBatchMode = true;
        f += Expect("batch mode silenced", AudioGate.ShouldRun, false);
        Application.isBatchMode = false;
        Console.WriteLine();
        return f;
    }

    // ─────────────────────────────────────────────────────────────────
    static int EngineSweep()
    {
        Console.WriteLine("== Engine: idle, pull to the limiter, overrun ==");
        var vc = new VehicleController();
        var e = new EngineP { vehicle = vc };
        e.Boot();

        var all = new System.Collections.Generic.List<float>();
        var buf = new float[Buf];
        int buffers = (int)(11.0 * Fs / Buf);

        for (int b = 0; b < buffers; b++)
        {
            float t = b * (float)Buf / Fs;
            if (t < 2f) { vc.currentRpm = 1400f; vc.throttleInput = 0f; vc.currentBoost = 0.1f; }
            else if (t < 8f)
            {
                // Three gears' worth of pulls, with the torque cut between them.
                float u = (t - 2f) / 2f; float seg = u - (int)u;
                vc.currentRpm = 2200f + seg * 5400f;
                vc.throttleInput = 1f;
                vc.currentBoost = Math.Min(1f, seg * 2f);
                vc.isShifting = seg > 0.93f;
            }
            else { vc.currentRpm = Math.Max(1400f, 6800f - (t - 8f) * 1800f); vc.throttleInput = 0f; vc.isShifting = false; }

            e.Tick();
            e.Fill(buf);
            all.AddRange(buf);
        }

        float[] sig = all.ToArray();
        var s = Stats(sig);
        Report("engine sweep", s);
        Wav("engine_sweep.wav", sig);

        // Per-second peaks, so a clipping report says WHEN rather than just "somewhere".
        // t 0-2 idle, 2-8 three pulls with a torque cut at the top of each, 8-11 overrun.
        Console.Write("   peak by second ");
        for (int sec = 0; sec * Fs < sig.Length; sec++)
        {
            int len = Math.Min(Fs, sig.Length - sec * Fs);
            var chunk = new float[len];
            Array.Copy(sig, sec * Fs, chunk, 0, len);
            Console.Write($"{Stats(chunk).peak:0.00} ");
        }
        Console.WriteLine();

        int f = 0;
        f += Expect("no NaN/Inf", s.bad == 0, true);
        f += Expect("peak below full scale", s.peak < 1.0f, true);
        f += Expect("audible (rms > 0.02)", s.rms > 0.02f, true);
        f += Expect("headroom left (peak > 0.2)", s.peak > 0.2f, true);
        Console.WriteLine();
        return f;
    }

    // ─────────────────────────────────────────────────────────────────
    /// <summary>
    /// The claim under test: unequal pulse spacing puts energy at the HALF-ORDER, an
    /// octave below the firing frequency, and that subharmonic is the boxer burble.
    /// At 3000 rpm the cycle is 25 Hz, the half-order 50 Hz, the firing order 100 Hz.
    /// Same filter both times, so a change at 50 Hz can only have come from the source.
    /// </summary>
    static int BoxerSpectrum()
    {
        Console.WriteLine("== Engine: is the boxer rumble actually there? ==");
        float half = Render3000(0.06f, out float[] burbleSig);
        float even = Render3000(0.00f, out float[] evenSig);

        Wav("engine_3000_uneven.wav", burbleSig);
        Wav("engine_3000_even.wav", evenSig);

        Console.WriteLine($"   50 Hz half-order   uneven {Db(half),7:0.0} dB   even {Db(even),7:0.0} dB   " +
                          $"gain {Db(half) - Db(even),5:0.0} dB");

        int f = 0;
        f += Expect("uneven firing lifts the half-order by >6 dB", Db(half) - Db(even) > 6.0, true);
        Console.WriteLine();
        return f;
    }

    // ─────────────────────────────────────────────────────────────────
    /// <summary>
    /// The rally voice against the legacy one, on the same scripted drive: idle, a pull
    /// through two upshifts, then a lift. Claims under test: lifting off at high revs is
    /// a clearly different sound from pulling at the same revs, and a gear change cuts
    /// the drive audibly rather than just bending the pitch. Writes both voices so they
    /// can be compared by ear.
    /// </summary>
    static int VoiceCharacter()
    {
        Console.WriteLine("== Engine: rally voice against legacy ==");
        var rally  = DriveVoice(EngineSynth.VoiceProfile.Rally);
        var legacy = DriveVoice(EngineSynth.VoiceProfile.Legacy);
        Wav("engine_ab_rally.wav", rally);
        Wav("engine_ab_legacy.wav", legacy);

        // 5000 rpm on load (t 3.0-3.4) against 5000 rpm off load (t 7.0-7.4).
        float rLoad = Stats(Slice(rally, 3.0f, 0.4f)).rms, rLift = Stats(Slice(rally, 7.0f, 0.4f)).rms;
        float lLoad = Stats(Slice(legacy, 3.0f, 0.4f)).rms, lLift = Stats(Slice(legacy, 7.0f, 0.4f)).rms;
        // Shift at t 4.0: the 60 ms after it against the 200 ms before it.
        float rCut = Stats(Slice(rally, 4.0f, 0.06f)).rms, rPre = Stats(Slice(rally, 3.8f, 0.2f)).rms;

        Console.WriteLine($"   load/lift  rally {Db(rLoad) - Db(rLift),5:0.0} dB   legacy {Db(lLoad) - Db(lLift),5:0.0} dB");
        Console.WriteLine($"   shift cut  rally {Db(rCut) - Db(rPre),5:0.0} dB");
        var rs = Stats(rally); var ls = Stats(legacy);
        Report("rally drive", rs);
        Report("legacy drive", ls);

        int f = 0;
        f += Expect("lift-off at least 8 dB below load", Db(rLoad) - Db(rLift) > 8.0, true);
        f += Expect("shift cut at least 3 dB down", Db(rPre) - Db(rCut) > 3.0, true);
        f += Expect("legacy voice still renders", ls.bad == 0 && ls.rms > 0.02f && ls.peak < 1f, true);
        f += Expect("rally voice in range", rs.bad == 0 && rs.peak < 1f && rs.peak > 0.2f, true);
        Console.WriteLine();
        return f;
    }

    // ─────────────────────────────────────────────────────────────────
    /// <summary>
    /// A streaming AudioClip asks for audio in chunks of about 4096 samples (85 ms), and the
    /// car's state only reaches the voice once per chunk. If the voice snaps to each new
    /// value and holds it, a steady rev rise comes out as a staircase: change, freeze,
    /// change, freeze. Drive a perfectly linear rise in 4096-sample chunks and measure the
    /// longest stretch where the revs effectively stop moving.
    /// </summary>
    static int RevsGlide()
    {
        Console.WriteLine("== Engine: revs glide between state updates ==");
        const int chunk = 4096;
        var v = new RallyEngineVoice();
        v.Reset(Fs);
        int chunks = (int)(3.0 * Fs / chunk);
        var trace = new float[chunks * chunk];
        for (int c = 0; c < chunks; c++)
        {
            float t = c * (float)chunk / Fs;
            v.SetState(2000f + 1500f * t, 1f, 0.8f, false, 2, true, 0.06f, chunk);
            for (int i = 0; i < chunk; i++) { v.Next(); trace[c * chunk + i] = v.Rpm; }
        }

        // Expected slope is 1500 rpm/s. Count the longest run, after the first second,
        // where the revs move at under a fifth of that.
        float slowStep = 1500f / Fs * 0.2f;
        int longest = 0, run = 0;
        for (int i = Fs + 1; i < trace.Length; i++)
        {
            run = trace[i] - trace[i - 1] < slowStep ? run + 1 : 0;
            if (run > longest) longest = run;
        }
        float ms = longest * 1000f / Fs;
        Console.WriteLine($"   longest frozen stretch {ms,5:0.0} ms   (chunk {chunk * 1000f / Fs:0} ms)");

        int f = 0;
        f += Expect("no rev plateau longer than 10 ms", ms < 10f, true);
        Console.WriteLine();
        return f;
    }

    // ─────────────────────────────────────────────────────────────────
    /// <summary>
    /// The gravel detail voices — stone strikes and surface rumble — against the plain
    /// three-voice sound, on the same scenes over a rough surface. Claims under test: the
    /// detail is present on gravel, grows in a slide, all but vanishes on tarmac, and does
    /// not take over the mix or clip. Writes both versions of a short gravel run.
    /// </summary>
    static int GravelDetail()
    {
        Console.WriteLine("== Tyres: gravel detail (stone strikes, rumble) on vs off ==");
        var scenes = new (string name, float loose, float omega, float slip)[]
        {
            ("gravel 120 km/h", 0.6f, 101f, 1f),
            ("gravel big slide", 0.6f,  90f, 9f),
            ("tarmac 120 km/h", 0.0f, 101f, 0.6f),
        };
        var gain = new float[scenes.Length];
        var strikes = new float[scenes.Length];
        var onRun = new System.Collections.Generic.List<float>();
        var offRun = new System.Collections.Generic.List<float>();
        int f = 0;
        for (int k = 0; k < scenes.Length; k++)
        {
            var sc = scenes[k];
            float[] on = RoughRun(true, sc.loose, sc.omega, sc.slip, 2.5f, out int count);
            float[] off = RoughRun(false, sc.loose, sc.omega, sc.slip, 2.5f, out _);
            strikes[k] = count / 2.5f;
            if (sc.loose > 0f) { onRun.AddRange(on); offRun.AddRange(off); }
            var a = Stats(Slice(on, 0.5f, 1.9f)); var b = Stats(Slice(off, 0.5f, 1.9f));
            gain[k] = (float)(Db(a.rms) - Db(b.rms));
            Console.WriteLine($"   {sc.name,-18} off {Db(b.rms),6:0.0} dB  on {Db(a.rms),6:0.0} dB  " +
                              $"(+{gain[k]:0.0})   peak on {a.peak:0.000}   strikes {strikes[k],5:0.0}/s");
            if (a.bad != 0 || a.peak >= 1f) { Console.WriteLine("      !! NaN or clipped"); f++; }
        }
        Wav("gravel_detail_on.wav", onRun.ToArray());
        Wav("gravel_detail_off.wav", offRun.ToArray());

        f += Expect("detail present but not dominant on gravel (+0.5 to +3 dB)", gain[0] > 0.5f && gain[0] < 3f, true);
        f += Expect("stone strikes at 120 km/h: 8 to 25 a second", strikes[0] >= 8f && strikes[0] <= 25f, true);
        f += Expect("at least twice the strikes in a slide", strikes[1] >= 2f * strikes[0], true);
        f += Expect("no stone strikes on tarmac", strikes[2] == 0f, true);
        f += Expect("detail stays out of tarmac (< +1.5 dB)", gain[2] < 1.5f, true);
        Console.WriteLine();
        return f;
    }

    /// <summary>A steady run over a rough surface: suspension travel cycling 2 cm at 3 Hz.</summary>
    static float[] RoughRun(bool detail, float loose, float omega, float slip, float seconds, out int strikeCount)
    {
        var vc = new VehicleController();
        var sus = new DynamicSuspension[4];
        var tyre = new PacejkaTireModel[4];
        for (int i = 0; i < 4; i++) { sus[i] = new DynamicSuspension(); tyre[i] = sus[i].AddSibling<PacejkaTireModel>(); }
        vc.suspensions = sus;
        var t = new TyreP { vehicle = vc, gravelDetail = detail };
        t.Boot();
        float load = vc.mass * 9.81f / 4f;
        var buf = new float[Buf];
        var all = new System.Collections.Generic.List<float>();
        int n = (int)(seconds * Fs / Buf);
        for (int b = 0; b < n; b++)
        {
            float time = b * (float)Buf / Fs;
            for (int i = 0; i < 4; i++)
            {
                sus[i].isGrounded = true; sus[i].normalLoad = load; sus[i].surfaceLooseness = loose;
                sus[i].travel = 0.02f * (float)Math.Sin(2 * Math.PI * 3 * time + i);
                tyre[i].angularVelocity = omega; tyre[i].slipSpeed = slip;
            }
            t.Tick(); t.Fill(buf); all.AddRange(buf);
        }
        strikeCount = t.StrikeCount;
        return all.ToArray();
    }

    static float[] DriveVoice(EngineSynth.VoiceProfile voice)
    {
        var vc = new VehicleController();
        var e = new EngineP { vehicle = vc, voice = voice };
        e.Boot();
        var all = new System.Collections.Generic.List<float>();
        var buf = new float[Buf];
        int buffers = (int)(9.0 * Fs / Buf);
        for (int b = 0; b < buffers; b++)
        {
            float t = b * (float)Buf / Fs;
            vc.isShifting = false;
            if (t < 1.5f) { vc.currentRpm = 1400f; vc.throttleInput = 0f; vc.currentBoost = 0.1f; vc.currentGear = 1; }
            else if (t < 6.5f)
            {
                // 1.5-4.0 first gear to 6500, 4.0 upshift, 4.0-6.5 second gear back up.
                bool second = t >= 4f;
                float u = second ? (t - 4f) / 2.5f : (t - 1.5f) / 2.5f;
                vc.currentGear = second ? 2 : 1;
                vc.isShifting = second && t < 4.12f;
                vc.currentRpm = second ? 4700f + u * 1800f : 2000f + u * 4500f;
                vc.throttleInput = 1f;
                vc.currentBoost = Math.Min(1f, 0.3f + u);
            }
            else { vc.currentRpm = Math.Max(1400f, 6500f - (t - 6.5f) * 600f); vc.throttleInput = 0f; vc.currentBoost = 0.8f; }
            e.Tick();
            e.Fill(buf);
            all.AddRange(buf);
        }
        return all.ToArray();
    }

    static float[] Slice(float[] x, float start, float seconds)
    {
        int a = (int)(start * Fs), n = (int)(seconds * Fs);
        var r = new float[n];
        Array.Copy(x, a, r, 0, n);
        return r;
    }

    static float Render3000(float uneven, out float[] sig)
    {
        var vc = new VehicleController { currentRpm = 3000f, throttleInput = 1f, currentBoost = 0.8f };
        var e = new EngineP { vehicle = vc, unevenFiring = uneven, antiLagLevel = 0f };
        e.Boot();

        var buf = new float[Buf];
        int warm = (int)(0.5 * Fs / Buf);
        for (int b = 0; b < warm; b++) { e.Tick(); e.Fill(buf); }

        int n = (int)(2.0 * Fs / Buf);
        sig = new float[n * Buf];
        for (int b = 0; b < n; b++) { e.Tick(); e.Fill(buf); Array.Copy(buf, 0, sig, b * Buf, Buf); }

        // Report the firing order too, so a change in overall level cannot be mistaken
        // for a change in the half-order specifically.
        Console.WriteLine($"   uneven={uneven:0.00}  25Hz {Db(Mag(sig, 25)),6:0.0}  " +
                          $"50Hz {Db(Mag(sig, 50)),6:0.0}  100Hz {Db(Mag(sig, 100)),6:0.0}  " +
                          $"200Hz {Db(Mag(sig, 200)),6:0.0} dB");
        return Mag(sig, 50);
    }

    // ─────────────────────────────────────────────────────────────────
    static int TyreScenarios()
    {
        Console.WriteLine("== Tyres: surface, slip, and going airborne ==");
        var vc = new VehicleController();
        var sus = new DynamicSuspension[4];
        var tyre = new PacejkaTireModel[4];
        for (int i = 0; i < 4; i++)
        {
            sus[i] = new DynamicSuspension();
            tyre[i] = sus[i].AddSibling<PacejkaTireModel>();
        }
        vc.suspensions = sus;

        var t = new TyreP { vehicle = vc };
        t.Boot();

        float staticLoad = vc.mass * 9.81f / 4f;
        var buf = new float[Buf];
        var all = new System.Collections.Generic.List<float>();
        var rms = new System.Collections.Generic.List<float>();

        // name, seconds, looseness, wheel rad/s, slip m/s, grounded
        var scenes = new (string name, float secs, float loose, float omega, float slip, bool ground)[]
        {
            ("stationary",        1.0f, 0.6f,   0f,  0f, true),
            ("gravel 30 km/h",    2.0f, 0.6f,  25f,  0.4f, true),
            ("gravel 120 km/h",   2.0f, 0.6f, 101f,  1.0f, true),
            ("gravel big slide",  2.0f, 0.6f,  90f,  9f, true),
            ("tarmac 120 km/h",   2.0f, 0.0f, 101f,  0.6f, true),
            ("tarmac slide",      2.0f, 0.0f,  90f,  8f, true),
            ("locked wheels",     1.5f, 0.6f,   0f, 28f, true),
            ("airborne",          1.5f, 0.6f, 101f,  0f, false),
            ("landing",           1.5f, 0.6f, 101f,  3f, true),
        };

        int f = 0;
        foreach (var sc in scenes)
        {
            for (int i = 0; i < 4; i++)
            {
                sus[i].isGrounded = sc.ground;
                sus[i].normalLoad = sc.ground ? staticLoad : 0f;
                sus[i].surfaceLooseness = sc.loose;
                tyre[i].angularVelocity = sc.omega;
                tyre[i].slipSpeed = sc.slip;
            }

            int n = (int)(sc.secs * Fs / Buf);
            var seg = new System.Collections.Generic.List<float>();
            for (int b = 0; b < n; b++) { t.Tick(); t.Fill(buf); seg.AddRange(buf); all.AddRange(buf); }

            // Measure only the last 40% of each scene. Skipping a fixed 200 ms was not
            // enough: the level smoothing has a ~60 ms time constant, so a quiet scene
            // following a loud one was still reading the previous scene's decay tail and
            // "airborne" looked like it was leaking when it was not.
            float[] tail = seg.GetRange(seg.Count * 3 / 5, seg.Count - seg.Count * 3 / 5).ToArray();
            var st = Stats(tail);
            rms.Add(st.rms);
            Console.WriteLine($"   {sc.name,-18} rms {st.rms:0.0000}  peak {st.peak:0.000}  ({Db(st.rms),6:0.0} dB)");
            if (st.bad != 0) { Console.WriteLine("      !! NaN/Inf"); f++; }
            if (st.peak >= 1.0f) { Console.WriteLine("      !! clipped"); f++; }
        }

        Wav("tyres.wav", all.ToArray());

        f += Expect("stationary is silent", rms[0] < 0.002f, true);
        f += Expect("120 km/h louder than 30", rms[2] > rms[1] * 1.5f, true);
        f += Expect("gravel louder than tarmac at the same speed", rms[2] > rms[4] * 1.3f, true);
        f += Expect("a slide is louder than steady rolling", rms[3] > rms[2] * 1.2f, true);
        f += Expect("locked wheels still make noise", rms[6] > 0.02f, true);
        f += Expect("airborne is silent", rms[7] < 0.002f, true);
        f += Expect("landing comes back", rms[8] > 0.02f, true);
        Console.WriteLine();
        return f;
    }

    // ─────────────────────────────────────────────────────────────────
    static int WindLadder()
    {
        Console.WriteLine("== Wind: level against speed ==");
        var vc = new VehicleController();
        var w = new WindP { vehicle = vc };
        w.Boot();

        var buf = new float[Buf];
        var all = new System.Collections.Generic.List<float>();
        var rms = new System.Collections.Generic.List<float>();
        float[] kmh = { 0f, 30f, 60f, 100f, 140f, 180f };

        int f = 0;
        foreach (float k in kmh)
        {
            vc.currentSpeedKmh = k;
            vc.driftAngleDeg = 0f;
            int n = (int)(1.5f * Fs / Buf);
            var seg = new System.Collections.Generic.List<float>();
            for (int b = 0; b < n; b++) { w.Tick(); w.Fill(buf); seg.AddRange(buf); all.AddRange(buf); }
            float[] tail = seg.GetRange(Fs / 2, seg.Count - Fs / 2).ToArray();
            var st = Stats(tail);
            rms.Add(st.rms);
            Console.WriteLine($"   {k,3:0} km/h   rms {st.rms:0.0000}  peak {st.peak:0.000}  ({Db(st.rms),6:0.0} dB)");
            if (st.bad != 0) { Console.WriteLine("      !! NaN/Inf"); f++; }
            if (st.peak >= 1.0f) { Console.WriteLine("      !! clipped"); f++; }
        }

        // Sideways at speed
        vc.currentSpeedKmh = 120f; vc.driftAngleDeg = 38f;
        var sideSeg = new System.Collections.Generic.List<float>();
        for (int b = 0; b < (int)(1.5f * Fs / Buf); b++) { w.Tick(); w.Fill(buf); sideSeg.AddRange(buf); all.AddRange(buf); }
        var side = Stats(sideSeg.GetRange(Fs / 2, sideSeg.Count - Fs / 2).ToArray());
        Console.WriteLine($"   120 km/h sideways  rms {side.rms:0.0000}  ({Db(side.rms),6:0.0} dB)");

        Wav("wind.wav", all.ToArray());

        f += Expect("stationary is silent", rms[0] < 0.002f, true);
        f += Expect("60 km/h is quiet (< -30 dB)", Db(rms[2]) < -30.0, true);
        f += Expect("180 km/h is present (> -20 dB)", Db(rms[5]) > -20.0, true);
        f += Expect("30 -> 180 km/h spans more than 25 dB", Db(rms[5]) - Db(rms[1]) > 25.0, true);
        f += Expect("sideways is louder than straight", side.rms > rms[3] * 1.2f, true);
        Console.WriteLine();
        return f;
    }

    // ─────────────────────────────────────────────────────────────────
    /// <summary>
    /// The only question the per-source tests cannot answer: how the three sit against
    /// EACH OTHER. Rendered at one representative condition — flat out on gravel at
    /// 120 km/h with a bit of slip — with each source's own DefaultVolume applied, which
    /// is read from the synth classes rather than restated here so it cannot drift from
    /// what RallyCarBuilder assembles.
    ///
    /// What we want: the engine on top, the tyres clearly under it, the wind under that.
    /// A rally car is an engine you happen to be steering.
    /// </summary>
    static int MixBalance()
    {
        Console.WriteLine("== Mix balance: flat out on gravel at 120 km/h ==");
        const float Kmh = 120f, Rpm = 5000f;

        var vcE = new VehicleController { currentRpm = Rpm, throttleInput = 1f, currentBoost = 0.9f };
        var e = new EngineP { vehicle = vcE }; e.Boot();

        var vcT = new VehicleController();
        var sus = new DynamicSuspension[4];
        var tyre = new PacejkaTireModel[4];
        for (int i = 0; i < 4; i++) { sus[i] = new DynamicSuspension(); tyre[i] = sus[i].AddSibling<PacejkaTireModel>(); }
        vcT.suspensions = sus;
        foreach (var s in sus) { s.normalLoad = vcT.mass * 9.81f / 4f; s.surfaceLooseness = 0.6f; }
        foreach (var t in tyre) { t.angularVelocity = (Kmh / 3.6f) / 0.330f; t.slipSpeed = 1.5f; }
        var ty = new TyreP { vehicle = vcT }; ty.Boot();

        var vcW = new VehicleController { currentSpeedKmh = Kmh };
        var w = new WindP { vehicle = vcW }; w.Boot();

        float eR = SteadyRms(e.Tick, e.Fill) * EngineSynth.DefaultVolume;
        float tR = SteadyRms(ty.Tick, ty.Fill) * TyreSynth.DefaultVolume;
        float wR = SteadyRms(w.Tick, w.Fill) * WindSynth.DefaultVolume;

        Console.WriteLine($"   engine  {Db(eR),6:0.0} dB   (volume {EngineSynth.DefaultVolume:0.00})");
        Console.WriteLine($"   tyres   {Db(tR),6:0.0} dB   ({Db(tR) - Db(eR),5:0.0} dB vs engine)");
        Console.WriteLine($"   wind    {Db(wR),6:0.0} dB   ({Db(wR) - Db(eR),5:0.0} dB vs engine)");

        int f = 0;
        double dT = Db(tR) - Db(eR), dW = Db(wR) - Db(eR);
        f += Expect("engine leads the tyres by 4-12 dB", dT < -4.0 && dT > -12.0, true);
        f += Expect("wind sits below the tyres", dW < dT, true);
        f += Expect("wind still audible (within 20 dB of the engine)", dW > -20.0, true);
        Console.WriteLine();
        return f;
    }

    // ─────────────────────────────────────────────────────────────────
    /// <summary>
    /// A scripted stage run with all three sources summed at their shipping levels — the
    /// one output a person can actually judge. Every mechanism gets exercised: launch
    /// wheelspin, three upshifts with the torque cut, anti-lag on a trailing throttle,
    /// a long slide, a jump that goes silent on the tyres while the engine hits the
    /// limiter unloaded, and the landing.
    /// </summary>
    static int DriveDemo()
    {
        Console.WriteLine("== Scripted stage run (mixed) ==");

        var vc = new VehicleController();
        var sus = new DynamicSuspension[4];
        var tyre = new PacejkaTireModel[4];
        for (int i = 0; i < 4; i++) { sus[i] = new DynamicSuspension(); tyre[i] = sus[i].AddSibling<PacejkaTireModel>(); }
        vc.suspensions = sus;

        var e = new EngineP { vehicle = vc }; e.Boot();
        var ty = new TyreP { vehicle = vc }; ty.Boot();
        var w = new WindP { vehicle = vc }; w.Boot();

        var eb = new float[Buf]; var tb = new float[Buf]; var wb = new float[Buf];
        var mix = new System.Collections.Generic.List<float>();
        float corner = vc.mass * 9.81f / 4f;
        int buffers = (int)(23.0 * Fs / Buf);

        for (int b = 0; b < buffers; b++)
        {
            float t = b * (float)Buf / Fs;
            Stage(t, out float kmh, out float rpm, out float thr, out float slip,
                  out bool ground, out bool shift, out float drift, out float spinFactor);

            vc.currentSpeedKmh = kmh;
            vc.currentRpm = rpm;
            vc.throttleInput = thr;
            vc.isShifting = shift;
            vc.driftAngleDeg = drift;
            vc.currentBoost = thr > 0.5f ? 0.9f : 0.35f;

            for (int i = 0; i < 4; i++)
            {
                sus[i].isGrounded = ground;
                sus[i].normalLoad = ground ? corner : 0f;
                sus[i].surfaceLooseness = 0.6f;
                tyre[i].angularVelocity = (kmh / 3.6f) * spinFactor / sus[i].wheelRadius;
                tyre[i].slipSpeed = ground ? slip : 0f;
            }

            e.Tick(); e.Fill(eb);
            ty.Tick(); ty.Fill(tb);
            w.Tick(); w.Fill(wb);

            for (int i = 0; i < Buf; i++)
                mix.Add(eb[i] * EngineSynth.DefaultVolume
                      + tb[i] * TyreSynth.DefaultVolume
                      + wb[i] * WindSynth.DefaultVolume);
        }

        float[] sig = mix.ToArray();
        var s = Stats(sig);
        Report("stage run (mixed)", s);
        Wav("stage_run.wav", sig);

        int f = 0;
        f += Expect("no NaN/Inf", s.bad == 0, true);
        f += Expect("summed mix does not clip", s.peak < 1.0f, true);
        f += Expect("summed mix has headroom (peak < 0.9)", s.peak < 0.9f, true);
        Console.WriteLine();
        return f;
    }

    /// <summary>Timeline for the demo. spinFactor is wheel speed over body speed.</summary>
    static void Stage(float t, out float kmh, out float rpm, out float thr, out float slip,
                      out bool ground, out bool shift, out float drift, out float spinFactor)
    {
        ground = true; shift = false; drift = 0f; spinFactor = 1f; slip = 0.3f;

        if (t < 2f) { kmh = 0f; rpm = 1400f; thr = 0f; slip = 0f; }
        else if (t < 3.5f)                                   // launch, 1st, wheelspin
        {
            float u = (t - 2f) / 1.5f;
            kmh = 45f * u; rpm = 3000f + 3800f * u; thr = 1f;
            spinFactor = Lerp(2.6f, 1.15f, u); slip = Lerp(11f, 1.5f, u);
        }
        else if (t < 3.7f) { kmh = 45f; rpm = 4600f; thr = 1f; shift = true; }
        else if (t < 5.5f)                                   // 2nd
        { float u = (t - 3.7f) / 1.8f; kmh = Lerp(45f, 85f, u); rpm = Lerp(3800f, 7250f, u); thr = 1f; slip = 1.2f; }
        else if (t < 5.7f) { kmh = 85f; rpm = 5000f; thr = 1f; shift = true; }
        else if (t < 8f)                                     // 3rd
        { float u = (t - 5.7f) / 2.3f; kmh = Lerp(85f, 125f, u); rpm = Lerp(4900f, 7250f, u); thr = 1f; slip = 1.0f; }
        else if (t < 8.2f) { kmh = 125f; rpm = 5400f; thr = 1f; shift = true; }
        else if (t < 10.5f)                                  // 4th
        { float u = (t - 8.2f) / 2.3f; kmh = Lerp(125f, 155f, u); rpm = Lerp(5500f, 6800f, u); thr = 1f; slip = 0.9f; }
        else if (t < 12.5f)                                  // lift and brake: anti-lag
        {
            float u = (t - 10.5f) / 2f;
            kmh = Lerp(155f, 80f, u); rpm = Lerp(6800f, 3600f, u); thr = 0f;
            spinFactor = Lerp(1f, 0.55f, u); slip = Lerp(2f, 14f, u); drift = 12f * u;
        }
        else if (t < 16f)                                    // long slide
        {
            float u = (t - 12.5f) / 3.5f;
            kmh = Lerp(80f, 75f, u); rpm = 4500f; thr = 0.6f;
            slip = 8f; drift = Lerp(28f, 34f, u); spinFactor = 1.25f;
        }
        else if (t < 18.5f)                                  // power out
        {
            float u = (t - 16f) / 2.5f;
            kmh = Lerp(75f, 120f, u); rpm = Lerp(4500f, 6800f, u); thr = 1f;
            slip = Lerp(6f, 1.5f, u); drift = Lerp(26f, 5f, u); spinFactor = Lerp(1.3f, 1.05f, u);
        }
        else if (t < 19.6f)                                  // airborne, engine unloaded
        {
            float u = (t - 18.5f) / 1.1f;
            kmh = 120f; rpm = Lerp(6800f, 7600f, u); thr = 1f; ground = false; slip = 0f;
        }
        else                                                 // land and go
        {
            float u = Math.Min(1f, (t - 19.6f) / 2.4f);
            kmh = Lerp(120f, 145f, u); rpm = Lerp(6200f, 7000f, u); thr = 1f;
            slip = Lerp(7f, 1.2f, Math.Min(1f, u * 4f)); spinFactor = 1.05f;
        }
    }

    static float Lerp(float a, float b, float u) => a + (b - a) * (u < 0f ? 0f : (u > 1f ? 1f : u));

    static float SteadyRms(Action tick, Action<float[]> fill)
    {
        var buf = new float[Buf];
        for (int b = 0; b < (int)(1.0 * Fs / Buf); b++) { tick(); fill(buf); }   // settle
        var seg = new System.Collections.Generic.List<float>();
        for (int b = 0; b < (int)(2.0 * Fs / Buf); b++) { tick(); fill(buf); seg.AddRange(buf); }
        return Stats(seg.ToArray()).rms;
    }

    // ─────────────────────────────────────────────────────────────────
    //  UTILITY
    // ─────────────────────────────────────────────────────────────────

    struct S { public float rms, peak; public int bad; }

    static S Stats(float[] x)
    {
        double sum = 0; float peak = 0; int bad = 0;
        foreach (float v in x)
        {
            if (float.IsNaN(v) || float.IsInfinity(v)) { bad++; continue; }
            sum += (double)v * v;
            float a = Math.Abs(v);
            if (a > peak) peak = a;
        }
        return new S { rms = (float)Math.Sqrt(sum / Math.Max(1, x.Length)), peak = peak, bad = bad };
    }

    static void Report(string name, S s) =>
        Console.WriteLine($"   {name,-18} rms {s.rms:0.0000}  peak {s.peak:0.000}  ({Db(s.rms),6:0.0} dB)  NaN {s.bad}");

    static double Db(double v) => 20.0 * Math.Log10(Math.Max(1e-12, v));

    static float Mag(float[] x, double freq)
    {
        double re = 0, im = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double a = 2.0 * Math.PI * freq * i / Fs;
            re += x[i] * Math.Cos(a);
            im -= x[i] * Math.Sin(a);
        }
        return (float)(2.0 * Math.Sqrt(re * re + im * im) / x.Length);
    }

    static int Expect(string what, bool got, bool want)
    {
        bool ok = got == want;
        Console.WriteLine($"   [{(ok ? "PASS" : "FAIL")}] {what}");
        return ok ? 0 : 1;
    }

    static void Wav(string name, float[] x)
    {
        string path = Path.Combine(outDir, name);
        using (var fs = new FileStream(path, FileMode.Create))
        using (var w = new BinaryWriter(fs))
        {
        int bytes = x.Length * 2;
        w.Write(new[] { 'R', 'I', 'F', 'F' }); w.Write(36 + bytes);
        w.Write(new[] { 'W', 'A', 'V', 'E' });
        w.Write(new[] { 'f', 'm', 't', ' ' }); w.Write(16); w.Write((short)1); w.Write((short)1);
        w.Write(Fs); w.Write(Fs * 2); w.Write((short)2); w.Write((short)16);
        w.Write(new[] { 'd', 'a', 't', 'a' }); w.Write(bytes);
        foreach (float v in x)
        {
            float c = v > 1f ? 1f : (v < -1f ? -1f : v);
            if (float.IsNaN(c)) c = 0f;
            w.Write((short)(c * 32767f));
        }
        }
    }
}
