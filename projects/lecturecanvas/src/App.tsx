import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import type {
  TranscriptItem,
  VisualizerSettings,
  LectureSessionStats,
} from './types';
import { LexiconIndex } from './services/textProcessor';
import { SpeechService } from './services/speechRecognition';
import type { SpeechStatus } from './services/speechRecognition';
import { AudioRecorderService } from './services/audioRecorder';
import { DEMO_LECTURES } from './data/demoLecture';
import { KineticForceCloud } from './components/visualizers/KineticForceCloud';
import { EditorialMosaicCloud } from './components/visualizers/EditorialMosaicCloud';
import { CelestialGalaxy } from './components/visualizers/CelestialGalaxy';
import { OrbitalHolosphere } from './components/visualizers/OrbitalHolosphere';
import { VoronoiTessellation } from './components/visualizers/VoronoiTessellation';
import { CyberMatrixWaterfall } from './components/visualizers/CyberMatrixWaterfall';
import { PrismSunburst } from './components/visualizers/PrismSunburst';
import { QuantumHoneycomb } from './components/visualizers/QuantumHoneycomb';
import { ControlToolbar } from './components/ControlToolbar';
import { WordListSidebar } from './components/WordListSidebar';
import { TranscriptDrawer } from './components/TranscriptDrawer';
import { MetricsHUD } from './components/MetricsHUD';
import { AudioFileUploader } from './components/AudioFileUploader';
import { AudioPlaybackBar } from './components/AudioPlaybackBar';
import { Toast } from './components/Toast';
import type { ToastNotice } from './components/Toast';
import { THEMES } from './utils/themeStyles';

const SETTINGS_STORAGE_KEY = 'lecture_canvas_settings';

const DEFAULT_SETTINGS: VisualizerSettings = {
  mode: 'kinetic',
  theme: 'cyber',
  fontFamily: 'space',
  minWordLength: 3,
  minRepetitions: 1,
  maxWords: 45,
  includeFillers: false,
  enableRotation: true,
  gravityStrength: 1.0,
  micSensitivity: 2.0,
};

function loadSettings(): VisualizerSettings {
  try {
    const saved = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!saved) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(saved);
    // Guard against a stale or hand-edited payload leaving the app unrenderable.
    if (!parsed || typeof parsed !== 'object') return DEFAULT_SETTINGS;
    const merged = { ...DEFAULT_SETTINGS, ...parsed };
    if (!THEMES[merged.theme as keyof typeof THEMES]) merged.theme = DEFAULT_SETTINGS.theme;
    return merged;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function App() {
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  // The in-flight phrase carries the moment it arrived, so the word tally can date
  // it without calling Date.now() during render.
  const [interim, setInterim] = useState<{ text: string; at: number }>({ text: '', at: 0 });
  const interimText = interim.text;
  const [bannedWords, setBannedWords] = useState<Set<string>>(new Set());
  const [settings, setSettings] = useState<VisualizerSettings>(loadSettings);

  const [isRecording, setIsRecording] = useState(false);
  const [speechStatus, setSpeechStatus] = useState<SpeechStatus>('inactive');
  const [recordedAudioBlob, setRecordedAudioBlob] = useState<Blob | null>(null);
  const [audioPlaybackSrc, setAudioPlaybackSrc] = useState<string | null>(null);
  const [audioPlaybackName, setAudioPlaybackName] = useState<string | undefined>(undefined);
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const [sessionElapsedSeconds, setSessionElapsedSeconds] = useState(0);

  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isTranscriptOpen, setIsTranscriptOpen] = useState(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [notice, setNotice] = useState<ToastNotice | null>(null);

  // References
  const speechServiceRef = useRef<SpeechService | null>(null);
  const audioRecorderRef = useRef<AudioRecorderService | null>(null);
  const activeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const demoIntervalRef = useRef<number | null>(null);

  // Running word tally. Transcript lines are folded in once, as they arrive, so a
  // two-hour lecture never re-tokenizes its whole history on every speech event.
  const lexiconRef = useRef(new LexiconIndex());
  const ingestedCountRef = useRef(0);
  const lastIngestedIdRef = useRef<string | null>(null);
  const [lexiconVersion, setLexiconVersion] = useState(0);

  // Object URL currently handed to the playback bar, so it can be released.
  const playbackUrlRef = useRef<string | null>(null);

  const showNotice = useCallback((next: ToastNotice) => setNotice(next), []);

  const setPlaybackSource = useCallback((url: string | null, name?: string) => {
    if (playbackUrlRef.current && playbackUrlRef.current !== url) {
      URL.revokeObjectURL(playbackUrlRef.current);
    }
    playbackUrlRef.current = url;
    setAudioPlaybackSrc(url);
    setAudioPlaybackName(name);
  }, []);

  useEffect(() => {
    return () => {
      if (playbackUrlRef.current) URL.revokeObjectURL(playbackUrlRef.current);
    };
  }, []);

  // Persist settings
  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // Private browsing / quota — settings simply won't persist.
    }
  }, [settings]);

  // Mic gain is a live parameter on the existing recorder, not a reason to rebuild it.
  useEffect(() => {
    audioRecorderRef.current?.setGain(settings.micSensitivity);
  }, [settings.micSensitivity]);

  // Session timer ticker
  useEffect(() => {
    if (!isRecording || !sessionStartTime) return;
    const interval = window.setInterval(() => {
      setSessionElapsedSeconds(Math.floor((Date.now() - sessionStartTime) / 1000));
    }, 1000);
    return () => window.clearInterval(interval);
  }, [isRecording, sessionStartTime]);

  // Handle Speech Callbacks
  const handleFinalResult = useCallback((text: string) => {
    const newItem: TranscriptItem = {
      id: `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      text,
      timestamp: Date.now(),
      isFinal: true,
    };
    setTranscripts((prev) => [...prev, newItem]);
    setInterim({ text: '', at: 0 });
  }, []);

  const handleSpeechError = useCallback(
    (error: string) => showNotice({ message: error, tone: 'error' }),
    [showNotice]
  );

  // The service is built once; callbacks are read through a ref so that changing a
  // setting mid-lecture can never tear down a live recognizer or leak its watchdog.
  const handleInterimResult = useCallback((text: string) => {
    setInterim({ text, at: Date.now() });
  }, []);

  const callbacksRef = useRef({
    onFinalResult: handleFinalResult,
    onInterimResult: handleInterimResult,
    onError: handleSpeechError,
    onStatusChange: setSpeechStatus,
  });
  useEffect(() => {
    callbacksRef.current = {
      onFinalResult: handleFinalResult,
      onInterimResult: handleInterimResult,
      onError: handleSpeechError,
      onStatusChange: setSpeechStatus,
    };
  }, [handleFinalResult, handleInterimResult, handleSpeechError]);

  useEffect(() => {
    speechServiceRef.current = new SpeechService({
      onFinalResult: (text) => callbacksRef.current.onFinalResult(text),
      onInterimResult: (text) => callbacksRef.current.onInterimResult(text),
      onError: (error) => callbacksRef.current.onError(error),
      onStatusChange: (status) => callbacksRef.current.onStatusChange(status),
    });

    audioRecorderRef.current = new AudioRecorderService();

    const speech = speechServiceRef.current;
    const recorder = audioRecorderRef.current;
    return () => {
      speech.cleanup();
      void recorder.stopRecording();
      if (demoIntervalRef.current) window.clearInterval(demoIntervalRef.current);
    };
  }, []);

  // Fold newly-arrived transcript lines into the running tally.
  useEffect(() => {
    const index = lexiconRef.current;
    let start = ingestedCountRef.current;

    // If the list was cleared or replaced wholesale rather than appended to, the
    // running tally no longer describes it — rebuild from scratch.
    const stillContiguous =
      start === 0 ||
      (transcripts.length >= start && transcripts[start - 1]?.id === lastIngestedIdRef.current);
    if (!stillContiguous) {
      index.reset();
      start = 0;
    }
    if (start === transcripts.length) return;

    for (let i = start; i < transcripts.length; i++) {
      index.ingest(transcripts[i].text, transcripts[i].timestamp);
    }
    ingestedCountRef.current = transcripts.length;
    lastIngestedIdRef.current = transcripts[transcripts.length - 1]?.id ?? null;
    setLexiconVersion((v) => v + 1);
  }, [transcripts]);

  // The in-flight phrase is tallied separately so revisions never corrupt the total.
  const interimIndex = useMemo(() => {
    if (!interim.text.trim()) return undefined;
    const index = new LexiconIndex();
    index.ingest(interim.text, interim.at);
    return index;
  }, [interim]);

  const processedData = useMemo(() => {
    void lexiconVersion; // recompute whenever the tally advances
    // The tally is deliberately outside React state: it is large and append-only,
    // and `lexiconVersion` is what makes this memo recompute when it advances.
    // oxlint-disable-next-line react/refs
    return lexiconRef.current.snapshot(
      {
        minWordLength: settings.minWordLength,
        minRepetitions: settings.minRepetitions,
        maxWords: settings.maxWords,
        includeFillers: settings.includeFillers,
        bannedWords,
      },
      interimIndex
    );
  }, [
    lexiconVersion,
    interimIndex,
    settings.minWordLength,
    settings.minRepetitions,
    settings.maxWords,
    settings.includeFillers,
    bannedWords,
  ]);

  // An imported lecture has no wall-clock session, so its duration comes from the
  // span of the transcript timestamps instead of a timer that never started.
  const importedSpanSeconds = useMemo(() => {
    if (transcripts.length < 2) return 0;
    const span = transcripts[transcripts.length - 1].timestamp - transcripts[0].timestamp;
    return span > 0 ? span / 1000 : 0;
  }, [transcripts]);

  const sessionStats: LectureSessionStats = useMemo(() => {
    const durationSeconds = sessionElapsedSeconds > 0 ? sessionElapsedSeconds : importedSpanSeconds;
    // Below a few seconds the rate is meaningless — a single word would read as 600 wpm.
    const wordsPerMinute =
      durationSeconds >= 5 ? Math.round(processedData.totalWords / (durationSeconds / 60)) : 0;
    const repetitionRatio =
      processedData.contentWords > 0
        ? ((processedData.contentWords - processedData.uniqueWords) / processedData.contentWords) * 100
        : 0;

    return {
      durationSeconds: Math.round(durationSeconds),
      totalWords: processedData.totalWords,
      uniqueWords: processedData.uniqueWords,
      wordsPerMinute: Number.isFinite(wordsPerMinute) ? wordsPerMinute : 0,
      topKeyword: processedData.frequencies[0]?.text ?? '',
      repetitionRatio,
    };
  }, [processedData, sessionElapsedSeconds, importedSpanSeconds]);

  // Top keywords set for transcript highlighter
  const topKeywordsSet = useMemo(() => {
    return new Set(processedData.frequencies.slice(0, 15).map((w) => w.text.toLowerCase()));
  }, [processedData.frequencies]);

  // Audio frequency data getter for visualizer
  const getAudioData = useCallback(() => {
    return audioRecorderRef.current?.getAudioFrequencyData() || null;
  }, []);

  const stopDemoPlayback = useCallback(() => {
    if (demoIntervalRef.current) {
      window.clearInterval(demoIntervalRef.current);
      demoIntervalRef.current = null;
    }
  }, []);

  // Recording Toggle
  const handleToggleRecording = useCallback(async () => {
    stopDemoPlayback();

    if (!isRecording) {
      audioRecorderRef.current?.setGain(settings.micSensitivity);
      const audioSuccess = await audioRecorderRef.current?.startRecording();
      if (!audioSuccess) {
        showNotice({
          message: 'Could not access the microphone. Check browser permissions and try again.',
          tone: 'error',
        });
      }
      speechServiceRef.current?.start();
      setIsRecording(true);
      if (!sessionStartTime) setSessionStartTime(Date.now());
      return;
    }

    speechServiceRef.current?.stop();
    const blob = await audioRecorderRef.current?.stopRecording();
    if (blob) {
      setRecordedAudioBlob(blob);
      setPlaybackSource(
        URL.createObjectURL(blob),
        `Lecture-${new Date().toLocaleTimeString()}.webm`
      );
    }
    setIsRecording(false);
  }, [isRecording, sessionStartTime, settings.micSensitivity, setPlaybackSource, showNotice, stopDemoPlayback]);

  // Clear Session
  const handleClearSession = useCallback(() => {
    stopDemoPlayback();
    setTranscripts([]);
    setInterim({ text: '', at: 0 });
    setRecordedAudioBlob(null);
    setPlaybackSource(null);
    setSessionStartTime(null);
    setSessionElapsedSeconds(0);
  }, [setPlaybackSource, stopDemoPlayback]);

  // Load Demo Lecture
  const handleLoadDemo = useCallback(
    (demoIndex: number) => {
      handleClearSession();

      const sample = DEMO_LECTURES[demoIndex] || DEMO_LECTURES[0];
      const startedAt = Date.now();
      setSessionStartTime(startedAt);

      // Seed a few lines so the diagram has shape immediately, then stream the rest.
      const seedCount = Math.min(3, sample.transcriptParts.length);
      setTranscripts(
        sample.transcriptParts.slice(0, seedCount).map((text, idx) => ({
          id: `demo-${startedAt}-${idx}`,
          text,
          timestamp: startedAt - (seedCount - idx) * 4000,
          isFinal: true,
        }))
      );

      let lineIndex = seedCount;
      demoIntervalRef.current = window.setInterval(() => {
        if (lineIndex >= sample.transcriptParts.length) {
          stopDemoPlayback();
          return;
        }
        const text = sample.transcriptParts[lineIndex];
        const index = lineIndex;
        setTranscripts((prev) => [
          ...prev,
          { id: `demo-${startedAt}-${index}`, text, timestamp: Date.now(), isFinal: true },
        ]);
        lineIndex++;
      }, 2400);
    },
    [handleClearSession, stopDemoPlayback]
  );

  // Audio File Upload & Streaming Handlers
  const handleTranscriptLoaded = useCallback(
    (items: TranscriptItem[], audioFile?: File) => {
      setTranscripts(items);
      if (audioFile) {
        setPlaybackSource(URL.createObjectURL(audioFile), audioFile.name);
      }
      showNotice({
        message: `Loaded ${items.length} lecture segments. Diagram generated.`,
        tone: 'success',
      });
    },
    [setPlaybackSource, showNotice]
  );

  const handleNewSegment = useCallback((item: TranscriptItem) => {
    setTranscripts((prev) => [...prev, item]);
  }, []);

  // Exclude / Ban Word
  const handleExcludeWord = useCallback(
    (word: string) => {
      const norm = word.toLowerCase();
      setBannedWords((prev) => new Set(prev).add(norm));
      showNotice({
        message: `Excluded "${word}" from the diagram.`,
        tone: 'info',
        action: {
          label: 'Undo',
          onAction: () =>
            setBannedWords((prev) => {
              const next = new Set(prev);
              next.delete(norm);
              return next;
            }),
        },
      });
    },
    [showNotice]
  );

  const handleUnbanWord = useCallback((word: string) => {
    setBannedWords((prev) => {
      const next = new Set(prev);
      next.delete(word.toLowerCase());
      return next;
    });
  }, []);

  const handleAddCustomBan = useCallback((word: string) => {
    setBannedWords((prev) => new Set(prev).add(word.toLowerCase()));
  }, []);

  // Export Diagram as PNG
  const handleExportImage = useCallback(() => {
    const canvas = activeCanvasRef.current;
    if (!canvas) {
      showNotice({ message: 'Diagram canvas is not ready for export yet.', tone: 'error' });
      return;
    }

    canvas.toBlob((blob) => {
      if (!blob) {
        showNotice({ message: 'Failed to export the diagram image.', tone: 'error' });
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `lecture-word-cloud-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`;
      a.click();
      URL.revokeObjectURL(url);
      showNotice({ message: 'Word diagram exported as PNG.', tone: 'success' });
    }, 'image/png');
  }, [showNotice]);

  // Export Audio
  const handleExportAudio = useCallback(() => {
    if (!recordedAudioBlob) return;
    const url = URL.createObjectURL(recordedAudioBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lecture-recording-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.webm`;
    a.click();
    URL.revokeObjectURL(url);
  }, [recordedAudioBlob]);

  const updateSettings = useCallback((newSettings: Partial<VisualizerSettings>) => {
    setSettings((prev) => ({ ...prev, ...newSettings }));
  }, []);

  // Keyboard shortcuts: space toggles recording, T/W toggle the panels.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTyping =
        target?.isContentEditable ||
        ['INPUT', 'TEXTAREA', 'SELECT'].includes(target?.tagName ?? '');
      if (isTyping || e.metaKey || e.ctrlKey || e.altKey || isUploadModalOpen) return;

      if (e.code === 'Space') {
        e.preventDefault();
        void handleToggleRecording();
      } else if (e.key === 't' || e.key === 'T') {
        setIsTranscriptOpen((open) => !open);
      } else if (e.key === 'w' || e.key === 'W') {
        setIsSidebarOpen((open) => !open);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleToggleRecording, isUploadModalOpen]);

  const themeColors = THEMES[settings.theme];

  const visualizerProps = {
    words: processedData.frequencies,
    settings,
    onExcludeWord: handleExcludeWord,
    exportRef: activeCanvasRef,
  };

  return (
    <div
      className="flex flex-col h-screen w-screen overflow-hidden select-none"
      style={{ backgroundColor: themeColors.background }}
    >
      {/* Top Controls Toolbar */}
      <ControlToolbar
        isRecording={isRecording}
        speechStatus={speechStatus}
        onToggleRecording={handleToggleRecording}
        onClearSession={handleClearSession}
        onLoadDemo={handleLoadDemo}
        onExportImage={handleExportImage}
        onExportAudio={handleExportAudio}
        onOpenUploadModal={() => setIsUploadModalOpen(true)}
        hasAudioRecording={Boolean(recordedAudioBlob)}
        getAudioData={getAudioData}
        settings={settings}
        onUpdateSettings={updateSettings}
      />

      {/* Main Visualizer Area */}
      <main className="relative flex-1 w-full h-full overflow-hidden">
        {/* Floating Top Metrics HUD */}
        <div className="absolute top-4 right-4 z-20 pointer-events-auto">
          <MetricsHUD stats={sessionStats} theme={settings.theme} />
        </div>

        {/* Dynamic Visualizer Switcher (8 Modes) */}
        {settings.mode === 'kinetic' && <KineticForceCloud {...visualizerProps} />}
        {settings.mode === 'editorial' && <EditorialMosaicCloud {...visualizerProps} />}
        {settings.mode === 'celestial' && <CelestialGalaxy {...visualizerProps} />}
        {settings.mode === 'holosphere' && <OrbitalHolosphere {...visualizerProps} />}
        {settings.mode === 'voronoi' && <VoronoiTessellation {...visualizerProps} />}
        {settings.mode === 'matrix' && <CyberMatrixWaterfall {...visualizerProps} />}
        {settings.mode === 'sunburst' && <PrismSunburst {...visualizerProps} />}
        {settings.mode === 'honeycomb' && <QuantumHoneycomb {...visualizerProps} />}

        {/* Left Word Frequency & Banned Words Sidebar */}
        <WordListSidebar
          words={processedData.frequencies}
          bannedWords={bannedWords}
          onExcludeWord={handleExcludeWord}
          onUnbanWord={handleUnbanWord}
          onAddCustomBan={handleAddCustomBan}
          theme={settings.theme}
          isOpen={isSidebarOpen}
          onToggleOpen={() => setIsSidebarOpen(!isSidebarOpen)}
        />

        {/* Bottom Right Live Transcript Drawer */}
        <TranscriptDrawer
          transcripts={transcripts}
          interimText={interimText}
          theme={settings.theme}
          isOpen={isTranscriptOpen}
          onToggleOpen={() => setIsTranscriptOpen(!isTranscriptOpen)}
          topKeywords={topKeywordsSet}
        />

        {/* Audio Playback Bar for Recorded/Uploaded Audio */}
        <AudioPlaybackBar
          audioSrc={audioPlaybackSrc}
          audioFileName={audioPlaybackName}
          onClose={() => setPlaybackSource(null)}
          theme={settings.theme}
        />

        {/* Upload Audio & 2-Hour Lecture Modal */}
        <AudioFileUploader
          isOpen={isUploadModalOpen}
          onClose={() => setIsUploadModalOpen(false)}
          onTranscriptLoaded={handleTranscriptLoaded}
          onNewSegment={handleNewSegment}
          theme={settings.theme}
        />

        <Toast notice={notice} onDismiss={() => setNotice(null)} />
      </main>
    </div>
  );
}

export default App;
