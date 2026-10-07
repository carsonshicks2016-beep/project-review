import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileAudio, AlertCircle, Loader2, Sparkles, X, Settings2 } from 'lucide-react';
import type { TranscriptItem, ColorTheme } from '../types';
import { THEMES } from '../utils/themeStyles';
import { parseTranscriptFile } from '../services/transcriptFiles';

/** Override with VITE_WHISPER_SERVER_URL when the server is not on this machine. */
const WHISPER_SERVER_URL =
  import.meta.env.VITE_WHISPER_SERVER_URL ?? 'http://127.0.0.1:5175';

const TEXT_TRANSCRIPT_PATTERN = /\.(txt|vtt|srt)$/i;

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onTranscriptLoaded: (items: TranscriptItem[], audioFile?: File) => void;
  onNewSegment: (item: TranscriptItem) => void;
  theme: ColorTheme;
}

export const AudioFileUploader: React.FC<Props> = ({
  isOpen,
  onClose,
  onTranscriptLoaded,
  onNewSegment,
  theme,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<string>('base.en');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [statusText, setStatusText] = useState('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [recentSegments, setRecentSegments] = useState<string[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const themeColors = THEMES[theme];

  // Abandon an in-flight upload if the component goes away.
  useEffect(() => () => abortControllerRef.current?.abort(), []);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isProcessing) onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen, isProcessing, onClose]);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setErrorMsg(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setErrorMsg(null);
    }
  };

  const handleCancel = () => {
    abortControllerRef.current?.abort();
    setIsProcessing(false);
    setProgressPercent(0);
    setStatusText('Cancelled');
  };

  const startTranscription = async () => {
    if (!file) return;

    // Plain text, SubRip and WebVTT transcripts are parsed locally, no server needed.
    if (TEXT_TRANSCRIPT_PATTERN.test(file.name)) {
      try {
        const items = parseTranscriptFile(file.name, await file.text());
        if (items.length === 0) {
          setErrorMsg('That transcript file contained no readable lines.');
          return;
        }
        onTranscriptLoaded(items);
        onClose();
      } catch (err) {
        console.error('Failed to parse transcript file:', err);
        setErrorMsg('Failed to parse the transcript file.');
      }
      return;
    }

    // Audio file transcription via local Whisper backend
    setIsProcessing(true);
    setProgressPercent(2);
    setStatusText('Connecting to local GPU Whisper server...');
    setErrorMsg(null);
    setRecentSegments([]);

    abortControllerRef.current = new AbortController();

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    const collectedItems: TranscriptItem[] = [];
    // All imported segments hang off one anchor so their relative spacing matches
    // the recording's own timeline.
    const transcriptBaseTime = Date.now();
    let completed = false;

    try {
      const response = await fetch(`${WHISPER_SERVER_URL}/api/transcribe`, {
        method: 'POST',
        body: formData,
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Response stream not readable');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const block of lines) {
          const trimmed = block.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const jsonStr = trimmed.slice(6);

          // Parsing is the only thing allowed to fail quietly here. An `error`
          // event from the server must escape to the outer handler and reach the
          // user, so it is raised outside this try block.
          let data: any;
          try {
            data = JSON.parse(jsonStr);
          } catch (e) {
            console.warn('Skipping unparseable SSE event:', e);
            continue;
          }

          if (data.type === 'error') {
            throw new Error(data.message || 'The transcription server reported an error.');
          }

          if (data.type === 'init' || data.type === 'metadata') {
            setStatusText(data.message);
            if (typeof data.percent === 'number') {
              setProgressPercent(Math.min(99, Math.round(data.percent)));
            }
          } else if (data.type === 'progress') {
            if (typeof data.percent === 'number') {
              setProgressPercent(Math.min(99, Math.round(data.percent)));
            }
            if (data.message) setStatusText(data.message);
          } else if (data.type === 'segment') {
            const newItem: TranscriptItem = {
              id: data.id || `seg-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
              text: data.text,
              // Anchor to the segment's real offset in the recording so the
              // session duration reflects the lecture, not the import.
              timestamp: transcriptBaseTime + (data.start || 0) * 1000,
              isFinal: true,
            };
            collectedItems.push(newItem);
            onNewSegment(newItem);

            if (typeof data.percent === 'number') {
              setProgressPercent(Math.min(99, Math.round(data.percent)));
            }

            setRecentSegments((prev) => [data.text, ...prev.slice(0, 3)]);
            setStatusText(
              data.elapsedLabel
                ? `Transcribing lecture — ${data.elapsedLabel}`
                : 'Transcribing lecture...'
            );
          } else if (data.type === 'complete') {
            setProgressPercent(100);
            setStatusText('Transcription complete. Building diagram...');
            completed = true;
          }
        }
      }
      if (completed) {
        onTranscriptLoaded(collectedItems, file);
        setIsProcessing(false);
        onClose();
      } else {
        // The stream ended without a completion event — treat it as a failure
        // rather than leaving the dialog spinning forever.
        throw new Error('The transcription stream ended unexpectedly.');
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setStatusText('Cancelled by user');
      } else {
        console.error('Transcription error:', err);
        setErrorMsg(
          err.message ||
            `Could not reach the Whisper server at ${WHISPER_SERVER_URL}. Make sure "python3 server.py" is running.`
        );
      }
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !isProcessing) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="import-lecture-title"
        className="w-full max-w-lg rounded-3xl border shadow-2xl p-6 relative overflow-hidden transition-all"
        style={{
          backgroundColor: themeColors.panelBg,
          borderColor: themeColors.borderColor,
          color: themeColors.textPrimary,
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-white/10">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
              <FileAudio className="w-5 h-5" />
            </div>
            <div>
              <h2 id="import-lecture-title" className="font-bold text-sm text-white">
                Import Recorded Lecture
              </h2>
              <p className="text-[11px] text-slate-400">Transcribe audio files up to 2+ hours with Whisper</p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isProcessing}
            className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors disabled:opacity-30"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Upload Zone */}
        {!isProcessing && (
          <div className="mt-5 space-y-4">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="group border-2 border-dashed border-white/15 hover:border-cyan-400/60 rounded-2xl p-6 text-center cursor-pointer transition-all bg-white/[0.02] hover:bg-white/[0.05]"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,.mp3,.wav,.m4a,.webm,.aac,.flac,.mp4,.txt,.vtt,.srt"
                onChange={handleFileChange}
                className="hidden"
              />

              <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-cyan-500/10 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform">
                <Upload className="w-6 h-6" />
              </div>

              {file ? (
                <div>
                  <p className="font-bold text-sm text-white">{file.name}</p>
                  <p className="text-xs text-cyan-300 font-mono mt-0.5">
                    {formatFileSize(file.size)}
                  </p>
                  <p className="text-[10px] text-slate-400 mt-2">Click or drag another file to replace</p>
                </div>
              ) : (
                <div>
                  <p className="font-semibold text-sm text-slate-200">
                    Click to browse or drag & drop lecture audio
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Supports MP3, WAV, M4A, WEBM, FLAC, AAC, MP4, TXT, SRT
                  </p>
                  <p className="text-[10px] text-cyan-400/80 mt-2 font-medium">
                    ⚡ Handles 2+ hour full-length classroom recordings
                  </p>
                </div>
              )}
            </div>

            {/* Model Selection */}
            <div className="p-3 rounded-xl bg-black/30 border border-white/10 space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
                <Settings2 className="w-3.5 h-3.5 text-cyan-400" />
                <span>Whisper Model Speed vs Accuracy</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                {[
                  { id: 'tiny.en', label: 'Tiny.en', desc: 'Fastest (~1 min)' },
                  { id: 'base.en', label: 'Base.en', desc: 'Balanced (Rec.)' },
                  { id: 'small.en', label: 'Small.en', desc: 'Max Accuracy' },
                ].map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setModel(m.id)}
                    className={`p-2 rounded-lg text-left border transition-all ${
                      model === m.id
                        ? 'border-cyan-400 bg-cyan-500/10 text-white font-bold'
                        : 'border-white/5 bg-white/5 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="text-xs">{m.label}</div>
                    <div className="text-[10px] text-slate-400 font-normal">{m.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {errorMsg && (
              <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-500/15 border border-rose-500/30 text-rose-300 text-xs">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 rounded-xl border border-white/10 text-xs text-slate-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={!file}
                onClick={startTranscription}
                className="flex items-center gap-1.5 px-5 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 text-white font-bold text-xs shadow-lg shadow-cyan-500/20 hover:opacity-95 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Transcribe Lecture</span>
              </button>
            </div>
          </div>
        )}

        {/* Live Processing Screen */}
        {isProcessing && (
          <div className="mt-5 space-y-4">
            <div className="text-center py-2">
              <div className="inline-flex p-3 rounded-2xl bg-cyan-500/20 text-cyan-400 animate-pulse mb-3">
                <Loader2 className="w-7 h-7 animate-spin" />
              </div>
              <h3 className="font-bold text-sm text-white">Transcribing Classroom Lecture</h3>
              <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">{statusText}</p>
            </div>

            {/* Progress bar */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400 font-medium">GPU Acceleration (Apple Silicon)</span>
                <span className="font-mono font-bold text-cyan-400">{progressPercent}%</span>
              </div>
              <div className="h-2.5 w-full rounded-full bg-white/10 overflow-hidden relative">
                <div
                  className="h-full bg-gradient-to-r from-cyan-400 via-indigo-500 to-fuchsia-500 transition-all duration-300 rounded-full"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>

            {/* Live Streaming Segment Preview */}
            {recentSegments.length > 0 && (
              <div className="p-3 rounded-xl bg-black/40 border border-white/10 text-xs space-y-1">
                <div className="text-[10px] text-cyan-400 font-mono uppercase tracking-wide">
                  Live Streaming Transcript
                </div>
                <p className="text-slate-300 italic truncate">{recentSegments[0]}</p>
              </div>
            )}

            <div className="flex justify-end pt-2">
              <button
                type="button"
                onClick={handleCancel}
                className="px-4 py-2 rounded-xl bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 text-xs font-semibold transition-colors"
              >
                Cancel Transcription
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
