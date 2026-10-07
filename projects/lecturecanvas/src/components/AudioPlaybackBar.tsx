import React, { useState, useRef } from 'react';
import { Play, Pause, Volume2, VolumeX, X, FastForward } from 'lucide-react';
import type { ColorTheme } from '../types';
import { THEMES } from '../utils/themeStyles';

interface Props {
  audioSrc: string | null;
  audioFileName?: string;
  onClose: () => void;
  theme: ColorTheme;
}

export const AudioPlaybackBar: React.FC<Props> = ({
  audioSrc,
  audioFileName,
  onClose,
  theme,
}) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  const themeColors = THEMES[theme];

  if (!audioSrc) return null;

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      // Autoplay policies can reject this; the onPlay/onPause handlers keep the
      // button honest either way.
      void audio.play().catch(() => undefined);
    } else {
      audio.pause();
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    const value = audioRef.current?.duration;
    // Streamed WebM recordings often report Infinity until fully buffered.
    setDuration(Number.isFinite(value) ? (value as number) : 0);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const target = Number(e.target.value);
    setCurrentTime(target);
    if (audioRef.current) {
      audioRef.current.currentTime = target;
    }
  };

  const cycleSpeed = () => {
    const speeds = [1, 1.25, 1.5, 2];
    const nextIdx = (speeds.indexOf(playbackRate) + 1) % speeds.length;
    const nextSpeed = speeds[nextIdx];
    setPlaybackRate(nextSpeed);
    if (audioRef.current) {
      audioRef.current.playbackRate = nextSpeed;
    }
  };

  const toggleMute = () => {
    if (audioRef.current) {
      audioRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs)) return '00:00';
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = Math.floor(secs % 60);
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div
      className="fixed bottom-4 left-1/2 -translate-x-1/2 z-30 w-[540px] max-w-[calc(100vw-2rem)] rounded-2xl border backdrop-blur-2xl shadow-2xl p-3 flex items-center gap-3 transition-all animate-in fade-in slide-in-from-bottom-4"
      style={{
        backgroundColor: themeColors.panelBg,
        borderColor: themeColors.borderColor,
        color: themeColors.textPrimary,
      }}
    >
      <audio
        ref={audioRef}
        src={audioSrc}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onLoadStart={() => {
          setIsPlaying(false);
          setCurrentTime(0);
          setDuration(0);
        }}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
      />

      {/* Play/Pause Button */}
      <button
        onClick={togglePlay}
        className="w-9 h-9 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black flex items-center justify-center shadow-lg shadow-cyan-500/20 transition-transform active:scale-95 flex-shrink-0"
      >
        {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
      </button>

      {/* Info & Scrubber */}
      <div className="flex-1 min-w-0 space-y-1">
        <div className="flex items-center justify-between text-[11px] leading-tight">
          <span className="font-semibold text-slate-200 truncate max-w-[200px]">
            {audioFileName || 'Recorded Classroom Audio'}
          </span>
          <span className="font-mono text-slate-400">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>

        {/* Seek slider */}
        <input
          type="range"
          min={0}
          max={duration || 100}
          disabled={duration === 0}
          aria-label="Seek"
          value={currentTime}
          onChange={handleSeek}
          className="w-full h-1.5 accent-cyan-400 cursor-pointer bg-white/10 rounded-lg appearance-none"
        />
      </div>

      {/* Speed Multiplier */}
      <button
        onClick={cycleSpeed}
        className="px-2 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-[11px] font-mono font-bold text-cyan-300 border border-white/5 flex items-center gap-1"
        title="Playback Speed"
      >
        <FastForward className="w-3 h-3" />
        <span>{playbackRate}×</span>
      </button>

      {/* Mute Toggle */}
      <button
        onClick={toggleMute}
        className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white"
        title="Toggle Mute"
      >
        {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4" />}
      </button>

      {/* Close Player */}
      <button
        onClick={onClose}
        className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white"
        title="Close Player"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
};
