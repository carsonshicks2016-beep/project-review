import React, { useEffect, useRef, useState } from 'react';
import type {
  VisualizerSettings,
  VisualizerMode,
  ColorTheme,
  FontFamily,
} from '../types';
import { THEMES, FONT_FAMILIES } from '../utils/themeStyles';
import { AudioVisualizerBar } from './AudioVisualizerBar';
import {
  Mic,
  Square,
  Sparkles,
  Image,
  SlidersHorizontal,
  Play,
  RotateCcw,
  Volume2,
  Upload,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Globe,
  Layers,
  Binary,
  Sun,
  Hexagon,
  Orbit,
  Grid3X3,
} from 'lucide-react';

import type { SpeechStatus } from '../services/speechRecognition';

interface Props {
  isRecording: boolean;
  speechStatus: SpeechStatus;
  onToggleRecording: () => void;
  onClearSession: () => void;
  onLoadDemo: (demoIndex: number) => void;
  onExportImage: () => void;
  onExportAudio: () => void;
  onOpenUploadModal: () => void;
  hasAudioRecording: boolean;
  getAudioData: () => Uint8Array | null;
  settings: VisualizerSettings;
  onUpdateSettings: (newSettings: Partial<VisualizerSettings>) => void;
}

export const ControlToolbar: React.FC<Props> = ({
  isRecording,
  speechStatus,
  onToggleRecording,
  onClearSession,
  onLoadDemo,
  onExportImage,
  onExportAudio,
  onOpenUploadModal,
  hasAudioRecording,
  getAudioData,
  settings,
  onUpdateSettings,
}) => {
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showDemoDropdown, setShowDemoDropdown] = useState(false);
  const [showVisualizerMenu, setShowVisualizerMenu] = useState(false);
  const headerRef = useRef<HTMLElement>(null);

  const themeColors = THEMES[settings.theme];

  // Dismiss the popovers on an outside click or Escape
  useEffect(() => {
    if (!showSettingsModal && !showDemoDropdown && !showVisualizerMenu) return;

    const onPointerDown = (e: MouseEvent) => {
      if (!headerRef.current?.contains(e.target as Node)) {
        setShowSettingsModal(false);
        setShowDemoDropdown(false);
        setShowVisualizerMenu(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowSettingsModal(false);
        setShowDemoDropdown(false);
        setShowVisualizerMenu(false);
      }
    };

    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [showSettingsModal, showDemoDropdown]);

  return (
    <header
      ref={headerRef}
      className="relative z-40 px-5 py-3 border-b flex items-center justify-between gap-4 backdrop-blur-xl transition-colors select-none"
      style={{
        backgroundColor: themeColors.panelBg,
        borderColor: themeColors.borderColor,
      }}
    >
      {/* Brand & Recording Action */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="font-extrabold text-sm tracking-tight text-white flex items-center gap-2">
              <span>LectureCanvas</span>
              <span className={`text-[10px] uppercase font-mono px-1.5 py-0.5 rounded border ${
                speechStatus === 'listening'
                  ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                  : speechStatus === 'paused'
                  ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                  : speechStatus === 'error'
                  ? 'bg-rose-500/20 text-rose-400 border-rose-500/30'
                  : 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
              }`}>
                {speechStatus === 'listening' ? 'Listening' : speechStatus === 'paused' ? 'Paused' : 'Ready'}
              </span>
            </h1>
            <p className="text-[10px] text-slate-400">Classroom Audio & Word Visualizer</p>
          </div>
        </div>

        {/* Primary Record Button */}
        <button
          onClick={onToggleRecording}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all shadow-md active:scale-95 ${
            isRecording
              ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/30'
              : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-cyan-500/30'
          }`}
        >
          {isRecording ? (
            <>
              <Square className="w-3.5 h-3.5 fill-white" />
              <span>Stop Recording</span>
            </>
          ) : (
            <>
              <Mic className="w-3.5 h-3.5" />
              <span>Start Listening</span>
            </>
          )}
        </button>

        {/* Input Audio File Button */}
        <button
          onClick={onOpenUploadModal}
          className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 transition-all shadow-sm active:scale-95"
          title="Input recorded audio file (up to 2+ hours)"
        >
          <Upload className="w-3.5 h-3.5" />
          <span>Input Audio</span>
        </button>

        {/* Live Audio Visualizer */}
        <AudioVisualizerBar
          getAudioData={getAudioData}
          isRecording={isRecording}
          theme={settings.theme}
        />
      </div>

      {/* Middle: Diagram Style Switcher with 8 modes */}
      <div className="relative">
        <div className="flex items-center gap-1 p-1 rounded-xl bg-black/40 border border-white/10 text-xs font-semibold">
          {/* Previous Mode Button */}
          <button
            onClick={() => {
              const modes: VisualizerMode[] = [
                'kinetic',
                'editorial',
                'celestial',
                'holosphere',
                'voronoi',
                'matrix',
                'sunburst',
                'honeycomb',
              ];
              const curIdx = modes.indexOf(settings.mode);
              const prevIdx = (curIdx - 1 + modes.length) % modes.length;
              onUpdateSettings({ mode: modes[prevIdx] });
            }}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
            title="Previous Visualizer"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>

          {/* Active Mode Trigger */}
          <button
            onClick={() => setShowVisualizerMenu(!showVisualizerMenu)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/10 text-cyan-300 hover:bg-white/15 transition-all shadow-sm"
          >
            {settings.mode === 'kinetic' && <Orbit className="w-3.5 h-3.5 text-cyan-400" />}
            {settings.mode === 'editorial' && <Layers className="w-3.5 h-3.5 text-amber-400" />}
            {settings.mode === 'celestial' && <Sparkles className="w-3.5 h-3.5 text-indigo-400" />}
            {settings.mode === 'holosphere' && <Globe className="w-3.5 h-3.5 text-emerald-400" />}
            {settings.mode === 'voronoi' && <Grid3X3 className="w-3.5 h-3.5 text-rose-400" />}
            {settings.mode === 'matrix' && <Binary className="w-3.5 h-3.5 text-green-400" />}
            {settings.mode === 'sunburst' && <Sun className="w-3.5 h-3.5 text-yellow-400" />}
            {settings.mode === 'honeycomb' && <Hexagon className="w-3.5 h-3.5 text-fuchsia-400" />}

            <span className="font-bold tracking-wide">
              {settings.mode === 'kinetic' && 'Kinetic Constellation'}
              {settings.mode === 'editorial' && 'Editorial Mosaic'}
              {settings.mode === 'celestial' && 'Celestial Spiral'}
              {settings.mode === 'holosphere' && '3D Holosphere'}
              {settings.mode === 'voronoi' && 'Voronoi Stained Glass'}
              {settings.mode === 'matrix' && 'Matrix Waterfall'}
              {settings.mode === 'sunburst' && 'Prism Sunburst'}
              {settings.mode === 'honeycomb' && 'Quantum Honeycomb'}
            </span>

            <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/40 text-slate-400 font-mono">
              8 Styles
            </span>

            <ChevronDown className="w-3 h-3 text-slate-400 ml-0.5" />
          </button>

          {/* Next Mode Button */}
          <button
            onClick={() => {
              const modes: VisualizerMode[] = [
                'kinetic',
                'editorial',
                'celestial',
                'holosphere',
                'voronoi',
                'matrix',
                'sunburst',
                'honeycomb',
              ];
              const curIdx = modes.indexOf(settings.mode);
              const nextIdx = (curIdx + 1) % modes.length;
              onUpdateSettings({ mode: modes[nextIdx] });
            }}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
            title="Next Visualizer"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* 8-Mode Dropdown Popover */}
        {showVisualizerMenu && (
          <div
            className="absolute left-1/2 -translate-x-1/2 top-full mt-2 w-96 rounded-2xl border backdrop-blur-2xl shadow-2xl p-3 z-50 animate-in fade-in"
            style={{
              backgroundColor: themeColors.panelBg,
              borderColor: themeColors.borderColor,
            }}
          >
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-white/10 text-xs">
              <span className="font-bold text-slate-300">Choose Artistic Diagram Mode</span>
              <span className="text-[10px] text-slate-500 font-mono">8 Visualizers</span>
            </div>

            <div className="grid grid-cols-2 gap-1.5">
              {[
                { id: 'kinetic', label: 'Kinetic Constellation', desc: 'Physics Bubble Collisions', icon: Orbit },
                { id: 'editorial', label: 'Editorial Mosaic', desc: 'Typographic Poster', icon: Layers },
                { id: 'celestial', label: 'Celestial Spiral', desc: 'Golden Ratio Orbit', icon: Sparkles },
                { id: 'holosphere', label: '3D Holosphere', desc: 'Interactive 3D Globe', icon: Globe },
                { id: 'voronoi', label: 'Voronoi Stained Glass', desc: 'Organic Polygons', icon: Grid3X3 },
                { id: 'matrix', label: 'Matrix Waterfall', desc: 'Cyber Digital Rain', icon: Binary },
                { id: 'sunburst', label: 'Prism Sunburst', desc: 'Radial Sector Dial', icon: Sun },
                { id: 'honeycomb', label: 'Quantum Honeycomb', desc: 'Crystal Hex Lattice', icon: Hexagon },
              ].map((item) => {
                const IconComponent = item.icon;
                const isSelected = settings.mode === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      onUpdateSettings({ mode: item.id as VisualizerMode });
                      setShowVisualizerMenu(false);
                    }}
                    className={`p-2.5 rounded-xl text-left border transition-all flex items-start gap-2.5 ${
                      isSelected
                        ? 'border-cyan-400/80 bg-cyan-500/15 shadow-sm'
                        : 'border-white/5 bg-white/5 hover:bg-white/10 text-slate-300'
                    }`}
                  >
                    <div
                      className={`p-1.5 rounded-lg ${
                        isSelected ? 'bg-cyan-500 text-black' : 'bg-white/10 text-slate-400'
                      }`}
                    >
                      <IconComponent className="w-3.5 h-3.5" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className={`text-xs font-bold truncate ${isSelected ? 'text-white' : 'text-slate-200'}`}>
                        {item.label}
                      </div>
                      <div className="text-[10px] text-slate-400 truncate mt-0.5">{item.desc}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Right Controls: Presets, Settings, Exports */}
      <div className="flex items-center gap-2">
        {/* Demo Lecture Dropdown */}
        <div className="relative">
          <button
            onClick={() => {
              setShowDemoDropdown((open) => !open);
              setShowSettingsModal(false);
            }}
            aria-expanded={showDemoDropdown}
            aria-haspopup="menu"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 text-xs font-semibold text-slate-300 transition-colors"
            title="Load a sample lecture to preview immediately"
          >
            <Play className="w-3.5 h-3.5 text-amber-400 fill-amber-400/30" />
            <span className="hidden sm:inline">Load Sample</span>
          </button>

          {showDemoDropdown && (
            <div
              className="absolute right-0 top-full mt-2 w-64 rounded-xl border backdrop-blur-xl shadow-2xl p-2 z-50 animate-in fade-in"
              style={{
                backgroundColor: themeColors.panelBg,
                borderColor: themeColors.borderColor,
              }}
            >
              <div className="text-[11px] font-bold text-slate-400 px-2 py-1 mb-1">
                Sample Lectures
              </div>
              <button
                onClick={() => {
                  onLoadDemo(0);
                  setShowDemoDropdown(false);
                }}
                className="w-full text-left px-2.5 py-2 rounded-lg hover:bg-white/10 text-xs text-slate-200 transition-colors"
              >
                <div className="font-semibold text-white">Astrophysics & Black Holes</div>
                <div className="text-[10px] text-slate-400">Physics, Spacetime, Gravity</div>
              </button>
              <button
                onClick={() => {
                  onLoadDemo(1);
                  setShowDemoDropdown(false);
                }}
                className="w-full text-left px-2.5 py-2 rounded-lg hover:bg-white/10 text-xs text-slate-200 transition-colors"
              >
                <div className="font-semibold text-white">Neural Networks & Deep Learning</div>
                <div className="text-[10px] text-slate-400">AI, Optimization, Attention</div>
              </button>
            </div>
          )}
        </div>

        {/* Settings Button */}
        <button
          onClick={() => {
            setShowSettingsModal((open) => !open);
            setShowDemoDropdown(false);
          }}
          aria-expanded={showSettingsModal}
          className="p-2 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 text-slate-300 transition-colors"
          title="Adjust Visual & NLP Settings"
        >
          <SlidersHorizontal className="w-4 h-4" />
        </button>

        {/* Clear / Reset */}
        <button
          onClick={onClearSession}
          className="p-2 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 text-slate-300 hover:text-rose-400 transition-colors"
          title="Reset Lecture Canvas"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        {/* Export Image PNG */}
        <button
          onClick={onExportImage}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 text-xs font-semibold transition-colors"
          title="Export high-resolution diagram image (PNG)"
        >
          <Image className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Export Art</span>
        </button>

        {/* Export Audio (if recorded) */}
        {hasAudioRecording && (
          <button
            onClick={onExportAudio}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-indigo-500/30 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 text-xs font-semibold transition-colors"
            title="Download recorded lecture audio"
          >
            <Volume2 className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Save Audio</span>
          </button>
        )}
      </div>

      {/* Settings Modal Drawer */}
      {showSettingsModal && (
        <div
          className="absolute right-5 top-full mt-2 w-80 rounded-2xl border backdrop-blur-2xl shadow-2xl p-4 z-50 space-y-4 animate-in fade-in"
          style={{
            backgroundColor: themeColors.panelBg,
            borderColor: themeColors.borderColor,
            color: themeColors.textPrimary,
          }}
        >
          <div className="flex items-center justify-between pb-2 border-b border-white/10">
            <h2 className="font-bold text-xs">Visualizer & NLP Settings</h2>
            <button
              onClick={() => setShowSettingsModal(false)}
              className="text-slate-400 hover:text-white text-xs"
            >
              ✕
            </button>
          </div>

          {/* Theme selection */}
          <div className="space-y-1.5">
            <label className="text-[11px] text-slate-400 font-semibold">Aesthetic Theme</label>
            <div className="grid grid-cols-2 gap-1.5">
              {(Object.keys(THEMES) as ColorTheme[]).map((thm) => (
                <button
                  key={thm}
                  onClick={() => onUpdateSettings({ theme: thm })}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-medium text-left border transition-all ${
                    settings.theme === thm
                      ? 'border-cyan-400 bg-white/10 text-white font-bold'
                      : 'border-white/5 bg-white/5 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {THEMES[thm].name}
                </button>
              ))}
            </div>
          </div>

          {/* Font selection */}
          <div className="space-y-1.5">
            <label className="text-[11px] text-slate-400 font-semibold">Typography</label>
            <select
              value={settings.fontFamily}
              onChange={(e) => onUpdateSettings({ fontFamily: e.target.value as FontFamily })}
              className="w-full px-2.5 py-1.5 rounded-lg bg-black/40 border border-white/10 text-xs text-white focus:outline-none focus:border-cyan-400"
            >
              {(Object.keys(FONT_FAMILIES) as FontFamily[]).map((f) => (
                <option key={f} value={f} className="bg-slate-900 text-white">
                  {FONT_FAMILIES[f].name}
                </option>
              ))}
            </select>
          </div>

          {/* Minimum Word Length */}
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-slate-400 font-semibold">Min Word Length</span>
              <span className="font-mono text-cyan-300">{settings.minWordLength} chars</span>
            </div>
            <input
              type="range"
              min={2}
              max={10}
              value={settings.minWordLength}
              onChange={(e) => onUpdateSettings({ minWordLength: Number(e.target.value) })}
              className="w-full accent-cyan-400"
            />
          </div>

          {/* Min Repetition Filter */}
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-slate-400 font-semibold">Min Repetitions</span>
              <span className="font-mono text-cyan-300">{settings.minRepetitions}×</span>
            </div>
            <input
              type="range"
              min={1}
              max={10}
              value={settings.minRepetitions}
              onChange={(e) => onUpdateSettings({ minRepetitions: Number(e.target.value) })}
              className="w-full accent-cyan-400"
            />
          </div>

          {/* Max Words to Display */}
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-slate-400 font-semibold">Max Display Words</span>
              <span className="font-mono text-cyan-300">{settings.maxWords}</span>
            </div>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={settings.maxWords}
              onChange={(e) => onUpdateSettings({ maxWords: Number(e.target.value) })}
              className="w-full accent-cyan-400"
            />
          </div>

          {/* Fillers & Rotation Toggles */}
          <div className="space-y-2 pt-2 border-t border-white/10 text-xs">
            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-slate-300">Filter speech fillers ("like", "um")</span>
              <input
                type="checkbox"
                checked={!settings.includeFillers}
                onChange={(e) => onUpdateSettings({ includeFillers: !e.target.checked })}
                className="accent-cyan-400 rounded"
              />
            </label>
            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-slate-300">Allow vertical words (Mosaic)</span>
              <input
                type="checkbox"
                checked={settings.enableRotation}
                onChange={(e) => onUpdateSettings({ enableRotation: e.target.checked })}
                className="accent-cyan-400 rounded"
              />
            </label>
          </div>

          {/* Mic Boost / Sensitivity Slider */}
          <div className="space-y-1.5 pt-2 border-t border-white/10 text-xs">
            <div className="flex justify-between text-[11px]">
              <span className="text-slate-300 font-semibold">Mic Gain Boost (Soft/Distant Voice)</span>
              <span className="font-mono text-cyan-400 font-bold">{(settings.micSensitivity || 2.0).toFixed(1)}×</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={5.0}
              step={0.5}
              value={settings.micSensitivity || 2.0}
              onChange={(e) => onUpdateSettings({ micSensitivity: Number(e.target.value) })}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <p className="text-[10px] text-slate-400 leading-tight">
              Amplifies distant or soft-spoken lecture audio with dynamic compression to fix timid capture.
            </p>
          </div>
        </div>
      )}
    </header>
  );
};
