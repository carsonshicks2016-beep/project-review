import React, { useState, useRef, useEffect } from 'react';
import type { TranscriptItem, ColorTheme } from '../types';
import { THEMES } from '../utils/themeStyles';
import { FileText, Copy, Check, Download, Search, ChevronDown, ChevronUp } from 'lucide-react';

interface Props {
  transcripts: TranscriptItem[];
  interimText: string;
  theme: ColorTheme;
  isOpen: boolean;
  onToggleOpen: () => void;
  topKeywords: Set<string>;
}

export const TranscriptDrawer: React.FC<Props> = ({
  transcripts,
  interimText,
  theme,
  isOpen,
  onToggleOpen,
  topKeywords,
}) => {
  const [copied, setCopied] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  const themeColors = THEMES[theme];

  // Auto-scroll to bottom as new speech arrives
  useEffect(() => {
    if (scrollRef.current && isOpen) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcripts, interimText, isOpen]);

  const handleCopy = () => {
    const fullText = transcripts.map((t) => t.text).join(' ');
    navigator.clipboard.writeText(fullText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadTxt = () => {
    const lines = transcripts.map((t) => {
      const date = new Date(t.timestamp).toLocaleTimeString();
      return `[${date}] ${t.text}`;
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lecture-transcript-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatTime = (ms: number) => {
    return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  // Helper to highlight top keywords in transcript
  const renderHighlightedText = (text: string) => {
    if (topKeywords.size === 0 && !searchTerm) return text;

    const parts = text.split(/(\s+)/);
    return parts.map((part, i) => {
      const clean = part.toLowerCase().replace(/[^a-z0-9]/g, '');
      const isSearchMatch = searchTerm && clean.includes(searchTerm.toLowerCase());
      const isTopKeyword = topKeywords.has(clean);

      if (isSearchMatch) {
        return (
          <mark key={i} className="bg-amber-400 text-black px-1 rounded font-semibold">
            {part}
          </mark>
        );
      }
      if (isTopKeyword) {
        return (
          <span
            key={i}
            className="text-cyan-300 font-semibold underline decoration-cyan-500/50 decoration-2"
          >
            {part}
          </span>
        );
      }
      return part;
    });
  };

  if (!isOpen) {
    return (
      <button
        onClick={onToggleOpen}
        className="fixed bottom-4 right-4 z-30 flex items-center gap-2 px-3.5 py-2 rounded-xl border backdrop-blur-md shadow-xl transition-all hover:scale-105"
        style={{
          backgroundColor: themeColors.panelBg,
          borderColor: themeColors.borderColor,
          color: themeColors.textPrimary,
        }}
      >
        <FileText className="w-4 h-4 text-cyan-400" />
        <span className="text-xs font-semibold">Live Transcript</span>
        <span className="px-1.5 py-0.5 rounded-full bg-white/10 text-[10px] font-mono">
          {transcripts.length}
        </span>
        <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
      </button>
    );
  }

  return (
    <div
      className="fixed bottom-4 right-4 z-30 w-[460px] max-w-[calc(100vw-2rem)] h-80 rounded-2xl border backdrop-blur-2xl shadow-2xl flex flex-col overflow-hidden transition-all animate-in fade-in slide-in-from-bottom-6 duration-200"
      style={{
        backgroundColor: themeColors.panelBg,
        borderColor: themeColors.borderColor,
        color: themeColors.textPrimary,
      }}
    >
      {/* Header */}
      <div className="p-3 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-cyan-400" />
          <h2 className="font-bold text-xs tracking-wide">Classroom Transcript Feed</h2>
          <span className="px-1.5 py-0.5 rounded-full bg-white/10 text-[10px] font-mono">
            {transcripts.length} lines
          </span>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleCopy}
            className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
            title="Copy all transcript text"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
          <button
            onClick={handleDownloadTxt}
            className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
            title="Download .txt transcript"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onToggleOpen}
            className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors ml-1"
          >
            <ChevronDown className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Search filter */}
      <div className="px-3 py-1.5 border-b border-white/5 flex items-center gap-2 bg-black/20">
        <Search className="w-3 h-3 text-slate-500" />
        <input
          type="text"
          placeholder="Filter transcript lines..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full bg-transparent text-xs text-white placeholder:text-slate-500 focus:outline-none"
        />
        {searchTerm && (
          <button onClick={() => setSearchTerm('')} className="text-slate-500 hover:text-white text-[10px]">
            Clear
          </button>
        )}
      </div>

      {/* Transcript Scroll Feed */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3.5 space-y-2.5 font-sans text-xs">
        {transcripts.length === 0 && !interimText && (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 text-xs">
            <p>Waiting for speech...</p>
            <p className="text-[11px] text-slate-600 mt-1">Spoken words will stream here in real time.</p>
          </div>
        )}

        {transcripts.map((t) => (
          <div key={t.id} className="leading-relaxed group">
            <span className="text-[10px] text-slate-500 font-mono mr-2 select-none">
              {formatTime(t.timestamp)}
            </span>
            <span className="text-slate-200">{renderHighlightedText(t.text)}</span>
          </div>
        ))}

        {/* Live interim preview */}
        {interimText && (
          <div className="leading-relaxed italic text-cyan-400/80 animate-pulse">
            <span className="text-[10px] text-cyan-500/60 font-mono mr-2 select-none">speaking:</span>
            <span>{interimText}</span>
          </div>
        )}
      </div>
    </div>
  );
};
