import React, { useState } from 'react';
import type { WordFrequency, ColorTheme } from '../types';
import { THEMES } from '../utils/themeStyles';
import { Ban, Search, Plus, Trash2, BarChart2, ShieldAlert } from 'lucide-react';

interface Props {
  words: WordFrequency[];
  bannedWords: Set<string>;
  onExcludeWord: (word: string) => void;
  onUnbanWord: (word: string) => void;
  onAddCustomBan: (word: string) => void;
  theme: ColorTheme;
  isOpen: boolean;
  onToggleOpen: () => void;
}

export const WordListSidebar: React.FC<Props> = ({
  words,
  bannedWords,
  onExcludeWord,
  onUnbanWord,
  onAddCustomBan,
  theme,
  isOpen,
  onToggleOpen,
}) => {
  const [activeTab, setActiveTab] = useState<'frequent' | 'banned'>('frequent');
  const [searchQuery, setSearchQuery] = useState('');
  const [newBanInput, setNewBanInput] = useState('');

  const themeColors = THEMES[theme];

  const filteredWords = words.filter((w) =>
    w.text.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const bannedArray = Array.from(bannedWords).sort();
  const filteredBanned = bannedArray.filter((w) =>
    w.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const maxCount = words.length > 0 ? words[0].count : 1;

  const handleAddBan = (e: React.FormEvent) => {
    e.preventDefault();
    if (newBanInput.trim()) {
      onAddCustomBan(newBanInput.trim().toLowerCase());
      setNewBanInput('');
    }
  };

  if (!isOpen) {
    return (
      <button
        onClick={onToggleOpen}
        className="absolute left-4 top-20 z-30 p-2.5 rounded-xl border backdrop-blur-md shadow-xl transition-all hover:scale-105"
        style={{
          backgroundColor: themeColors.panelBg,
          borderColor: themeColors.borderColor,
          color: themeColors.textPrimary,
        }}
        title="Open Word Frequency Inspector"
      >
        <BarChart2 className="w-4 h-4" />
      </button>
    );
  }

  return (
    <div
      className="absolute left-4 top-20 bottom-24 z-30 w-80 rounded-2xl border backdrop-blur-xl shadow-2xl flex flex-col overflow-hidden transition-all"
      style={{
        backgroundColor: themeColors.panelBg,
        borderColor: themeColors.borderColor,
        color: themeColors.textPrimary,
      }}
    >
      {/* Header */}
      <div className="p-3.5 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart2 className="w-4 h-4 text-cyan-400" />
          <h2 className="font-bold text-sm tracking-wide">Lexical Frequency</h2>
        </div>
        <button
          onClick={onToggleOpen}
          className="text-slate-400 hover:text-white text-xs px-2 py-1 rounded-lg hover:bg-white/5"
        >
          ✕
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/10 text-xs font-semibold">
        <button
          onClick={() => setActiveTab('frequent')}
          className={`flex-1 py-2 text-center transition-colors border-b-2 ${
            activeTab === 'frequent'
              ? 'border-cyan-400 text-cyan-400 bg-white/5'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Top Repeated ({words.length})
        </button>
        <button
          onClick={() => setActiveTab('banned')}
          className={`flex-1 py-2 text-center transition-colors border-b-2 flex items-center justify-center gap-1.5 ${
            activeTab === 'banned'
              ? 'border-rose-400 text-rose-400 bg-white/5'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <ShieldAlert className="w-3 h-3" />
          Ignored ({bannedArray.length})
        </button>
      </div>

      {/* Search Bar */}
      <div className="p-2.5 border-b border-white/10">
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400" />
          <input
            type="text"
            placeholder="Search keywords..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-black/30 border border-white/10 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-cyan-400/50"
          />
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-2.5 space-y-1.5">
        {activeTab === 'frequent' ? (
          filteredWords.length > 0 ? (
            filteredWords.map((word, idx) => {
              const widthPct = (word.count / maxCount) * 100;
              return (
                <div
                  key={word.text}
                  className="group relative flex items-center justify-between p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  {/* Subtle bar background */}
                  <div
                    className="absolute left-0 top-0 bottom-0 rounded-lg bg-cyan-500/10 pointer-events-none transition-all"
                    style={{ width: `${widthPct}%` }}
                  />

                  <div className="relative z-10 flex items-center gap-2 truncate pr-2">
                    <span className="text-[10px] text-slate-500 font-mono w-4">
                      #{idx + 1}
                    </span>
                    <span className="text-xs font-medium truncate">{word.text}</span>
                  </div>

                  <div className="relative z-10 flex items-center gap-1.5 flex-shrink-0">
                    <span className="text-[11px] font-mono font-bold px-1.5 py-0.5 rounded bg-black/40 text-cyan-300">
                      ×{word.count}
                    </span>
                    <button
                      onClick={() => onExcludeWord(word.text)}
                      title="Exclude word from cloud"
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 transition-opacity"
                    >
                      <Ban className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="text-center py-8 text-xs text-slate-500">
              No words matching criteria.
            </div>
          )
        ) : (
          <div className="space-y-3">
            {/* Add custom ban */}
            <form onSubmit={handleAddBan} className="flex gap-1.5">
              <input
                type="text"
                placeholder="Add word to ignore..."
                value={newBanInput}
                onChange={(e) => setNewBanInput(e.target.value)}
                className="flex-1 px-2.5 py-1.5 rounded-lg bg-black/30 border border-white/10 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:border-rose-400/50"
              />
              <button
                type="submit"
                className="px-2.5 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 text-xs font-semibold flex items-center gap-1"
              >
                <Plus className="w-3.5 h-3.5" /> Add
              </button>
            </form>

            <div className="space-y-1">
              {filteredBanned.length > 0 ? (
                filteredBanned.map((bw) => (
                  <div
                    key={bw}
                    className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-white/5 text-xs text-slate-300"
                  >
                    <span className="capitalize">{bw}</span>
                    <button
                      onClick={() => onUnbanWord(bw)}
                      className="p-1 text-slate-400 hover:text-emerald-400 hover:bg-white/5 rounded"
                      title="Restore word"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))
              ) : (
                <div className="text-center py-6 text-xs text-slate-500">
                  No custom ignored words yet.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
