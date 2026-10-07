import React from 'react';
import type { LectureSessionStats, ColorTheme } from '../types';
import { THEMES } from '../utils/themeStyles';
import { Clock, MessageSquare, Sparkles, Activity, KeyRound } from 'lucide-react';

interface Props {
  stats: LectureSessionStats;
  theme: ColorTheme;
}

export const MetricsHUD: React.FC<Props> = ({ stats, theme }) => {
  const themeColors = THEMES[theme];

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div
      className="flex flex-wrap items-center gap-3 px-4 py-2 rounded-2xl border backdrop-blur-md text-xs shadow-lg transition-all"
      style={{
        backgroundColor: themeColors.panelBg,
        borderColor: themeColors.borderColor,
        color: themeColors.textPrimary,
      }}
    >
      {/* Session Timer */}
      <div className="flex items-center gap-2 pr-3 border-r border-white/10">
        <Clock className="w-3.5 h-3.5 text-cyan-400" />
        <div>
          <div className="text-[10px] text-slate-400 font-medium">Session</div>
          <div className="font-mono font-bold">{formatTime(stats.durationSeconds)}</div>
        </div>
      </div>

      {/* Speaking Pace (WPM) */}
      <div className="flex items-center gap-2 pr-3 border-r border-white/10">
        <Activity className="w-3.5 h-3.5 text-emerald-400" />
        <div>
          <div className="text-[10px] text-slate-400 font-medium">Pace</div>
          <div className="font-mono font-bold">
            {stats.wordsPerMinute} <span className="text-[9px] text-slate-400 font-normal">wpm</span>
          </div>
        </div>
      </div>

      {/* Total Words */}
      <div className="flex items-center gap-2 pr-3 border-r border-white/10">
        <MessageSquare className="w-3.5 h-3.5 text-indigo-400" />
        <div>
          <div className="text-[10px] text-slate-400 font-medium">Total Words</div>
          <div className="font-mono font-bold">{stats.totalWords.toLocaleString()}</div>
        </div>
      </div>

      {/* Unique Vocabulary */}
      <div className="flex items-center gap-2 pr-3 border-r border-white/10">
        <Sparkles className="w-3.5 h-3.5 text-amber-400" />
        <div>
          <div className="text-[10px] text-slate-400 font-medium">Vocabulary</div>
          <div className="font-mono font-bold">{stats.uniqueWords.toLocaleString()}</div>
        </div>
      </div>

      {/* Top Concept */}
      {stats.topKeyword && (
        <div className="flex items-center gap-2">
          <KeyRound className="w-3.5 h-3.5 text-rose-400" />
          <div>
            <div className="text-[10px] text-slate-400 font-medium">Core Theme</div>
            <div className="font-bold text-cyan-300 truncate max-w-[120px]">
              {stats.topKeyword}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
