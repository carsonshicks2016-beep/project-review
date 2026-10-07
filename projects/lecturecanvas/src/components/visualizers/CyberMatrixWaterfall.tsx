import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { WordFrequency, VisualizerSettings } from '../../types';
import { THEMES, FONT_FAMILIES } from '../../utils/themeStyles';

interface MatrixWord {
  text: string;
  count: number;
  percentage: number;
  color: string;
  fontSize: number;
  x: number;
  y: number;
  width: number;
  height: number;
  lastSeen: number;
  lane: number;
}

interface Drop {
  x: number;
  y: number;
  speed: number;
  chars: string[];
}

interface Props {
  words: WordFrequency[];
  settings: VisualizerSettings;
  onExcludeWord: (word: string) => void;
  exportRef: React.MutableRefObject<HTMLCanvasElement | null>;
}

export const CyberMatrixWaterfall: React.FC<Props> = ({
  words,
  settings,
  onExcludeWord,
  exportRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredWord, setHoveredWord] = useState<MatrixWord | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const matrixWordsRef = useRef<MatrixWord[]>([]);
  const dropsRef = useRef<Drop[]>([]);

  const theme = THEMES[settings.theme];
  const font = FONT_FAMILIES[settings.fontFamily];

  // Initialize and distribute words into lanes
  useEffect(() => {
    if (!containerRef.current || words.length === 0) {
      matrixWordsRef.current = [];
      return;
    }

    const rect = containerRef.current.getBoundingClientRect();
    const width = rect.width || 800;
    const height = rect.height || 600;

    const maxCount = d3.max(words, (d) => d.count) || 1;
    const minCount = d3.min(words, (d) => d.count) || 1;

    const fontScale = d3
      .scalePow()
      .exponent(0.6)
      .domain([minCount, maxCount])
      .range([13, 34]);

    const colorScale = d3.scaleOrdinal(theme.palette);

    // Organize words into 3 to 5 visual vertical columns/lanes
    const numLanes = width > 900 ? 4 : 3;
    const laneWidth = (width - 80) / numLanes;
    const laneHeights = new Array(numLanes).fill(50);

    const newMatrixWords: MatrixWord[] = [];

    words.forEach((w) => {
      const fontSize = fontScale(w.count);
      const estWidth = w.text.length * fontSize * 0.65 + 36;
      const estHeight = fontSize + 24;

      // Find lane with smallest current height
      let minLane = 0;
      for (let l = 1; l < numLanes; l++) {
        if (laneHeights[l] < laneHeights[minLane]) {
          minLane = l;
        }
      }

      const x = 40 + minLane * laneWidth + laneWidth / 2;
      const y = laneHeights[minLane] + estHeight / 2;
      laneHeights[minLane] += estHeight + 16;

      newMatrixWords.push({
        text: w.text,
        count: w.count,
        percentage: w.percentage,
        color: colorScale(w.text),
        fontSize,
        x,
        y: Math.min(height - 40, y),
        width: estWidth,
        height: estHeight,
        lastSeen: w.lastSeen,
        lane: minLane,
      });
    });

    matrixWordsRef.current = newMatrixWords;

    // Initialize ambient code drops
    const drops: Drop[] = [];
    const glyphs = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン⚡∑∏∆Ω≈≠';
    const dropCount = Math.floor(width / 32);

    for (let i = 0; i < dropCount; i++) {
      drops.push({
        x: i * 32 + 16,
        y: Math.random() * height,
        speed: 1.5 + Math.random() * 2.5,
        chars: Array.from({ length: 12 }, () => glyphs[Math.floor(Math.random() * glyphs.length)]),
      });
    }
    dropsRef.current = drops;
  }, [words, theme]);

  // Main Cyberpunk Waterfall Animation Loop
  useEffect(() => {
    let animationId: number;

    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas || !containerRef.current) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const rect = containerRef.current.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = rect.width;
      const height = rect.height;

      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      // Deep dark terminal background with slight green/blue tint
      ctx.fillStyle = theme.background;
      ctx.fillRect(0, 0, width, height);

      // Draw faint vertical scanlines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.015)';
      ctx.lineWidth = 1;
      for (let y = 0; y < height; y += 4) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // 1. Draw animated matrix code drops
      const drops = dropsRef.current;
      ctx.font = "10px 'JetBrains Mono', monospace";
      for (let i = 0; i < drops.length; i++) {
        const drop = drops[i];
        drop.y += drop.speed;
        if (drop.y > height + 150) {
          drop.y = -150;
        }

        for (let j = 0; j < drop.chars.length; j++) {
          const cy = drop.y - j * 14;
          if (cy < 0 || cy > height) continue;

          // Leading character glows brightest
          if (j === 0) {
            ctx.fillStyle = '#ffffff';
            ctx.shadowBlur = 6;
            ctx.shadowColor = theme.palette[0];
          } else {
            ctx.shadowBlur = 0;
            const alpha = Math.max(0.04, (1 - j / drop.chars.length) * 0.25);
            ctx.fillStyle = `${theme.palette[0]}${Math.floor(alpha * 255).toString(16).padStart(2, '0')}`;
          }
          ctx.fillText(drop.chars[j], drop.x, cy);
        }
      }

      ctx.shadowBlur = 0;
      const now = Date.now();

      // 2. Draw keyword pedestals
      const matrixWords = matrixWordsRef.current;
      for (let i = 0; i < matrixWords.length; i++) {
        const mw = matrixWords[i];
        const isHovered = hoveredWord?.text === mw.text;
        const isRecent = now - mw.lastSeen < 6000;

        const hw = mw.width / 2;
        const hh = mw.height / 2;

        // Pedestal bounding box
        ctx.beginPath();
        ctx.roundRect(mw.x - hw, mw.y - hh, mw.width, mw.height, 8);

        if (isHovered) {
          ctx.fillStyle = `${mw.color}25`;
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.shadowBlur = 18;
          ctx.shadowColor = mw.color;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else if (isRecent) {
          const pulse = (Math.sin(now * 0.009) + 1) * 0.5;
          ctx.fillStyle = `${mw.color}15`;
          ctx.fill();
          ctx.strokeStyle = mw.color;
          ctx.lineWidth = 1.5 + pulse;
          ctx.shadowBlur = 12;
          ctx.shadowColor = mw.color;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
          ctx.fill();
          ctx.strokeStyle = `${mw.color}44`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Cybernetic bracket accents [ ]
        ctx.strokeStyle = isHovered ? '#ffffff' : mw.color;
        ctx.lineWidth = 1.5;
        const bracketLen = 6;
        // Top-left
        ctx.beginPath();
        ctx.moveTo(mw.x - hw + bracketLen, mw.y - hh);
        ctx.lineTo(mw.x - hw, mw.y - hh);
        ctx.lineTo(mw.x - hw, mw.y - hh + bracketLen);
        ctx.stroke();
        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(mw.x + hw - bracketLen, mw.y + hh);
        ctx.lineTo(mw.x + hw, mw.y + hh);
        ctx.lineTo(mw.x + hw, mw.y + hh - bracketLen);
        ctx.stroke();

        // Typography
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `700 ${mw.fontSize}px ${font.css}`;

        if (isHovered) {
          ctx.shadowBlur = 16;
          ctx.shadowColor = mw.color;
          ctx.fillStyle = '#ffffff';
        } else if (isRecent) {
          ctx.shadowBlur = 10;
          ctx.shadowColor = mw.color;
          ctx.fillStyle = '#ffffff';
        } else {
          ctx.shadowBlur = 4;
          ctx.shadowColor = mw.color;
          ctx.fillStyle = theme.textPrimary;
        }

        ctx.fillText(mw.text, mw.x, mw.y - 2);

        // Repetition count pill
        ctx.shadowBlur = 0;
        ctx.font = `600 ${Math.max(9, mw.fontSize * 0.35)}px 'Inter', sans-serif`;
        ctx.fillStyle = isHovered ? '#ffffff' : mw.color;
        ctx.fillText(`×${mw.count}`, mw.x, mw.y + mw.fontSize * 0.62);
      }

      ctx.restore();
      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [theme, font, hoveredWord]);

  // Connect export ref
  useEffect(() => {
    if (exportRef && canvasRef.current) {
      exportRef.current = canvasRef.current;
    }
  }, [exportRef]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setMousePos({ x: e.clientX, y: e.clientY });

    let found: MatrixWord | null = null;
    for (const mw of matrixWordsRef.current) {
      const hw = mw.width / 2;
      const hh = mw.height / 2;
      if (mx >= mw.x - hw && mx <= mw.x + hw && my >= mw.y - hh && my <= mw.y + hh) {
        found = mw;
        break;
      }
    }
    setHoveredWord(found);
  };

  const handleMouseLeave = () => {
    setHoveredWord(null);
    setMousePos(null);
  };

  const handleClick = () => {
    if (hoveredWord) {
      onExcludeWord(hoveredWord.text);
      setHoveredWord(null);
    }
  };

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden select-none">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-pointer"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />

      {/* Floating Hover Tooltip */}
      {hoveredWord && mousePos && (
        <div
          className="fixed pointer-events-none z-50 px-3.5 py-2 rounded-xl text-xs backdrop-blur-md shadow-2xl transition-transform duration-75 border"
          style={{
            left: `${mousePos.x + 14}px`,
            top: `${mousePos.y + 14}px`,
            backgroundColor: theme.panelBg,
            borderColor: theme.borderColor,
            color: theme.textPrimary,
          }}
        >
          <div className="font-bold text-sm flex items-center gap-2">
            <span>{hoveredWord.text}</span>
            <span
              className="px-1.5 py-0.5 rounded text-[10px] font-mono"
              style={{ backgroundColor: `${hoveredWord.color}33`, color: hoveredWord.color }}
            >
              ×{hoveredWord.count}
            </span>
          </div>
          <div className="text-slate-400 mt-1 flex items-center justify-between gap-4">
            <span>Class frequency:</span>
            <span className="font-mono text-slate-200">{hoveredWord.percentage.toFixed(1)}%</span>
          </div>
          <div className="text-[10px] text-rose-400/90 mt-1.5">Click word to ban / exclude</div>
        </div>
      )}

      {words.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 pointer-events-none">
          <p className="text-sm font-medium text-slate-300">Awaiting speech for Cyber Matrix Waterfall...</p>
        </div>
      )}
    </div>
  );
};
