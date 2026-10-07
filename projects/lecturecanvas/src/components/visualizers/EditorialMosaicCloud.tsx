import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import cloud from 'd3-cloud';
import type { WordFrequency, VisualizerSettings } from '../../types';
import { THEMES, FONT_FAMILIES } from '../../utils/themeStyles';

interface CloudWord extends cloud.Word {
  text: string;
  size: number;
  count: number;
  color: string;
  percentage: number;
  rotate?: number;
}

/**
 * Stable per-word rotation. Using Math.random() here re-rolls every word's
 * orientation on each re-layout, so a live lecture reshuffles the whole poster
 * every couple of seconds; hashing the text keeps each word where the reader
 * last saw it.
 */
function rotationFor(text: string): number {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = (hash * 31 + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % 4 === 0 ? 90 : 0;
}

/** Re-packing is expensive, so bursts of incoming speech collapse into one layout. */
const RELAYOUT_DEBOUNCE_MS = 450;

interface Props {
  words: WordFrequency[];
  settings: VisualizerSettings;
  onExcludeWord: (word: string) => void;
  exportRef: React.MutableRefObject<HTMLCanvasElement | null>;
}

export const EditorialMosaicCloud: React.FC<Props> = ({
  words,
  settings,
  onExcludeWord,
  exportRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredWord, setHoveredWord] = useState<CloudWord | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const layoutWordsRef = useRef<CloudWord[]>([]);
  const hoveredWordRef = useRef<CloudWord | null>(null);
  // Whether anything has been painted yet for this mount.
  const hasPaintedRef = useRef(false);
  // Bumped whenever the container resizes, to re-run the packing layout.
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  const theme = THEMES[settings.theme];
  const font = FONT_FAMILIES[settings.fontFamily];

  // Read by drawCloud, which must stay referentially stable across hover changes.
  useEffect(() => {
    hoveredWordRef.current = hoveredWord;
  }, [hoveredWord]);

  // d3-cloud packs into a fixed box, so a resized window needs a fresh layout.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (!box) return;
      setContainerSize((prev) =>
        Math.abs(prev.width - box.width) < 1 && Math.abs(prev.height - box.height) < 1
          ? prev
          : { width: box.width, height: box.height }
      );
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const drawCloud = useCallback(
    (cloudWords: CloudWord[], width: number, height: number, dpr: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      // Fill background
      ctx.fillStyle = theme.background;
      ctx.fillRect(0, 0, width, height);

      // Decorative geometric corner framing for editorial poster aesthetic
      ctx.strokeStyle = theme.borderColor;
      ctx.lineWidth = 1;
      ctx.strokeRect(20, 20, width - 40, height - 40);

      // Translate to center for d3-cloud coordinates
      ctx.translate(width / 2, height / 2);

      const hovered = hoveredWordRef.current;

      cloudWords.forEach((word) => {
        if (word.x === undefined || word.y === undefined) return;

        ctx.save();
        ctx.translate(word.x, word.y);
        if (word.rotate) {
          ctx.rotate((word.rotate * Math.PI) / 180);
        }

        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `700 ${word.size}px ${font.css}`;

        if (hovered?.text === word.text) {
          ctx.shadowBlur = 20;
          ctx.shadowColor = word.color;
          ctx.fillStyle = '#ffffff';
        } else {
          ctx.shadowBlur = 8;
          ctx.shadowColor = `${word.color}66`;
          ctx.fillStyle = word.color;
        }

        ctx.fillText(word.text, 0, 0);
        ctx.restore();
      });

      ctx.restore();
    },
    [theme, font]
  );

  useEffect(() => {
    if (!containerRef.current || !canvasRef.current || words.length === 0) {
      layoutWordsRef.current = [];
      return;
    }

    let cancelled = false;
    const container = containerRef.current;
    const canvas = canvasRef.current;

    const rect = container.getBoundingClientRect();
    const width = rect.width || 800;
    const height = rect.height || 600;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const maxCount = d3.max(words, (d) => d.count) || 1;
    const minCount = d3.min(words, (d) => d.count) || 1;

    const minFont = Math.max(14, Math.min(width, height) * 0.024);
    const maxFont = Math.max(52, Math.min(width, height) * 0.11);

    const fontScale = d3
      .scaleSqrt()
      .domain([minCount, maxCount])
      .range([minFont, maxFont]);

    const colorScale = d3.scaleOrdinal(theme.palette);

    const cloudData: CloudWord[] = words.map((w) => ({
      text: w.text,
      size: fontScale(w.count),
      count: w.count,
      color: colorScale(w.text),
      percentage: w.percentage,
    }));

    const layout = cloud<CloudWord>()
      // Pack into a slightly inset box. d3-cloud measures from a word's center and
      // will happily place one so that it overhangs the bounds; the inset turns
      // that overhang into margin instead of a word sliced off at the edge.
      .size([width * 0.92, height * 0.9])
      .words(cloudData)
      .padding(8)
      .rotate((d) => (settings.enableRotation ? rotationFor(d.text) : 0))
      .font(font.css)
      .fontSize((d) => d.size)
      .on('end', (outputWords) => {
        if (cancelled) return;
        layoutWordsRef.current = outputWords;
        hasPaintedRef.current = true;
        drawCloud(outputWords, width, height, dpr);
      });

    // Debouncing only makes sense once something is on screen; making the very
    // first layout wait leaves the canvas blank, and a cancelled first run (a
    // re-mount, a resize) would leave it blank indefinitely.
    const delay = hasPaintedRef.current ? RELAYOUT_DEBOUNCE_MS : 0;
    const timer = window.setTimeout(() => {
      if (!cancelled) layout.start();
    }, delay);

    // Packing is asynchronous. Without this guard a superseded run can finish
    // after a newer one and repaint the canvas with stale word positions.
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      layout.stop();
    };
  }, [words, settings.enableRotation, containerSize, theme, font, drawCloud]);

  // Re-draw when hover changes
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    drawCloud(layoutWordsRef.current, rect.width, rect.height, dpr);
  }, [hoveredWord, drawCloud]);

  // Connect export ref
  useEffect(() => {
    if (exportRef && canvasRef.current) {
      exportRef.current = canvasRef.current;
    }
  }, [exportRef]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !containerRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    // Convert to centered coordinates
    const mx = e.clientX - rect.left - width / 2;
    const my = e.clientY - rect.top - height / 2;
    setMousePos({ x: e.clientX, y: e.clientY });

    // Hit test words
    let found: CloudWord | null = null;
    for (const w of layoutWordsRef.current) {
      if (w.x === undefined || w.y === undefined) continue;
      // Approximate bounding box
      const halfW = (w.text.length * w.size * 0.32);
      const halfH = w.size * 0.5;

      if (
        mx >= w.x - halfW &&
        mx <= w.x + halfW &&
        my >= w.y - halfH &&
        my <= w.y + halfH
      ) {
        found = w;
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
          <div className="text-[10px] text-rose-400/90 mt-1.5">
            Click word to ban / exclude
          </div>
        </div>
      )}

      {words.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 pointer-events-none">
          <p className="text-sm font-medium text-slate-300">Awaiting speech for typographic mosaic...</p>
        </div>
      )}
    </div>
  );
};
