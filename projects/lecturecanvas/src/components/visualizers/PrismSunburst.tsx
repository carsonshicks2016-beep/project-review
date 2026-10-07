import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { WordFrequency, VisualizerSettings } from '../../types';
import { THEMES, FONT_FAMILIES } from '../../utils/themeStyles';

interface SunburstWedge {
  text: string;
  count: number;
  percentage: number;
  color: string;
  startAngle: number;
  endAngle: number;
  innerRadius: number;
  outerRadius: number;
  fontSize: number;
  tier: number;
  lastSeen: number;
}

interface Props {
  words: WordFrequency[];
  settings: VisualizerSettings;
  onExcludeWord: (word: string) => void;
  exportRef: React.MutableRefObject<HTMLCanvasElement | null>;
}

export const PrismSunburst: React.FC<Props> = ({
  words,
  settings,
  onExcludeWord,
  exportRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredWedge, setHoveredWedge] = useState<SunburstWedge | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const wedgesRef = useRef<SunburstWedge[]>([]);
  const rotationOffsetRef = useRef(0);

  const theme = THEMES[settings.theme];
  const font = FONT_FAMILIES[settings.fontFamily];

  // Calculate concentric tiers and angle sweeps
  useEffect(() => {
    if (!containerRef.current || words.length === 0) {
      wedgesRef.current = [];
      return;
    }

    const rect = containerRef.current.getBoundingClientRect();
    const width = rect.width || 800;
    const height = rect.height || 600;
    const maxR = Math.min(width, height) * 0.44;

    const colorScale = d3.scaleOrdinal(theme.palette);

    // Split words into 3 concentric tiers:
    // Tier 1 (Core): top 6 words
    // Tier 2 (Frequent): next 12 words
    // Tier 3 (Emerging): remaining words
    const tier1Words = words.slice(0, 6);
    const tier2Words = words.slice(6, 18);
    const tier3Words = words.slice(18, 36);

    const tiers = [
      { list: tier1Words, r0: maxR * 0.2, r1: maxR * 0.48, baseFont: 16 },
      { list: tier2Words, r0: maxR * 0.52, r1: maxR * 0.74, baseFont: 13 },
      { list: tier3Words, r0: maxR * 0.78, r1: maxR * 0.98, baseFont: 11 },
    ];

    const newWedges: SunburstWedge[] = [];

    tiers.forEach((tier, tierIdx) => {
      if (tier.list.length === 0) return;
      const totalCount = d3.sum(tier.list, (d) => d.count) || 1;
      let curAngle = 0;

      tier.list.forEach((w) => {
        const sweep = (w.count / totalCount) * (Math.PI * 2);
        const startAngle = curAngle;
        const endAngle = curAngle + sweep;
        curAngle = endAngle;

        newWedges.push({
          text: w.text,
          count: w.count,
          percentage: w.percentage,
          color: colorScale(w.text),
          startAngle,
          endAngle,
          innerRadius: tier.r0,
          outerRadius: tier.r1,
          fontSize: tier.baseFont,
          tier: tierIdx + 1,
          lastSeen: w.lastSeen,
        });
      });
    });

    wedgesRef.current = newWedges;
  }, [words, theme]);

  // Main Render Loop
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

      // Deep space ambient background
      ctx.fillStyle = theme.background;
      ctx.fillRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height / 2;

      // Slow radiant drift
      rotationOffsetRef.current += 0.0008;
      const rot = rotationOffsetRef.current;

      // Center glowing reactor core
      const coreGrad = ctx.createRadialGradient(centerX, centerY, 4, centerX, centerY, 50);
      coreGrad.addColorStop(0, `${theme.palette[0]}55`);
      coreGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 50, 0, Math.PI * 2);
      ctx.fill();

      // Render wedges
      const wedges = wedgesRef.current;
      const now = Date.now();

      for (let i = 0; i < wedges.length; i++) {
        const w = wedges[i];
        const a0 = w.startAngle + rot;
        const a1 = w.endAngle + rot;
        const isHovered = hoveredWedge?.text === w.text;
        const isRecent = now - w.lastSeen < 6000;

        // Draw Arc Sector
        ctx.beginPath();
        ctx.arc(centerX, centerY, w.outerRadius, a0, a1, false);
        ctx.arc(centerX, centerY, w.innerRadius, a1, a0, true);
        ctx.closePath();

        if (isHovered) {
          ctx.fillStyle = `${w.color}44`;
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2.5;
          ctx.shadowBlur = 18;
          ctx.shadowColor = w.color;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else if (isRecent) {
          const pulse = (Math.sin(now * 0.008) + 1) * 0.5;
          ctx.fillStyle = `${w.color}25`;
          ctx.fill();
          ctx.strokeStyle = w.color;
          ctx.lineWidth = 1.5 + pulse;
          ctx.shadowBlur = 10;
          ctx.shadowColor = w.color;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else {
          ctx.fillStyle = `${w.color}10`;
          ctx.fill();
          ctx.strokeStyle = `${w.color}35`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Tangential & Radial Typography
        const midAngle = (a0 + a1) / 2;
        const midRadius = (w.innerRadius + w.outerRadius) / 2;
        const tx = centerX + Math.cos(midAngle) * midRadius;
        const ty = centerY + Math.sin(midAngle) * midRadius;

        ctx.save();
        ctx.translate(tx, ty);

        // Orient text radially so it's readable
        let textAngle = midAngle;
        if (textAngle > Math.PI / 2 && textAngle < (3 * Math.PI) / 2) {
          textAngle += Math.PI; // Flip so not upside down
        }
        ctx.rotate(textAngle);

        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `600 ${w.fontSize}px ${font.css}`;

        if (isHovered) {
          ctx.shadowBlur = 14;
          ctx.shadowColor = w.color;
          ctx.fillStyle = '#ffffff';
        } else if (isRecent) {
          ctx.shadowBlur = 8;
          ctx.shadowColor = w.color;
          ctx.fillStyle = '#ffffff';
        } else {
          ctx.shadowBlur = 2;
          ctx.shadowColor = w.color;
          ctx.fillStyle = theme.textPrimary;
        }

        ctx.fillText(w.text, 0, -w.fontSize * 0.4);

        // Repetition count
        ctx.shadowBlur = 0;
        ctx.font = `500 ${Math.max(8, w.fontSize * 0.42)}px 'Inter', sans-serif`;
        ctx.fillStyle = isHovered ? '#ffffff' : w.color;
        ctx.fillText(`×${w.count}`, 0, w.fontSize * 0.6);

        ctx.restore();
      }

      ctx.restore();
      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [theme, font, hoveredWedge]);

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
    const mx = e.clientX - rect.left - width / 2;
    const my = e.clientY - rect.top - height / 2;
    setMousePos({ x: e.clientX, y: e.clientY });

    // Polar distance and angle
    const dist = Math.sqrt(mx * mx + my * my);
    let angle = Math.atan2(my, mx) - (rotationOffsetRef.current % (Math.PI * 2));
    if (angle < 0) angle += Math.PI * 2;

    let found: SunburstWedge | null = null;
    for (const w of wedgesRef.current) {
      if (dist >= w.innerRadius && dist <= w.outerRadius) {
        if (angle >= w.startAngle && angle <= w.endAngle) {
          found = w;
          break;
        }
      }
    }
    setHoveredWedge(found);
  };

  const handleMouseLeave = () => {
    setHoveredWedge(null);
    setMousePos(null);
  };

  const handleClick = () => {
    if (hoveredWedge) {
      onExcludeWord(hoveredWedge.text);
      setHoveredWedge(null);
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
      {hoveredWedge && mousePos && (
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
            <span>{hoveredWedge.text}</span>
            <span
              className="px-1.5 py-0.5 rounded text-[10px] font-mono"
              style={{ backgroundColor: `${hoveredWedge.color}33`, color: hoveredWedge.color }}
            >
              Tier {hoveredWedge.tier} · ×{hoveredWedge.count}
            </span>
          </div>
          <div className="text-slate-400 mt-1 flex items-center justify-between gap-4">
            <span>Class frequency:</span>
            <span className="font-mono text-slate-200">{hoveredWedge.percentage.toFixed(1)}%</span>
          </div>
          <div className="text-[10px] text-rose-400/90 mt-1.5">Click wedge to ban / exclude</div>
        </div>
      )}

      {words.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 pointer-events-none">
          <p className="text-sm font-medium text-slate-300">Awaiting speech for Prism Sunburst...</p>
        </div>
      )}
    </div>
  );
};
