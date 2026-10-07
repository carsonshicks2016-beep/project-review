import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { WordFrequency, VisualizerSettings } from '../../types';
import { THEMES, FONT_FAMILIES } from '../../utils/themeStyles';

interface GalaxyNode {
  text: string;
  count: number;
  percentage: number;
  color: string;
  angle: number;
  baseRadius: number;
  fontSize: number;
  x: number;
  y: number;
  lastSeen: number;
}

/** Scratch context for measuring label widths; see hit testing below. */
let measureContext: CanvasRenderingContext2D | null = null;

function measureTextWidth(text: string, fontSize: number, fontCss: string): number {
  if (!measureContext) {
    measureContext = document.createElement('canvas').getContext('2d');
  }
  if (!measureContext) return text.length * fontSize * 0.55;
  measureContext.font = `600 ${fontSize}px ${fontCss}`;
  return measureContext.measureText(text).width;
}

interface Props {
  words: WordFrequency[];
  settings: VisualizerSettings;
  onExcludeWord: (word: string) => void;
  exportRef: React.MutableRefObject<HTMLCanvasElement | null>;
}

export const CelestialGalaxy: React.FC<Props> = ({
  words,
  settings,
  onExcludeWord,
  exportRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<GalaxyNode | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const nodesRef = useRef<GalaxyNode[]>([]);
  const rotationAngleRef = useRef(0);
  const hoveredNodeRef = useRef<GalaxyNode | null>(null);

  // Read by the animation loop, so hovering does not restart it every frame.
  useEffect(() => {
    hoveredNodeRef.current = hoveredNode;
  }, [hoveredNode]);

  const theme = THEMES[settings.theme];
  const font = FONT_FAMILIES[settings.fontFamily];

  // Map words into golden spiral positions
  useEffect(() => {
    if (!containerRef.current || words.length === 0) {
      nodesRef.current = [];
      return;
    }

    const rect = containerRef.current.getBoundingClientRect();
    const width = rect.width || 800;
    const height = rect.height || 600;
    const maxDimension = Math.min(width, height) * 0.46;

    const maxCount = d3.max(words, (d) => d.count) || 1;
    const minCount = d3.min(words, (d) => d.count) || 1;

    const minFont = Math.max(13, Math.min(width, height) * 0.02);
    const maxFont = Math.max(46, Math.min(width, height) * 0.08);

    const fontScale = d3
      .scalePow()
      .exponent(0.6)
      .domain([minCount, maxCount])
      .range([minFont, maxFont]);

    const colorScale = d3.scaleOrdinal(theme.palette);

    // Golden ratio spiral calculation:
    // Most repeated words sit closer to the core; lower frequency words spiral outward
    const goldenAngle = 137.5 * (Math.PI / 180);

    const newNodes: GalaxyNode[] = words.map((w, index) => {
      // Index 0 is the #1 most repeated word (at center)
      const rankNorm = index / Math.max(1, words.length);
      const angle = index * goldenAngle;
      // Exponential distribution of radius
      const baseRadius = (Math.sqrt(rankNorm) * maxDimension) + 20;

      const fontSize = fontScale(w.count);

      return {
        text: w.text,
        count: w.count,
        percentage: w.percentage,
        color: colorScale(w.text),
        angle,
        baseRadius,
        fontSize,
        x: 0,
        y: 0,
        lastSeen: w.lastSeen,
      };
    });

    nodesRef.current = newNodes;
  }, [words, theme, font]);

  // Animation & Rendering Loop
  useEffect(() => {
    let animationFrameId: number;

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

      // Deep space background
      ctx.fillStyle = theme.background;
      ctx.fillRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height / 2;

      // Increment celestial rotation slowly
      rotationAngleRef.current += 0.0015;
      const currentRot = rotationAngleRef.current;

      // Draw celestial orbital rings
      const rings = [0.2, 0.4, 0.65, 0.9];
      const maxDim = Math.min(width, height) * 0.46;
      ctx.lineWidth = 1;
      rings.forEach((rRatio) => {
        ctx.beginPath();
        ctx.arc(centerX, centerY, maxDim * rRatio, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.stroke();
      });

      // Draw faint spiral trail
      ctx.beginPath();
      for (let theta = 0; theta < Math.PI * 12; theta += 0.1) {
        const r = (theta / (Math.PI * 12)) * maxDim;
        const x = centerX + Math.cos(theta + currentRot) * r;
        const y = centerY + Math.sin(theta + currentRot) * r;
        if (theta === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `${theme.glowColor.replace(/[\d.]+\)$/, '0.08)')}`;
      ctx.stroke();

      const now = Date.now();
      const hovered = hoveredNodeRef.current;

      // Update positions and draw words
      const nodes = nodesRef.current;
      nodes.forEach((node) => {
        const totalAngle = node.angle + currentRot;
        node.x = centerX + Math.cos(totalAngle) * node.baseRadius;
        node.y = centerY + Math.sin(totalAngle) * node.baseRadius;

        const isHovered = hovered?.text === node.text;
        const isRecent = now - node.lastSeen < 6000;

        // Draw orbital connection spoke to center
        if (isHovered || isRecent) {
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.lineTo(node.x, node.y);
          ctx.strokeStyle = `${node.color}44`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Planetary aura
        const auraRadius = Math.max(node.fontSize * 0.5, 8);
        ctx.beginPath();
        ctx.arc(node.x, node.y, auraRadius, 0, Math.PI * 2);
        ctx.fillStyle = isHovered ? `${node.color}55` : `${node.color}15`;
        ctx.fill();

        // Node center star
        ctx.beginPath();
        ctx.arc(node.x, node.y, isHovered ? 4 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();

        // Word text label
        ctx.font = `600 ${node.fontSize}px ${font.css}`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        if (isHovered) {
          ctx.shadowBlur = 18;
          ctx.shadowColor = node.color;
          ctx.fillStyle = '#ffffff';
        } else {
          ctx.shadowBlur = isRecent ? 12 : 4;
          ctx.shadowColor = node.color;
          ctx.fillStyle = isRecent ? '#ffffff' : theme.textPrimary;
        }

        // Offset text slightly so it sits next to the star point
        ctx.fillText(node.text, node.x, node.y - node.fontSize * 0.7);

        // Subtitle badge
        ctx.font = `500 ${Math.max(10, node.fontSize * 0.35)}px 'Inter', sans-serif`;
        ctx.fillStyle = node.color;
        ctx.fillText(`×${node.count}`, node.x, node.y + node.fontSize * 0.65);

        ctx.shadowBlur = 0;
      });

      ctx.restore();
      animationFrameId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [theme, font]);

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

    // The label is drawn above the star point and the count badge below it, so a
    // plain radius around the point misses the very text the user is aiming at.
    // Test the box that actually spans label, star and badge.
    let found: GalaxyNode | null = null;
    for (const node of nodesRef.current) {
      const halfWidth = Math.max(20, measureTextWidth(node.text, node.fontSize, font.css) / 2 + 6);
      const top = node.y - node.fontSize * 0.7 - node.fontSize * 0.6;
      const bottom = node.y + node.fontSize * 0.65 + node.fontSize * 0.3;
      if (mx >= node.x - halfWidth && mx <= node.x + halfWidth && my >= top && my <= bottom) {
        found = node;
        break;
      }
    }
    setHoveredNode(found);
  };

  const handleMouseLeave = () => {
    setHoveredNode(null);
    setMousePos(null);
  };

  const handleClick = () => {
    if (hoveredNode) {
      onExcludeWord(hoveredNode.text);
      setHoveredNode(null);
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
      {hoveredNode && mousePos && (
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
            <span>{hoveredNode.text}</span>
            <span
              className="px-1.5 py-0.5 rounded text-[10px] font-mono"
              style={{ backgroundColor: `${hoveredNode.color}33`, color: hoveredNode.color }}
            >
              ×{hoveredNode.count}
            </span>
          </div>
          <div className="text-slate-400 mt-1 flex items-center justify-between gap-4">
            <span>Class frequency:</span>
            <span className="font-mono text-slate-200">{hoveredNode.percentage.toFixed(1)}%</span>
          </div>
          <div className="text-[10px] text-rose-400/90 mt-1.5">
            Click word to ban / exclude
          </div>
        </div>
      )}

      {words.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 pointer-events-none">
          <p className="text-sm font-medium text-slate-300">Awaiting speech for celestial spiral...</p>
        </div>
      )}
    </div>
  );
};
