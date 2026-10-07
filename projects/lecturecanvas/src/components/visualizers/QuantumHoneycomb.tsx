import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { WordFrequency, VisualizerSettings } from '../../types';
import { THEMES, FONT_FAMILIES } from '../../utils/themeStyles';

interface HoneycombNode {
  text: string;
  count: number;
  percentage: number;
  color: string;
  radius: number;
  fontSize: number;
  x: number;
  y: number;
  lastSeen: number;
  neighbors: number[];
}

interface Props {
  words: WordFrequency[];
  settings: VisualizerSettings;
  onExcludeWord: (word: string) => void;
  exportRef: React.MutableRefObject<HTMLCanvasElement | null>;
}

export const QuantumHoneycomb: React.FC<Props> = ({
  words,
  settings,
  onExcludeWord,
  exportRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<HoneycombNode | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const nodesRef = useRef<HoneycombNode[]>([]);

  const theme = THEMES[settings.theme];
  const font = FONT_FAMILIES[settings.fontFamily];

  // Distribute words in a hexagonal spiral grid
  useEffect(() => {
    if (!containerRef.current || words.length === 0) {
      nodesRef.current = [];
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
      .range([12, 28]);

    const colorScale = d3.scaleOrdinal(theme.palette);

    // Hexagonal ring coordinates generator
    const hexRadius = Math.max(34, Math.min(58, Math.min(width, height) / (Math.sqrt(words.length) * 2.6)));
    const hexH = hexRadius * Math.sqrt(3);

    const hexPositions: { x: number; y: number }[] = [{ x: 0, y: 0 }];

    // Generate concentric rings of hexagons: ring 1 (6), ring 2 (12), ring 3 (18)...
    const directions = [
      { q: 1, r: -1 },
      { q: 0, r: -1 },
      { q: -1, r: 0 },
      { q: -1, r: 1 },
      { q: 0, r: 1 },
      { q: 1, r: 0 },
    ];

    let ring = 1;
    while (hexPositions.length < words.length + 10) {
      let q = 0;
      let r = ring;
      for (let i = 0; i < 6; i++) {
        for (let j = 0; j < ring; j++) {
          const x = hexRadius * (3 / 2 * q);
          const y = hexRadius * (Math.sqrt(3) / 2 * q + Math.sqrt(3) * r);
          hexPositions.push({ x, y });
          q += directions[i].q;
          r += directions[i].r;
        }
      }
      ring++;
    }

    const centerX = width / 2;
    const centerY = height / 2;

    const newNodes: HoneycombNode[] = words.map((w, i) => {
      const pos = hexPositions[i] || { x: 0, y: 0 };
      const fontSize = fontScale(w.count);
      const radius = hexRadius * (0.82 + (w.count / maxCount) * 0.25);

      return {
        text: w.text,
        count: w.count,
        percentage: w.percentage,
        color: colorScale(w.text),
        radius,
        fontSize,
        x: centerX + pos.x,
        y: centerY + pos.y,
        lastSeen: w.lastSeen,
        neighbors: [],
      };
    });

    // Determine neighbor connections for energy conduits
    for (let i = 0; i < newNodes.length; i++) {
      for (let j = i + 1; j < newNodes.length; j++) {
        const dx = newNodes[i].x - newNodes[j].x;
        const dy = newNodes[i].y - newNodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < hexH * 1.35) {
          newNodes[i].neighbors.push(j);
        }
      }
    }

    nodesRef.current = newNodes;
  }, [words, theme]);

  // Main Canvas Render Loop
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

      // Background
      ctx.fillStyle = theme.background;
      ctx.fillRect(0, 0, width, height);

      const nodes = nodesRef.current;
      const now = Date.now();

      // 1. Draw glowing energy conduits between adjacent honeycomb chambers
      ctx.lineWidth = 1;
      for (let i = 0; i < nodes.length; i++) {
        const n1 = nodes[i];
        for (const j of n1.neighbors) {
          const n2 = nodes[j];
          const isLinkActive = hoveredNode?.text === n1.text || hoveredNode?.text === n2.text;

          ctx.beginPath();
          ctx.moveTo(n1.x, n1.y);
          ctx.lineTo(n2.x, n2.y);

          if (isLinkActive) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.shadowBlur = 10;
            ctx.shadowColor = n1.color;
            ctx.stroke();
            ctx.shadowBlur = 0;
            ctx.lineWidth = 1;
          } else {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.stroke();
          }
        }
      }

      // Helper function to draw hexagon path
      const drawHexagon = (cx: number, cy: number, r: number) => {
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
          const angle = (Math.PI / 3) * i;
          const x = cx + r * Math.cos(angle);
          const y = cy + r * Math.sin(angle);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
      };

      // 2. Draw Hexagonal Crystal Chambers
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const isHovered = hoveredNode?.text === node.text;
        const isRecent = now - node.lastSeen < 6000;

        drawHexagon(node.x, node.y, node.radius);

        if (isHovered) {
          ctx.fillStyle = `${node.color}35`;
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2.5;
          ctx.shadowBlur = 18;
          ctx.shadowColor = node.color;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else if (isRecent) {
          const pulse = (Math.sin(now * 0.009) + 1) * 0.5;
          ctx.fillStyle = `${node.color}20`;
          ctx.fill();
          ctx.strokeStyle = node.color;
          ctx.lineWidth = 1.5 + pulse;
          ctx.shadowBlur = 12;
          ctx.shadowColor = node.color;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else {
          ctx.fillStyle = `${node.color}0c`;
          ctx.fill();
          ctx.strokeStyle = `${node.color}40`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Internal faceted reflection line
        ctx.beginPath();
        ctx.moveTo(node.x, node.y - node.radius);
        ctx.lineTo(node.x, node.y + node.radius);
        ctx.strokeStyle = `${node.color}15`;
        ctx.stroke();

        // Typography
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `700 ${node.fontSize}px ${font.css}`;

        if (isHovered) {
          ctx.shadowBlur = 16;
          ctx.shadowColor = node.color;
          ctx.fillStyle = '#ffffff';
        } else if (isRecent) {
          ctx.shadowBlur = 10;
          ctx.shadowColor = node.color;
          ctx.fillStyle = '#ffffff';
        } else {
          ctx.shadowBlur = 3;
          ctx.shadowColor = node.color;
          ctx.fillStyle = theme.textPrimary;
        }

        ctx.fillText(node.text, node.x, node.y - 2);

        // Repetition count badge
        ctx.shadowBlur = 0;
        ctx.font = `600 ${Math.max(8, node.fontSize * 0.38)}px 'Inter', sans-serif`;
        ctx.fillStyle = isHovered ? '#ffffff' : node.color;
        ctx.fillText(`×${node.count}`, node.x, node.y + node.fontSize * 0.65);
      }

      ctx.restore();
      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [theme, font, hoveredNode]);

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

    let found: HoneycombNode | null = null;
    for (const node of nodesRef.current) {
      const dx = mx - node.x;
      const dy = my - node.y;
      if (dx * dx + dy * dy <= node.radius * node.radius) {
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
          <div className="text-[10px] text-rose-400/90 mt-1.5">Click chamber to ban / exclude</div>
        </div>
      )}

      {words.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 pointer-events-none">
          <p className="text-sm font-medium text-slate-300">Awaiting speech for Quantum Honeycomb...</p>
        </div>
      )}
    </div>
  );
};
