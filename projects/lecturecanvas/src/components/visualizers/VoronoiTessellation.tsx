import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { WordFrequency, VisualizerSettings } from '../../types';
import { THEMES, FONT_FAMILIES } from '../../utils/themeStyles';

interface VoronoiNode {
  text: string;
  count: number;
  percentage: number;
  color: string;
  fontSize: number;
  x: number;
  y: number;
  lastSeen: number;
  polygon?: [number, number][];
}

interface Props {
  words: WordFrequency[];
  settings: VisualizerSettings;
  onExcludeWord: (word: string) => void;
  exportRef: React.MutableRefObject<HTMLCanvasElement | null>;
}

export const VoronoiTessellation: React.FC<Props> = ({
  words,
  settings,
  onExcludeWord,
  exportRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const nodesRef = useRef<VoronoiNode[]>([]);
  const delaunayRef = useRef<d3.Delaunay<[number, number]> | null>(null);

  const theme = THEMES[settings.theme];
  const font = FONT_FAMILIES[settings.fontFamily];

  // Calculate Voronoi points and cells
  useEffect(() => {
    if (!containerRef.current || words.length === 0) {
      nodesRef.current = [];
      delaunayRef.current = null;
      return;
    }

    const rect = containerRef.current.getBoundingClientRect();
    const width = rect.width || 800;
    const height = rect.height || 600;

    const maxCount = d3.max(words, (d) => d.count) || 1;
    const minCount = d3.min(words, (d) => d.count) || 1;

    const fontScale = d3
      .scalePow()
      .exponent(0.65)
      .domain([minCount, maxCount])
      .range([13, 36]);

    const colorScale = d3.scaleOrdinal(theme.palette);

    // Distribute points organically around the canvas center
    const points: [number, number][] = [];
    const existingMap = new Map(nodesRef.current.map((n) => [n.text, n]));

    const newNodes: VoronoiNode[] = words.map((w, i) => {
      const existing = existingMap.get(w.text);
      let x = existing?.x;
      let y = existing?.y;

      if (x === undefined || y === undefined) {
        // Spiral lattice placement
        const angle = i * 137.5 * (Math.PI / 180);
        const radius = Math.sqrt(i / Math.max(1, words.length)) * (Math.min(width, height) * 0.42);
        x = width / 2 + Math.cos(angle) * radius + (Math.random() - 0.5) * 30;
        y = height / 2 + Math.sin(angle) * radius + (Math.random() - 0.5) * 30;
      }

      // Clamp within bounds
      x = Math.max(40, Math.min(width - 40, x));
      y = Math.max(40, Math.min(height - 40, y));

      points.push([x, y]);

      return {
        text: w.text,
        count: w.count,
        percentage: w.percentage,
        color: colorScale(w.text),
        fontSize: fontScale(w.count),
        x,
        y,
        lastSeen: w.lastSeen,
      };
    });

    if (points.length > 0) {
      const delaunay = d3.Delaunay.from(points);
      const voronoi = delaunay.voronoi([0, 0, width, height]);

      for (let i = 0; i < newNodes.length; i++) {
        const poly = voronoi.cellPolygon(i);
        if (poly) {
          newNodes[i].polygon = poly as [number, number][];
          // Recenter text at polygon centroid
          const centroid = d3.polygonCentroid(poly);
          newNodes[i].x = centroid[0];
          newNodes[i].y = centroid[1];
        }
      }
      delaunayRef.current = delaunay;
    }

    nodesRef.current = newNodes;
  }, [words, theme]);

  // Canvas Render Loop
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

      // Render each Voronoi stained-glass cell
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const poly = node.polygon;
        if (!poly || poly.length < 3) continue;

        const isHovered = hoveredIndex === i;
        const isRecent = now - node.lastSeen < 6000;

        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let p = 1; p < poly.length; p++) {
          ctx.lineTo(poly[p][0], poly[p][1]);
        }
        ctx.closePath();

        // Stained glass gradient
        const grad = ctx.createRadialGradient(
          node.x,
          node.y,
          5,
          node.x,
          node.y,
          Math.max(40, node.fontSize * 3)
        );

        if (isHovered) {
          grad.addColorStop(0, `${node.color}55`);
          grad.addColorStop(1, `${node.color}22`);
          ctx.fillStyle = grad;
          ctx.fill();

          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2.5;
          ctx.shadowBlur = 18;
          ctx.shadowColor = node.color;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else if (isRecent) {
          const pulse = (Math.sin(now * 0.008) + 1) * 0.5;
          grad.addColorStop(0, `${node.color}44`);
          grad.addColorStop(1, `${node.color}15`);
          ctx.fillStyle = grad;
          ctx.fill();

          ctx.strokeStyle = node.color;
          ctx.lineWidth = 1.5 + pulse;
          ctx.shadowBlur = 10;
          ctx.shadowColor = node.color;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else {
          grad.addColorStop(0, `${node.color}22`);
          grad.addColorStop(1, 'rgba(255,255,255,0.02)');
          ctx.fillStyle = grad;
          ctx.fill();

          ctx.strokeStyle = 'rgba(255, 255, 255, 0.09)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Cell Center Dot
        ctx.beginPath();
        ctx.arc(node.x, node.y, isHovered ? 4 : 2, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.fill();

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
          ctx.shadowBlur = 4;
          ctx.shadowColor = `${node.color}44`;
          ctx.fillStyle = theme.textPrimary;
        }

        ctx.fillText(node.text, node.x, node.y - 2);

        // Repetition badge
        ctx.shadowBlur = 0;
        ctx.font = `600 ${Math.max(9, node.fontSize * 0.36)}px 'Inter', sans-serif`;
        ctx.fillStyle = isHovered ? '#ffffff' : node.color;
        ctx.fillText(`×${node.count}`, node.x, node.y + node.fontSize * 0.65);
      }

      ctx.restore();
      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [theme, font, hoveredIndex]);

  // Connect export ref
  useEffect(() => {
    if (exportRef && canvasRef.current) {
      exportRef.current = canvasRef.current;
    }
  }, [exportRef]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !delaunayRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setMousePos({ x: e.clientX, y: e.clientY });

    const idx = delaunayRef.current.find(mx, my);
    if (idx !== undefined && idx < nodesRef.current.length) {
      setHoveredIndex(idx);
    } else {
      setHoveredIndex(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
    setMousePos(null);
  };

  const handleClick = () => {
    if (hoveredIndex !== null && nodesRef.current[hoveredIndex]) {
      onExcludeWord(nodesRef.current[hoveredIndex].text);
      setHoveredIndex(null);
    }
  };

  const hoveredNode = hoveredIndex !== null ? nodesRef.current[hoveredIndex] : null;

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
          <div className="text-[10px] text-rose-400/90 mt-1.5">Click cell to ban / exclude</div>
        </div>
      )}

      {words.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 pointer-events-none">
          <p className="text-sm font-medium text-slate-300">Awaiting speech for Voronoi tessellation...</p>
        </div>
      )}
    </div>
  );
};
