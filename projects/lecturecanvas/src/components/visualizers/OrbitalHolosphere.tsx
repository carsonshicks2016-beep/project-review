import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { WordFrequency, VisualizerSettings } from '../../types';
import { THEMES, FONT_FAMILIES } from '../../utils/themeStyles';

interface HoloNode {
  text: string;
  count: number;
  percentage: number;
  color: string;
  baseSize: number;
  phi: number; // latitude -pi/2 to pi/2
  theta: number; // longitude 0 to 2pi
  lastSeen: number;
  // Computed 3D and projected 2D
  x3: number;
  y3: number;
  z3: number;
  projX: number;
  projY: number;
  projScale: number;
  alpha: number;
}

interface Props {
  words: WordFrequency[];
  settings: VisualizerSettings;
  onExcludeWord: (word: string) => void;
  exportRef: React.MutableRefObject<HTMLCanvasElement | null>;
}

export const OrbitalHolosphere: React.FC<Props> = ({
  words,
  settings,
  onExcludeWord,
  exportRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<HoloNode | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  // Rotation angles
  const rotYRef = useRef(0);
  const rotXRef = useRef(0.2);
  const isDraggingRef = useRef(false);
  const lastMousePosRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const nodesRef = useRef<HoloNode[]>([]);

  const theme = THEMES[settings.theme];
  const font = FONT_FAMILIES[settings.fontFamily];

  // Distribute words across the sphere using Fibonacci golden spiral lattice
  useEffect(() => {
    if (!containerRef.current || words.length === 0) {
      nodesRef.current = [];
      return;
    }

    const maxCount = d3.max(words, (d) => d.count) || 1;
    const minCount = d3.min(words, (d) => d.count) || 1;

    const fontScale = d3
      .scalePow()
      .exponent(0.6)
      .domain([minCount, maxCount])
      .range([13, 38]);

    const colorScale = d3.scaleOrdinal(theme.palette);
    const goldenRatio = (1 + Math.sqrt(5)) / 2;

    const newNodes: HoloNode[] = words.map((w, i) => {
      // Fibonacci sphere distribution: i = 0 is high frequency
      const y = 1 - (i / Math.max(1, words.length - 1)) * 2; // from 1 to -1
      const radiusAtY = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = 2 * Math.PI * i / goldenRatio;
      const phi = Math.asin(Math.max(-1, Math.min(1, y)));

      return {
        text: w.text,
        count: w.count,
        percentage: w.percentage,
        color: colorScale(w.text),
        baseSize: fontScale(w.count),
        phi,
        theta,
        lastSeen: w.lastSeen,
        x3: radiusAtY * Math.cos(theta),
        y3: y,
        z3: radiusAtY * Math.sin(theta),
        projX: 0,
        projY: 0,
        projScale: 1,
        alpha: 1,
      };
    });

    nodesRef.current = newNodes;
  }, [words, theme]);

  // Main 3D Render Loop
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

      // Auto-rotation when not dragging
      if (!isDraggingRef.current) {
        rotYRef.current += 0.003;
      }

      const sphereRadius = Math.min(width, height) * 0.38;
      const centerX = width / 2;
      const centerY = height / 2;
      const fov = 600;

      const cosY = Math.cos(rotYRef.current);
      const sinY = Math.sin(rotYRef.current);
      const cosX = Math.cos(rotXRef.current);
      const sinX = Math.sin(rotXRef.current);

      // Draw faint orbital longitude/latitude guide rings
      ctx.lineWidth = 1;
      ctx.strokeStyle = `${theme.glowColor.replace(/[\d.]+\)$/, '0.08)')}`;

      // Equatorial ring
      ctx.beginPath();
      for (let a = 0; a <= Math.PI * 2; a += 0.1) {
        const rx = Math.cos(a) * sphereRadius;
        const rz = Math.sin(a) * sphereRadius;
        const ry = 0;

        const x1 = rx * cosY + rz * sinY;
        const z1 = -rx * sinY + rz * cosY;
        const y2 = ry * cosX - z1 * sinX;
        const z2 = ry * sinX + z1 * cosX;

        const s = fov / (fov + z2);
        const px = centerX + x1 * s;
        const py = centerY + y2 * s;

        if (a === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();

      // Transform all nodes in 3D
      const nodes = nodesRef.current;
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const bx = n.x3 * sphereRadius;
        const by = n.y3 * sphereRadius;
        const bz = n.z3 * sphereRadius;

        // Yaw rotation (Y-axis)
        const x1 = bx * cosY + bz * sinY;
        const z1 = -bx * sinY + bz * cosY;

        // Pitch rotation (X-axis)
        const y2 = by * cosX - z1 * sinX;
        const z2 = by * sinX + z1 * cosX;

        // Perspective scale & projection
        const scale = fov / (fov + z2);
        n.projX = centerX + x1 * scale;
        n.projY = centerY + y2 * scale;
        n.projScale = scale;

        // Depth fog (front z2 < 0 is closer, back z2 > 0 is further)
        // Normalized depth: 0 (furthest back) to 1 (closest front)
        const depthNorm = 1 - (z2 + sphereRadius) / (sphereRadius * 2);
        n.alpha = Math.max(0.2, Math.min(1.0, depthNorm * 0.9 + 0.1));
        (n as any).depth = z2;
      }

      // Sort nodes by depth so rear nodes draw first
      const sorted = [...nodes].sort((a, b) => (b as any).depth - (a as any).depth);
      const now = Date.now();

      // Draw nodes
      for (let i = 0; i < sorted.length; i++) {
        const n = sorted[i];
        const isHovered = hoveredNode?.text === n.text;
        const isRecent = now - n.lastSeen < 6000;
        const fontSize = Math.max(9, n.baseSize * n.projScale);

        ctx.save();
        ctx.globalAlpha = isHovered ? 1.0 : n.alpha;

        // Node center halo
        const dotRadius = Math.max(2, 3 * n.projScale);
        ctx.beginPath();
        ctx.arc(n.projX, n.projY, dotRadius, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();

        if (isRecent || isHovered) {
          ctx.beginPath();
          ctx.arc(n.projX, n.projY, dotRadius * 3, 0, Math.PI * 2);
          ctx.strokeStyle = n.color;
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Word typography
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `600 ${fontSize}px ${font.css}`;

        if (isHovered) {
          ctx.shadowBlur = 18;
          ctx.shadowColor = n.color;
          ctx.fillStyle = '#ffffff';
        } else if (isRecent) {
          ctx.shadowBlur = 12;
          ctx.shadowColor = n.color;
          ctx.fillStyle = '#ffffff';
        } else {
          ctx.shadowBlur = n.alpha > 0.6 ? 4 : 0;
          ctx.shadowColor = n.color;
          ctx.fillStyle = n.alpha > 0.6 ? theme.textPrimary : theme.textSecondary;
        }

        ctx.fillText(n.text, n.projX, n.projY - fontSize * 0.7);

        // Count badge
        if (n.alpha > 0.45 || isHovered) {
          ctx.font = `500 ${Math.max(8, fontSize * 0.38)}px 'Inter', sans-serif`;
          ctx.fillStyle = n.color;
          ctx.fillText(`×${n.count}`, n.projX, n.projY + fontSize * 0.65);
        }

        ctx.restore();
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

  // Mouse drag 3D rotation controls
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDraggingRef.current = true;
    lastMousePosRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    setMousePos({ x: e.clientX, y: e.clientY });

    if (isDraggingRef.current) {
      const dx = e.clientX - lastMousePosRef.current.x;
      const dy = e.clientY - lastMousePosRef.current.y;
      rotYRef.current += dx * 0.006;
      rotXRef.current -= dy * 0.006;
      lastMousePosRef.current = { x: e.clientX, y: e.clientY };
      return;
    }

    // Hit test front-most nodes
    let found: HoloNode | null = null;
    const sortedFront = [...nodesRef.current]
      .filter((n) => n.alpha > 0.4)
      .sort((a, b) => (a as any).depth - (b as any).depth);

    for (const n of sortedFront) {
      const dx = mx - n.projX;
      const dy = my - n.projY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < Math.max(24, n.baseSize * n.projScale)) {
        found = n;
        break;
      }
    }
    setHoveredNode(found);
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleClick = () => {
    if (hoveredNode) {
      onExcludeWord(hoveredNode.text);
      setHoveredNode(null);
    }
  };

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden select-none cursor-grab active:cursor-grabbing">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          isDraggingRef.current = false;
          setHoveredNode(null);
          setMousePos(null);
        }}
        onClick={handleClick}
      />

      {/* 3D Interaction Hint */}
      <div className="absolute bottom-4 left-4 pointer-events-none px-2.5 py-1 rounded-lg bg-black/40 border border-white/10 text-[10px] text-slate-400 backdrop-blur-md flex items-center gap-1.5">
        <span>🌐 Drag to spin in 3D</span>
      </div>

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
          <div className="text-[10px] text-rose-400/90 mt-1.5">Click word to ban / exclude</div>
        </div>
      )}

      {words.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 pointer-events-none">
          <p className="text-sm font-medium text-slate-300">Awaiting speech for 3D holosphere...</p>
        </div>
      )}
    </div>
  );
};
