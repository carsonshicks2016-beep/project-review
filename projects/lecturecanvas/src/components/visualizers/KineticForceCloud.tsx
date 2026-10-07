import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { WordFrequency, VisualizerSettings } from '../../types';
import { THEMES, FONT_FAMILIES } from '../../utils/themeStyles';

interface KineticNode extends d3.SimulationNodeDatum {
  text: string;
  count: number;
  radius: number;
  fontSize: number;
  color: string;
  lastSeen: number;
  percentage: number;
}

/**
 * Measures rendered text width using a scratch canvas.
 *
 * Estimating width as `length * fontSize * 0.32` is wildly wrong for long words:
 * a 30-character term at 48px produced a 475px-radius bubble that ran off the
 * screen. Measuring costs one canvas call per word per relayout.
 */
let measureContext: CanvasRenderingContext2D | null = null;

function measureTextWidth(text: string, fontSize: number, fontCss: string): number {
  if (!measureContext) {
    measureContext = document.createElement('canvas').getContext('2d');
  }
  if (!measureContext) return text.length * fontSize * 0.32;
  measureContext.font = `600 ${fontSize}px ${fontCss}`;
  return measureContext.measureText(text).width;
}

interface Props {
  words: WordFrequency[];
  settings: VisualizerSettings;
  onExcludeWord: (word: string) => void;
  exportRef: React.MutableRefObject<HTMLCanvasElement | null>;
}

export const KineticForceCloud: React.FC<Props> = ({
  words,
  settings,
  onExcludeWord,
  exportRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simulationRef = useRef<d3.Simulation<KineticNode, undefined> | null>(null);
  const nodesRef = useRef<KineticNode[]>([]);
  const [hoveredNode, setHoveredNode] = useState<KineticNode | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const hoveredNodeRef = useRef<KineticNode | null>(null);
  const dragRef = useRef<{ node: KineticNode; moved: boolean } | null>(null);
  const draggedRecentlyRef = useRef(false);

  // The animation loop reads hover state through a ref so that moving the mouse
  // does not tear down and restart the requestAnimationFrame loop.
  useEffect(() => {
    hoveredNodeRef.current = hoveredNode;
  }, [hoveredNode]);

  const theme = THEMES[settings.theme];
  const font = FONT_FAMILIES[settings.fontFamily];

  // Sync canvas dimensions
  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current || !canvasRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvasRef.current.width = rect.width * dpr;
      canvasRef.current.height = rect.height * dpr;
      canvasRef.current.style.width = `${rect.width}px`;
      canvasRef.current.style.height = `${rect.height}px`;

      if (simulationRef.current) {
        simulationRef.current
          .force('center', d3.forceCenter(rect.width / 2, rect.height / 2))
          .alpha(0.3)
          .restart();
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Update nodes and force simulation when words or theme change
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

    // Responsive scaling
    const minFontSize = Math.max(14, Math.min(width, height) * 0.022);
    const maxFontSize = Math.max(48, Math.min(width, height) * 0.09);

    const fontScale = d3
      .scalePow()
      .exponent(0.65)
      .domain([minCount, maxCount])
      .range([minFontSize, maxFontSize]);

    const colorScale = d3.scaleOrdinal(theme.palette);

    // Maintain existing node positions for smooth morphing transitions
    const existingMap = new Map(nodesRef.current.map((n) => [n.text, n]));

    // No single bubble may swallow the canvas, however long its word is.
    const maxRadius = Math.min(width, height) * 0.22;

    const newNodes: KineticNode[] = words.map((w) => {
      let fontSize = fontScale(w.count);
      let radius = Math.max(fontSize * 0.9, measureTextWidth(w.text, fontSize, font.css) / 2 + 14);

      // A long word shrinks to fit its cap rather than pushing everything off screen.
      if (radius > maxRadius) {
        fontSize = Math.max(11, fontSize * (maxRadius / radius));
        radius = Math.max(
          fontSize * 0.9,
          Math.min(maxRadius, measureTextWidth(w.text, fontSize, font.css) / 2 + 12)
        );
      }

      const existing = existingMap.get(w.text);

      return {
        text: w.text,
        count: w.count,
        radius,
        fontSize,
        color: colorScale(w.text),
        lastSeen: w.lastSeen,
        percentage: w.percentage,
        x: existing?.x ?? width / 2 + (Math.random() - 0.5) * 150,
        y: existing?.y ?? height / 2 + (Math.random() - 0.5) * 150,
        vx: existing?.vx ?? 0,
        vy: existing?.vy ?? 0,
        // Keep a word pinned under the cursor if it is mid-drag.
        fx: existing?.fx ?? null,
        fy: existing?.fy ?? null,
      };
    });

    nodesRef.current = newNodes;

    // Node objects are rebuilt on every word update; re-point an in-flight drag
    // at the new object, or it would go on moving a discarded one.
    if (dragRef.current) {
      const replacement = newNodes.find((n) => n.text === dragRef.current!.node.text);
      if (replacement) dragRef.current.node = replacement;
      else dragRef.current = null;
    }

    // Keep every bubble fully inside the canvas. forceCenter pulls toward the
    // middle but does nothing to stop a crowded cluster from pushing its
    // outermost words off the edge, where they are simply unreadable.
    // Bounds are read per tick rather than captured, so a resized window takes
    // effect immediately instead of clamping to the old dimensions.
    const containWithinCanvas = () => {
      const box = containerRef.current?.getBoundingClientRect();
      const boundsWidth = box?.width || width;
      const boundsHeight = box?.height || height;
      for (const node of nodesRef.current) {
        if (node.x === undefined || node.y === undefined) continue;
        const margin = node.radius + 4;
        node.x = Math.max(margin, Math.min(Math.max(margin, boundsWidth - margin), node.x));
        node.y = Math.max(margin, Math.min(Math.max(margin, boundsHeight - margin), node.y));
      }
    };

    // Rebuild or update D3 force simulation
    if (!simulationRef.current) {
      simulationRef.current = d3
        .forceSimulation<KineticNode>(newNodes)
        .velocityDecay(0.3)
        .force('center', d3.forceCenter(width / 2, height / 2).strength(0.06))
        .force('charge', d3.forceManyBody<KineticNode>().strength((d) => -d.radius * 2.2 * settings.gravityStrength))
        .force(
          'collision',
          d3.forceCollide<KineticNode>().radius((d) => d.radius + 8).iterations(3)
        )
        .on('tick', containWithinCanvas);
    } else {
      simulationRef.current.nodes(newNodes);
      simulationRef.current
        .force('center', d3.forceCenter(width / 2, height / 2).strength(0.06))
        .force('charge', d3.forceManyBody<KineticNode>().strength((d) => -d.radius * 2.2 * settings.gravityStrength))
        .force(
          'collision',
          d3.forceCollide<KineticNode>().radius((d) => d.radius + 8).iterations(3)
        )
        .on('tick', containWithinCanvas)
        .alpha(0.4)
        .restart();
    }
  }, [words, theme, font, settings.gravityStrength]);

  // The force simulation runs on its own d3 timer; without this it keeps ticking
  // after the component unmounts (e.g. switching to another diagram style).
  useEffect(() => {
    return () => {
      simulationRef.current?.stop();
      simulationRef.current = null;
    };
  }, []);

  // Main Canvas Rendering Loop
  useEffect(() => {
    let animationFrameId: number;

    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      // Background ambient star dust / constellation subtle grid
      ctx.fillStyle = theme.background;
      ctx.fillRect(0, 0, width, height);

      // Draw faint constellation connecting lines between nearest frequent words
      const nodes = nodesRef.current;
      ctx.lineWidth = 1;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const n1 = nodes[i];
          const n2 = nodes[j];
          if (n1.x === undefined || n1.y === undefined || n2.x === undefined || n2.y === undefined) {
            continue;
          }
          const dx = n1.x - n2.x;
          const dy = n1.y - n2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          // Connect if close enough
          if (dist < 180) {
            const alpha = (1 - dist / 180) * 0.15;
            ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(n1.x, n1.y);
            ctx.lineTo(n2.x, n2.y);
            ctx.stroke();
          }
        }
      }

      const now = Date.now();
      const hovered = hoveredNodeRef.current;

      // Render each word node
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        if (node.x === undefined || node.y === undefined) continue;

        const isHovered = hovered?.text === node.text;
        const timeSinceSpoken = now - (node.lastSeen || 0);
        const isRecent = timeSinceSpoken < 6000; // Spoken in the last 6 seconds

        // Recency glow ripple animation
        if (isRecent) {
          const pulse = (Math.sin(now * 0.008) + 1) * 0.5;
          const rippleRadius = node.radius + 6 + pulse * 12;
          ctx.beginPath();
          ctx.arc(node.x, node.y, rippleRadius, 0, Math.PI * 2);
          ctx.strokeStyle = node.color;
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = 0.4 * (1 - timeSinceSpoken / 6000);
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }

        // Circular aura/bubble
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);

        // Gradient bubble fill
        const gradient = ctx.createRadialGradient(
          node.x - node.radius * 0.3,
          node.y - node.radius * 0.3,
          node.radius * 0.1,
          node.x,
          node.y,
          node.radius
        );

        if (isHovered) {
          gradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
          gradient.addColorStop(1, `${node.color}55`);
          ctx.shadowBlur = 24;
          ctx.shadowColor = node.color;
        } else {
          gradient.addColorStop(0, `${node.color}22`);
          gradient.addColorStop(1, `${node.color}08`);
          ctx.shadowBlur = isRecent ? 16 : 6;
          ctx.shadowColor = isRecent ? node.color : 'rgba(0,0,0,0.5)';
        }

        ctx.fillStyle = gradient;
        ctx.fill();

        // Border ring
        ctx.strokeStyle = isHovered ? '#ffffff' : `${node.color}66`;
        ctx.lineWidth = isHovered ? 2.5 : 1.2;
        ctx.stroke();

        // Reset shadow for text
        ctx.shadowBlur = 0;

        // Render Typography
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `600 ${node.fontSize}px ${font.css}`;
        ctx.fillStyle = isHovered ? '#ffffff' : theme.textPrimary;
        ctx.fillText(node.text, node.x, node.y - 2);

        // Render frequency badge beneath text
        const badgeFont = Math.max(10, node.fontSize * 0.32);
        ctx.font = `500 ${badgeFont}px 'Inter', sans-serif`;
        ctx.fillStyle = `${node.color}ee`;
        ctx.fillText(`×${node.count}`, node.x, node.y + (node.fontSize * 0.62));
      }

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

  // Mouse Interaction: Hover & Dragging
  const nodeAt = (x: number, y: number): KineticNode | null => {
    for (let i = nodesRef.current.length - 1; i >= 0; i--) {
      const node = nodesRef.current[i];
      if (node.x === undefined || node.y === undefined) continue;
      const dx = x - node.x;
      const dy = y - node.y;
      if (dx * dx + dy * dy <= node.radius * node.radius) return node;
    }
    return null;
  };

  const canvasPoint = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = canvasPoint(e);

    const drag = dragRef.current;
    if (drag) {
      drag.moved = true;
      drag.node.fx = x;
      drag.node.fy = y;
      simulationRef.current?.alphaTarget(0.25).restart();
      setMousePos({ x: e.clientX, y: e.clientY });
      return;
    }

    const found = nodeAt(x, y);
    // Only track the cursor while a tooltip is actually on screen.
    if (found || hoveredNodeRef.current) setMousePos({ x: e.clientX, y: e.clientY });
    if (found?.text !== hoveredNodeRef.current?.text) setHoveredNode(found);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = canvasPoint(e);
    const node = nodeAt(x, y);
    if (!node) return;
    node.fx = node.x;
    node.fy = node.y;
    dragRef.current = { node, moved: false };
    simulationRef.current?.alphaTarget(0.25).restart();
  };

  const releaseDrag = (): boolean => {
    const drag = dragRef.current;
    if (!drag) return false;
    drag.node.fx = null;
    drag.node.fy = null;
    dragRef.current = null;
    simulationRef.current?.alphaTarget(0);
    return drag.moved;
  };

  const handleMouseUp = () => {
    // A drag ends here; the click that follows must not also ban the word.
    const wasDragged = releaseDrag();
    if (wasDragged) draggedRecentlyRef.current = true;
  };

  const handleMouseLeave = () => {
    releaseDrag();
    setHoveredNode(null);
    setMousePos(null);
  };

  const handleClick = () => {
    if (draggedRecentlyRef.current) {
      draggedRecentlyRef.current = false;
      return;
    }
    if (hoveredNode) {
      onExcludeWord(hoveredNode.text);
      setHoveredNode(null);
    }
  };

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden select-none">
      <canvas
        ref={canvasRef}
        className={`w-full h-full ${hoveredNode ? 'cursor-grab active:cursor-grabbing' : 'cursor-default'}`}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
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
            <span>Class repetition:</span>
            <span className="font-mono text-slate-200">{hoveredNode.percentage.toFixed(1)}%</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-1.5">
            Drag to reposition · click to exclude
          </div>
        </div>
      )}

      {/* Empty state hint */}
      {words.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 pointer-events-none">
          <div className="p-4 rounded-full bg-slate-900/60 border border-slate-800 mb-3 animate-pulse">
            <svg className="w-8 h-8 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z" />
            </svg>
          </div>
          <p className="text-sm font-medium text-slate-300">Listening for classroom lecture...</p>
          <p className="text-xs text-slate-500 mt-1">Start speaking or click "Load Sample Lecture" below to see the diagram form.</p>
        </div>
      )}
    </div>
  );
};
