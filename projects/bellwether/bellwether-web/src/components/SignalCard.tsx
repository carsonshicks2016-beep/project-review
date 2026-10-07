"use client";

import { useEffect, useState } from "react";
import { motion, animate } from "framer-motion";
import { useMotionLevel } from "@/lib/motion";
import { Signal } from "@/lib/useJson";

const dirColor = (d: string) =>
  d === "bullish" ? "#34d399" : d === "bearish" ? "#fb7185" : "#9aa6b8";

function Arrow({ dir, color }: { dir: string; color: string }) {
  const up = dir === "bullish";
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={color}
      strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {up ? <><path d="M7 17 17 7" /><path d="M8 7h9v9" /></>
         : <><path d="M7 7l10 10" /><path d="M17 8v9H8" /></>}
    </svg>
  );
}

export function SignalCard({ s, i }: { s: Signal; i: number }) {
  const { intensity } = useMotionLevel();
  const color = dirColor(s.direction);
  const [score, setScore] = useState(0);

  useEffect(() => {
    const controls = animate(0, s.materiality, {
      duration: 0.5 + 0.5 * intensity,
      delay: 0.15 + i * 0.08,
      ease: "easeOut",
      onUpdate: (v) => setScore(Math.round(v)),
    });
    return () => controls.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.materiality]);

  return (
    <motion.div
      className="glass lift"
      initial={{ opacity: 0, x: -24 * (0.3 + intensity) }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: i * 0.08, duration: 0.45, ease: [0.2, 0.85, 0.25, 1] }}
      style={{
        borderLeft: `3px solid ${color}`,
        padding: "14px 16px",
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.12), 0 0 ${26 * intensity}px ${color}33`,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span className="mono" style={{ fontSize: 14, fontWeight: 500 }}>{s.ticker}</span>
          <span style={{ fontSize: 11, padding: "2px 9px", borderRadius: 7, background: `${color}22`, color }}>
            {s.catalyst_type}
          </span>
          {s.form_type && (
            <span style={{ fontSize: 11, color: "var(--text-dim)" }}>{s.form_type}</span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <Arrow dir={s.direction} color={color} />
          <span className="mono" style={{ fontSize: 15, fontWeight: 500, color }}>
            {score}
            <span style={{ fontSize: 11, color: "var(--text-dim)" }}>/10</span>
          </span>
        </div>
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.5, marginBottom: 8 }}>{s.summary}</div>
      <div
        style={{
          fontSize: 12,
          color: "var(--text-dim)",
          fontStyle: "italic",
          borderLeft: "2px solid var(--glass-border)",
          paddingLeft: 10,
          lineHeight: 1.5,
        }}
      >
        {s.evidence_quote}
      </div>
    </motion.div>
  );
}
