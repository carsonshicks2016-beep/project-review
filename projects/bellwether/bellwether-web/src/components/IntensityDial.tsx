"use client";

import { useMotionLevel } from "@/lib/motion";

export default function IntensityDial() {
  const { intensity, setIntensity } = useMotionLevel();
  return (
    <div
      className="glass"
      style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 14px", borderRadius: 14 }}
      title="Motion intensity"
    >
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--text-dim)"
        strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M13 2 3 14h7l-1 8 10-12h-7l1-8Z" />
      </svg>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={intensity}
        onChange={(e) => setIntensity(parseFloat(e.target.value))}
        style={{ width: 90 }}
        aria-label="Motion intensity"
      />
      <span className="mono" style={{ fontSize: 12, width: 32, textAlign: "right", color: "var(--text-dim)" }}>
        {Math.round(intensity * 100)}%
      </span>
    </div>
  );
}
