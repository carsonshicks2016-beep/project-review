"use client";

import { motion } from "framer-motion";
import { SECTIONS, SectionKey } from "@/lib/sections";

function Bell() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M10 5a2 2 0 1 1 4 0c4 1 5 4 5 9 .5 2 1.5 3 2 3H3c.5 0 1.5-1 2-3 0-5 1-8 5-9" />
      <path d="M10 20a2 2 0 0 0 4 0" />
    </svg>
  );
}

export default function NavBar({
  active,
  onNav,
}: {
  active: SectionKey;
  onNav: (k: SectionKey) => void;
}) {
  return (
    <div
      className="glass"
      style={{ display: "flex", alignItems: "center", gap: 4, padding: "8px 10px", borderRadius: 16 }}
    >
      <span
        style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 500, fontSize: 14, letterSpacing: 0.4, padding: "0 10px 0 6px", color: "#fff" }}
      >
        <Bell />
        Bellwether
      </span>
      {SECTIONS.map((s) => (
        <button
          key={s.key}
          onClick={() => onNav(s.key)}
          style={{
            position: "relative",
            padding: "7px 14px",
            borderRadius: 11,
            fontSize: 13,
            fontWeight: active === s.key ? 500 : 400,
            color: active === s.key ? "#fff" : "var(--text-dim)",
            background: "transparent",
            border: "none",
            cursor: "pointer",
            transition: "color 0.2s",
          }}
        >
          {active === s.key && (
            <motion.span
              layoutId="navpill"
              style={{
                position: "absolute",
                inset: 0,
                background: "rgba(255,255,255,0.16)",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: 11,
                zIndex: -1,
              }}
              transition={{ type: "spring", stiffness: 420, damping: 34 }}
            />
          )}
          {s.label}
        </button>
      ))}
    </div>
  );
}
