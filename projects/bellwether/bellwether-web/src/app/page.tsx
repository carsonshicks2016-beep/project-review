"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { SectionKey, sectionTheme } from "@/lib/sections";
import { useMotionLevel } from "@/lib/motion";
import LivingBackground from "@/components/LivingBackground";
import NavBar from "@/components/NavBar";
import IntensityDial from "@/components/IntensityDial";
import {
  OverviewSection,
  SignalsSection,
  UniverseSection,
  BacktestSection,
  AuditSection,
} from "@/components/sections";

export default function Page() {
  const [active, setActive] = useState<SectionKey>("overview");
  const [sweep, setSweep] = useState<string | null>(null);
  const { intensity } = useMotionLevel();
  const theme = sectionTheme(active);

  function nav(k: SectionKey) {
    if (k === active) return;
    if (intensity > 0.05) {
      const t = sectionTheme(k);
      setSweep(t.sweep);
      window.setTimeout(() => setSweep(null), 700);
    }
    setActive(k);
  }

  return (
    <main style={{ position: "relative", minHeight: "100vh" }}>
      <LivingBackground c1={theme.c1} c2={theme.c2} />

      <div
        style={{
          position: "relative",
          zIndex: 2,
          maxWidth: 1060,
          margin: "0 auto",
          padding: "22px 20px 70px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
            marginBottom: 26,
          }}
        >
          <NavBar active={active} onNav={nav} />
          <IntensityDial />
        </div>

        <AnimatePresence mode="wait">
          <motion.h1
            key={active + "-t"}
            initial={{ rotateX: -88, opacity: 0 }}
            animate={{ rotateX: 0, opacity: 1 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.5, ease: [0.2, 0.85, 0.25, 1] }}
            style={{
              transformPerspective: 800,
              transformOrigin: "top center",
              fontSize: 34,
              fontWeight: 500,
              margin: "0 6px 20px",
            }}
          >
            {theme.label}
          </motion.h1>
        </AnimatePresence>

        <AnimatePresence mode="wait">
          <motion.div
            key={active}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.32 }}
          >
            {active === "overview" && <OverviewSection />}
            {active === "signals" && <SignalsSection />}
            {active === "universe" && <UniverseSection />}
            {active === "backtest" && <BacktestSection theme={theme} />}
            {active === "audit" && <AuditSection theme={theme} />}
          </motion.div>
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {sweep && (
          <motion.div
            initial={{ x: "-100%" }}
            animate={{ x: "100%" }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.66, ease: "easeInOut" }}
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 5,
              pointerEvents: "none",
              background: `linear-gradient(110deg, ${sweep}, transparent 70%)`,
            }}
          />
        )}
      </AnimatePresence>
    </main>
  );
}
