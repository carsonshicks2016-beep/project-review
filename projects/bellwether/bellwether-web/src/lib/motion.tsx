"use client";

import { createContext, useContext, useEffect, useState } from "react";

interface MotionCtx {
  intensity: number; // 0..1 — the live "motion dial"
  setIntensity: (n: number) => void;
  reduced: boolean; // OS prefers-reduced-motion
}

const Ctx = createContext<MotionCtx>({
  intensity: 0.7,
  setIntensity: () => {},
  reduced: false,
});

export const useMotionLevel = () => useContext(Ctx);

export function MotionProvider({ children }: { children: React.ReactNode }) {
  const [intensity, setIntensity] = useState(0.7);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const stored = localStorage.getItem("bw-intensity");
    if (stored !== null) setIntensity(parseFloat(stored));
    else if (mq.matches) setIntensity(0.15);
  }, []);

  useEffect(() => {
    localStorage.setItem("bw-intensity", String(intensity));
  }, [intensity]);

  return (
    <Ctx.Provider value={{ intensity, setIntensity, reduced }}>
      {children}
    </Ctx.Provider>
  );
}
