"use client";

import { useEffect } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useMotionLevel } from "@/lib/motion";

export default function LivingBackground({ c1, c2 }: { c1: string; c2: string }) {
  const { intensity } = useMotionLevel();
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 40, damping: 20 });
  const sy = useSpring(my, { stiffness: 40, damping: 20 });

  useEffect(() => {
    const h = (e: PointerEvent) => {
      mx.set(e.clientX / window.innerWidth - 0.5);
      my.set(e.clientY / window.innerHeight - 0.5);
    };
    window.addEventListener("pointermove", h);
    return () => window.removeEventListener("pointermove", h);
  }, [mx, my]);

  const par = 70 * intensity;
  const x1 = useTransform(sx, (v) => v * par);
  const y1 = useTransform(sy, (v) => v * par);
  const x2 = useTransform(sx, (v) => -v * par);
  const y2 = useTransform(sy, (v) => -v * par);

  const dur1 = `${24 - 13 * intensity}s`;
  const dur2 = `${28 - 13 * intensity}s`;
  const animate = intensity > 0.04;

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 0, overflow: "hidden" }}>
      <motion.div
        style={{ x: x1, y: y1, position: "absolute", top: "-18%", left: "-12%", width: "62vw", height: "62vw" }}
      >
        <div
          style={{
            width: "100%",
            height: "100%",
            borderRadius: "50%",
            filter: "blur(80px)",
            background: `radial-gradient(circle, ${c1}, transparent 60%)`,
            opacity: 0.5,
            mixBlendMode: "screen",
            transition: "background 0.9s ease",
            animation: animate ? `drift1 ${dur1} ease-in-out infinite` : "none",
          }}
        />
      </motion.div>
      <motion.div
        style={{ x: x2, y: y2, position: "absolute", bottom: "-22%", right: "-12%", width: "66vw", height: "66vw" }}
      >
        <div
          style={{
            width: "100%",
            height: "100%",
            borderRadius: "50%",
            filter: "blur(90px)",
            background: `radial-gradient(circle, ${c2}, transparent 60%)`,
            opacity: 0.46,
            mixBlendMode: "screen",
            transition: "background 0.9s ease",
            animation: animate ? `drift2 ${dur2} ease-in-out infinite` : "none",
          }}
        />
      </motion.div>
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "radial-gradient(ellipse at 50% -10%, rgba(255,255,255,0.05), transparent 55%)",
        }}
      />
    </div>
  );
}
