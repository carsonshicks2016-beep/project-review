export type SectionKey = "overview" | "signals" | "universe" | "backtest" | "audit";

export interface SectionTheme {
  key: SectionKey;
  label: string;
  c1: string; // background blob A
  c2: string; // background blob B
  accent: string; // UI accent
  sweep: string; // transition wipe color
}

export const SECTIONS: SectionTheme[] = [
  { key: "overview", label: "Overview", c1: "#14b8a6", c2: "#0e7490", accent: "#2dd4bf", sweep: "#0d9488" },
  { key: "signals", label: "Signals", c1: "#818cf8", c2: "#7c3aed", accent: "#a5b4fc", sweep: "#6366f1" },
  { key: "universe", label: "Universe", c1: "#f59e0b", c2: "#b45309", accent: "#fbbf24", sweep: "#d97706" },
  { key: "backtest", label: "Backtest", c1: "#fb7185", c2: "#9f1239", accent: "#fda4af", sweep: "#e11d48" },
  { key: "audit", label: "Audit", c1: "#10b981", c2: "#047857", accent: "#34d399", sweep: "#059669" },
];

export const sectionTheme = (k: SectionKey): SectionTheme =>
  SECTIONS.find((s) => s.key === k) ?? SECTIONS[0];
