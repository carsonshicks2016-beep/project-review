import type { ColorTheme, FontFamily } from '../types';

export interface ThemeColors {
  name: string;
  background: string;
  panelBg: string;
  borderColor: string;
  textPrimary: string;
  textSecondary: string;
  palette: string[];
  glowColor: string;
}

export const THEMES: Record<ColorTheme, ThemeColors> = {
  cyber: {
    name: 'Cyber Neon',
    background: '#090a0f',
    panelBg: 'rgba(15, 18, 28, 0.85)',
    borderColor: 'rgba(6, 182, 212, 0.25)',
    textPrimary: '#f8fafc',
    textSecondary: '#94a3b8',
    palette: ['#06b6d4', '#a855f7', '#ec4899', '#3b82f6', '#10b981', '#f59e0b', '#14b8a6'],
    glowColor: 'rgba(6, 182, 212, 0.6)',
  },
  editorial: {
    name: 'Warm Editorial',
    background: '#121110',
    panelBg: 'rgba(28, 25, 23, 0.85)',
    borderColor: 'rgba(217, 119, 6, 0.25)',
    textPrimary: '#fef3c7',
    textSecondary: '#d97706',
    palette: ['#f59e0b', '#d97706', '#b45309', '#f97316', '#ea580c', '#eab308', '#ca8a04'],
    glowColor: 'rgba(245, 158, 11, 0.5)',
  },
  bioluminescent: {
    name: 'Bioluminescent',
    background: '#041017',
    panelBg: 'rgba(7, 26, 38, 0.85)',
    borderColor: 'rgba(20, 184, 166, 0.25)',
    textPrimary: '#ccfbf1',
    textSecondary: '#2dd4bf',
    palette: ['#2dd4bf', '#06b6d4', '#38bdf8', '#34d399', '#10b981', '#0ea5e9', '#6ee7b7'],
    glowColor: 'rgba(45, 212, 191, 0.55)',
  },
  sunset: {
    name: 'Sunset Horizon',
    background: '#140c18',
    panelBg: 'rgba(32, 17, 40, 0.85)',
    borderColor: 'rgba(244, 63, 94, 0.25)',
    textPrimary: '#ffe4e6',
    textSecondary: '#fb7185',
    palette: ['#f43f5e', '#fb7185', '#e11d48', '#fb923c', '#f97316', '#c084fc', '#e879f9'],
    glowColor: 'rgba(244, 63, 94, 0.55)',
  },
  monochrome: {
    name: 'Minimal Titanium',
    background: '#0d0e12',
    panelBg: 'rgba(22, 24, 30, 0.85)',
    borderColor: 'rgba(255, 255, 255, 0.15)',
    textPrimary: '#ffffff',
    textSecondary: '#94a3b8',
    palette: ['#ffffff', '#e2e8f0', '#cbd5e1', '#94a3b8', '#64748b', '#38bdf8', '#818cf8'],
    glowColor: 'rgba(255, 255, 255, 0.4)',
  },
};

export const FONT_FAMILIES: Record<FontFamily, { name: string; css: string }> = {
  inter: {
    name: 'Modern Sans',
    css: "'Inter', sans-serif",
  },
  playfair: {
    name: 'Editorial Serif',
    css: "'Playfair Display', serif",
  },
  jetbrains: {
    name: 'Code Monospace',
    css: "'JetBrains Mono', monospace",
  },
  space: {
    name: 'Kinetic Grotesk',
    css: "'Space Grotesk', sans-serif",
  },
  cinzel: {
    name: 'Classical Cinzel',
    css: "'Cinzel', serif",
  },
};
