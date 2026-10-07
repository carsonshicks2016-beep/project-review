import type { SimStats } from '../sim/types';

interface HudField {
  key: keyof Pick<
    SimStats,
    'waterCount' | 'nodeCount' | 'brokenBeams' | 'floodedNodes' | 'fps' | 'backend'
  >;
  label: string;
  format: (stats: SimStats) => string;
}

const FIELDS: HudField[] = [
  {
    key: 'waterCount',
    label: 'Water',
    format: (s) => String(s.waterCount),
  },
  {
    key: 'nodeCount',
    label: 'Parts',
    format: (s) => String(s.nodeCount),
  },
  {
    key: 'brokenBeams',
    label: 'Broken',
    format: (s) => String(s.brokenBeams),
  },
  {
    key: 'floodedNodes',
    label: 'Flooded',
    format: (s) => String(s.floodedNodes),
  },
  {
    key: 'fps',
    label: 'FPS',
    format: (s) => String(Math.round(s.fps)),
  },
  {
    key: 'backend',
    label: 'Backend',
    format: (s) => s.backend,
  },
];

export function mountHud(el: HTMLElement): { update: (stats: SimStats) => void } {
  el.replaceChildren();
  el.classList.add('hud');

  const values = new Map<string, HTMLElement>();

  for (const field of FIELDS) {
    const item = document.createElement('div');
    item.className = 'hud-item';
    item.dataset.stat = field.key;

    const label = document.createElement('span');
    label.className = 'hud-label';
    label.textContent = field.label;

    const value = document.createElement('span');
    value.className = 'hud-value';
    value.textContent = '—';
    value.setAttribute('aria-label', field.label);

    item.append(label, value);
    el.append(item);
    values.set(field.key, value);
  }

  function update(stats: SimStats): void {
    for (const field of FIELDS) {
      const valueEl = values.get(field.key);
      if (valueEl) valueEl.textContent = field.format(stats);
    }
  }

  return { update };
}
