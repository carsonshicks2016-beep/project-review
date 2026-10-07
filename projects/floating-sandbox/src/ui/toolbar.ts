import type { ToolId } from '../sim/types';

interface ToolDef {
  id: ToolId;
  label: string;
  shortcut: string;
}

const TOOLS: ToolDef[] = [
  { id: 'drag', label: 'Drag', shortcut: '1' },
  { id: 'smash', label: 'Smash', shortcut: '2' },
  { id: 'cut', label: 'Cut', shortcut: '3' },
  { id: 'bomb', label: 'Bomb', shortcut: '4' },
  { id: 'pin', label: 'Pin', shortcut: '5' },
  { id: 'repair', label: 'Repair', shortcut: '6' },
  { id: 'addWater', label: 'Add water', shortcut: '7' },
  { id: 'removeWater', label: 'Remove water', shortcut: '8' },
];

export function mountToolbar(
  el: HTMLElement,
  opts: {
    onTool: (id: ToolId) => void;
    getTool: () => ToolId;
  },
): { refresh: () => void } {
  el.replaceChildren();
  el.classList.add('toolbar');

  const buttons = new Map<ToolId, HTMLButtonElement>();

  for (const tool of TOOLS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'tool-btn';
    btn.dataset.tool = tool.id;
    btn.title = `${tool.label} (${tool.shortcut})`;
    btn.setAttribute('aria-label', `${tool.label}, shortcut ${tool.shortcut}`);
    btn.setAttribute('aria-pressed', 'false');

    const name = document.createElement('span');
    name.className = 'tool-btn-label';
    name.textContent = tool.label;

    const key = document.createElement('span');
    key.className = 'tool-btn-key';
    key.textContent = tool.shortcut;
    key.setAttribute('aria-hidden', 'true');

    btn.append(name, key);
    btn.addEventListener('click', () => {
      opts.onTool(tool.id);
      refresh();
    });

    el.append(btn);
    buttons.set(tool.id, btn);
  }

  function refresh(): void {
    const active = opts.getTool();
    for (const [id, btn] of buttons) {
      const on = id === active;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
  }

  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const target = e.target as HTMLElement | null;
    if (
      target &&
      (target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable)
    ) {
      return;
    }

    const tool = TOOLS.find((t) => t.shortcut === e.key);
    if (!tool) return;
    e.preventDefault();
    opts.onTool(tool.id);
    refresh();
  };

  window.addEventListener('keydown', onKeyDown);
  refresh();

  return { refresh };
}
