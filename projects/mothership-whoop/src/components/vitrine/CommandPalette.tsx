'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

export type Command = {
  id: string;
  label: string;
  hint?: string;
  group: string;
  run: () => void;
};

/**
 * Cmd/Ctrl-K palette. Deliberately plain: this is chrome, so it stays bone on
 * black with no colour at all.
 */
export function CommandPalette({ commands }: { commands: Command[] }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((v) => !v);
        return;
      }
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    if (open) {
      setQuery('');
      setActive(0);
      // Focus after paint, or the input is not in the DOM yet.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter(
      (c) =>
        c.label.toLowerCase().includes(q) ||
        c.group.toLowerCase().includes(q) ||
        (c.hint ?? '').toLowerCase().includes(q),
    );
  }, [commands, query]);

  if (!open) return null;

  const choose = (c: Command | undefined) => {
    if (!c) return;
    setOpen(false);
    c.run();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/70 px-4 pt-[12vh] backdrop-blur-sm"
      onPointerDown={(e) => e.target === e.currentTarget && setOpen(false)}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        className="w-full max-w-lg border border-vit-rule bg-vit-bg font-vit-mono"
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
          }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault();
              setActive((i) => Math.min(i + 1, results.length - 1));
            } else if (e.key === 'ArrowUp') {
              e.preventDefault();
              setActive((i) => Math.max(i - 1, 0));
            } else if (e.key === 'Enter') {
              e.preventDefault();
              choose(results[active]);
            }
          }}
          placeholder="Search commands…"
          className="w-full border-b border-vit-rule bg-transparent px-4 py-3.5 text-[13px] tracking-[0.08em] text-vit-bone placeholder:text-vit-dim focus:outline-none"
        />

        <ul className="max-h-[52vh] overflow-y-auto py-1">
          {results.length === 0 && (
            <li className="px-4 py-6 text-center text-[10px] tracking-[0.18em] text-vit-dim uppercase">
              No matching command
            </li>
          )}
          {results.map((c, i) => (
            <li key={c.id}>
              <button
                type="button"
                onPointerEnter={() => setActive(i)}
                onClick={() => choose(c)}
                className={`flex w-full items-baseline gap-3 px-4 py-2.5 text-left text-[11px] tracking-[0.1em] uppercase ${
                  i === active ? 'bg-[#15151d] text-vit-bone' : 'text-vit-dim'
                }`}
              >
                <span className="w-16 shrink-0 text-[9px] text-vit-dim">{c.group}</span>
                <span className="min-w-0 flex-1 truncate">{c.label}</span>
                {c.hint && <span className="shrink-0 text-[9px] text-vit-dim">{c.hint}</span>}
              </button>
            </li>
          ))}
        </ul>

        <div className="flex gap-4 border-t border-vit-rule px-4 py-2 text-[9px] tracking-[0.14em] text-vit-dim uppercase">
          <span>↑↓ move</span>
          <span>⏎ run</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  );
}
