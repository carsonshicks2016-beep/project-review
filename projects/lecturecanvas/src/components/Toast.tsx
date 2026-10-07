import React, { useEffect, useRef } from 'react';

export interface ToastNotice {
  message: string;
  tone?: 'info' | 'success' | 'error';
  /** Optional single action, e.g. undoing a word exclusion. */
  action?: { label: string; onAction: () => void };
}

interface Props {
  notice: ToastNotice | null;
  onDismiss: () => void;
}

const TONE_STYLES: Record<NonNullable<ToastNotice['tone']>, string> = {
  info: 'border-cyan-500/40 text-cyan-200',
  success: 'border-emerald-500/40 text-emerald-200',
  error: 'border-rose-500/50 text-rose-200',
};

/** Errors stay up longer than confirmations, and anything with an action longer still. */
function dismissDelay(notice: ToastNotice): number {
  if (notice.action) return 7000;
  return notice.tone === 'error' ? 6000 : 3500;
}

export const Toast: React.FC<Props> = ({ notice, onDismiss }) => {
  const onDismissRef = useRef(onDismiss);
  useEffect(() => {
    onDismissRef.current = onDismiss;
  });

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => onDismissRef.current(), dismissDelay(notice));
    return () => window.clearTimeout(timer);
  }, [notice]);

  if (!notice) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className={`fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-4 py-2.5 rounded-xl bg-slate-900/95 border text-xs font-semibold shadow-2xl backdrop-blur-md ${
        TONE_STYLES[notice.tone ?? 'info']
      }`}
    >
      <span>{notice.message}</span>
      {notice.action && (
        <button
          type="button"
          onClick={() => {
            notice.action?.onAction();
            onDismiss();
          }}
          className="px-2 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
        >
          {notice.action.label}
        </button>
      )}
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss notification"
        className="text-slate-500 hover:text-white transition-colors"
      >
        ✕
      </button>
    </div>
  );
};
