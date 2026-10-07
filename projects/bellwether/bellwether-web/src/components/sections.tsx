"use client";

import { motion } from "framer-motion";
import { useJson, Signal, Company } from "@/lib/useJson";
import { SignalCard } from "./SignalCard";
import { SectionTheme } from "@/lib/sections";

interface Overview {
  pond: number;
  filings: number;
  signals: number;
  sectors: number;
  avg_analysts: number | null;
}

function Stat({ label, value, i }: { label: string; value: string | number; i: number }) {
  return (
    <motion.div
      className="glass lift"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: i * 0.06, duration: 0.4 }}
      style={{ flex: 1, padding: "15px 17px", borderRadius: 16 }}
    >
      <div style={{ fontSize: 12, color: "var(--text-dim)" }}>{label}</div>
      <div style={{ fontSize: 25, fontWeight: 500, marginTop: 2 }}>{value}</div>
    </motion.div>
  );
}

export function OverviewSection() {
  const o = useJson<Overview>("/data/overview.json");
  const pond = useJson<Company[]>("/data/universe.json");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
        <Stat label="In universe" value={o?.pond ?? "—"} i={0} />
        <Stat label="Filings tracked" value={o?.filings ?? "—"} i={1} />
        <Stat label="Signals" value={o?.signals ? o.signals : "—"} i={2} />
        <Stat label="Avg analysts" value={o?.avg_analysts ?? "—"} i={3} />
      </div>
      <motion.div
        className="glass"
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.28, duration: 0.4 }}
        style={{ padding: "15px 17px" }}
      >
        <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 11 }}>
          Energy-weighted pond
        </div>
        <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
          {(pond ?? []).map((c) => (
            <span
              key={c.ticker}
              className="mono"
              style={{
                fontSize: 12,
                padding: "4px 9px",
                borderRadius: 8,
                background: "rgba(255,255,255,0.07)",
                border: "1px solid var(--glass-border)",
              }}
            >
              {c.ticker}
            </span>
          ))}
        </div>
      </motion.div>
    </div>
  );
}

export function SignalsSection() {
  const data = useJson<{ preview: boolean; items: Signal[] }>("/data/signals.json");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
      {data?.preview && (
        <div style={{ fontSize: 11, color: "var(--text-dim)" }}>
          preview · Phase 3 — illustrative until the extractor ships
        </div>
      )}
      {(data?.items ?? []).map((s, i) => (
        <SignalCard key={s.ticker + i} s={s} i={i} />
      ))}
    </div>
  );
}

export function UniverseSection() {
  const pond = useJson<Company[]>("/data/universe.json");
  return (
    <div className="glass" style={{ padding: "8px 18px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          color: "var(--text-dim)",
          padding: "8px 0",
          borderBottom: "1px solid var(--glass-border)",
        }}
      >
        <span>Ticker</span>
        <span>Sector · analysts · cap</span>
      </div>
      {(pond ?? []).map((c, i) => (
        <motion.div
          key={c.ticker}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.04, duration: 0.3 }}
          style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "9px 0", fontSize: 13 }}
        >
          <span className="mono" style={{ fontWeight: 500 }}>{c.ticker}</span>
          <span style={{ color: "var(--text-dim)" }}>
            {c.sector} · {c.analyst_count ?? "—"} an ·{" "}
            {c.mktcap ? `$${(c.mktcap / 1e9).toFixed(2)}B` : "—"}
          </span>
        </motion.div>
      ))}
    </div>
  );
}

interface AuditData {
  state: string | null;
  repo_url: string | null;
  generated_at: string;
  captures: { ticker: string; form_type: string; acceptance_datetime: string; latency_min: number | null }[];
  runs: { conclusion: string | null; status: string; createdAt: string; event: string; displayTitle?: string }[];
}

const fmt = (s: string) => {
  try { return new Date(s).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
  catch { return s; }
};

export function AuditSection({ theme }: { theme: SectionTheme }) {
  const a = useJson<AuditData>("/data/audit.json");
  const lastRun = a?.runs?.[0];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
        <Stat label="Cloud audit" value={a?.state ?? "—"} i={0} />
        <Stat label="Last run" value={lastRun ? (lastRun.conclusion ?? lastRun.status) : "—"} i={1} />
        <Stat label="Captured" value={a?.captures?.length ?? "—"} i={2} />
        <Stat label="Recent runs" value={a?.runs?.length ?? "—"} i={3} />
      </div>

      <motion.div className="glass" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.24, duration: 0.4 }} style={{ padding: "15px 17px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 11 }}>
          <span style={{ fontSize: 12, color: "var(--text-dim)" }}>Filings captured during audit</span>
          {a?.repo_url && (
            <a href={`${a.repo_url}/actions`} target="_blank" rel="noreferrer"
              style={{ fontSize: 12, color: theme.accent, textDecoration: "none" }}>
              live runs on GitHub ↗
            </a>
          )}
        </div>
        {a && a.captures.length === 0 && (
          <div style={{ fontSize: 12, color: "var(--text-dim)" }}>None yet — quiet names. The cloud poll keeps watching.</div>
        )}
        {(a?.captures ?? []).map((c, i) => (
          <motion.div key={c.acceptance_datetime + i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "9px 0", fontSize: 13, borderTop: i ? "1px solid var(--glass-border)" : "none" }}>
            <span><span className="mono" style={{ fontWeight: 500 }}>{c.ticker}</span>
              <span style={{ color: "var(--text-dim)", marginLeft: 8 }}>{c.form_type}</span></span>
            <span style={{ color: "var(--text-dim)" }}>
              filed {fmt(c.acceptance_datetime)}
              {c.latency_min != null && <span className="mono" style={{ marginLeft: 10, color: c.latency_min < 15 ? "#34d399" : "#fbbf24" }}>+{c.latency_min}m</span>}
            </span>
          </motion.div>
        ))}
      </motion.div>

      <motion.div className="glass" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.32, duration: 0.4 }} style={{ padding: "15px 17px" }}>
        <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 10 }}>Recent cloud runs</div>
        {(a?.runs ?? []).map((r, i) => (
          <div key={r.createdAt + i} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", fontSize: 13 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: r.conclusion === "success" ? "#34d399" : "#fbbf24" }} />
              {r.conclusion ?? r.status}
            </span>
            <span style={{ color: "var(--text-dim)" }}>
              {r.displayTitle?.match(/\(([^)]+)\)$/)?.[1] ?? r.event} · {fmt(r.createdAt)}
            </span>
          </div>
        ))}
        {a && (
          <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 8, fontStyle: "italic" }}>
            snapshot · generated {fmt(a.generated_at)} — controls live in the local cockpit
          </div>
        )}
      </motion.div>
    </div>
  );
}

export function BacktestSection({ theme }: { theme: SectionTheme }) {
  return (
    <motion.div
      className="glass"
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      style={{ padding: "34px 20px", textAlign: "center" }}
    >
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke={theme.accent}
        strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden
        style={{ margin: "0 auto 10px", display: "block" }}>
        <path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3" />
      </svg>
      <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 5 }}>
        The make-or-break gate
      </div>
      <div style={{ fontSize: 12, color: "var(--text-dim)", lineHeight: 1.6, maxWidth: 400, margin: "0 auto" }}>
        Did high-materiality flags precede real abnormal returns? Defeating
        look-ahead, survivorship, and slippage. Lights up in Phase 4.
      </div>
    </motion.div>
  );
}
