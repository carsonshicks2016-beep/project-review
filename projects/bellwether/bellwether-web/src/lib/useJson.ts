"use client";

import { useEffect, useState } from "react";

/** Fetch a static JSON snapshot from /public/data. */
export function useJson<T>(path: string): T | null {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    let alive = true;
    fetch(path)
      .then((r) => r.json())
      .then((d) => alive && setData(d))
      .catch(() => alive && setData(null));
    return () => {
      alive = false;
    };
  }, [path]);
  return data;
}

export interface Signal {
  ticker: string;
  catalyst_type: string;
  direction: string;
  materiality: number;
  summary: string;
  evidence_quote: string;
  form_type?: string;
}

export interface Company {
  ticker: string;
  name: string;
  sector: string;
  mktcap: number | null;
  adv_usd: number | null;
  analyst_count: number | null;
  notes: string;
}
