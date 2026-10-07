import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, ArrowDownToLine, ArrowLeft, ArrowRight, BarChart3, CalendarClock, Check, ChevronDown, ChevronRight, CircleHelp, Database, Download, HeartPulse, Info, Link2, LoaderCircle, Moon, Plus, RefreshCw, Search, Shield, Sparkles, Trash2, Unplug, X } from 'lucide-react';
import { fieldDefinition, fieldUnit, flattenValues } from '../server/analytics';

type Connection = { connected: boolean; configured: boolean; tokenExpired: boolean; lastSync: string | null; counts: Array<{ category: string; count: number; last_synced: string | null }>; redirectUri: string; setupHint: string | null };
type Daily = { date: string; recovery: number | null; hrv: number | null; restingHr: number | null; sleepPerformance: number | null; strain: number | null; sleepMinutes: number | null };
type Insight = { latest: Daily | null; baseline: number | null; baselineSamples: number; shift: null | { date: string; value: number; baseline: number; delta: number; confidence: string }; note: string };
type RecordItem = { category: string; record_id: string; start_at: string | null; end_at: string | null; updated_at: string | null; timezone_offset: string | null; raw_json: string; synced_at: string; raw: Record<string, unknown> };
type EventItem = { id: string; label: string; category: string; start_at: string; end_at: string | null; notes: string; nearestDays?: Daily[]; date?: string; recoveryResponse?: { status: string; days: number | null; baseline: number | null; baselineSamples: number; returnDate: string | null } };
type Experiment = { id: string; name: string; hypothesis: string; created_at: string; entries: Array<{ id: string; date: string; condition: string; notes: string }> };

const fmtDate = (value?: string | null, options: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric', year: 'numeric' }) => value ? new Intl.DateTimeFormat(undefined, options).format(new Date(value)) : '—';
const fmtTime = (value?: string | null) => value ? new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(new Date(value)) : '—';
const titleCase = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, ch => ch.toUpperCase());
const today = new Date().toISOString().slice(0, 10);
function experimentComparison(experiment: Experiment, daily: Daily[]) {
  const byDate = new Map(daily.map(point => [point.date, point]));
  const result = (condition: string, metric: 'recovery' | 'hrv') => {
    const values = experiment.entries.filter(entry => entry.condition === condition).map(entry => byDate.get(entry.date)?.[metric]).filter((value): value is number => typeof value === 'number');
    return { count: values.length, mean: values.length ? values.reduce((sum,value)=>sum+value,0)/values.length : null };
  };
  return { yes: result('Yes','recovery'), no: result('No','recovery') };
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...init, headers: { ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers } });
  const type = response.headers.get('content-type') || '';
  const result = type.includes('json') ? await response.json() : null;
  if (!response.ok) throw new Error(result?.error || `Request failed (${response.status}).`);
  return result as T;
}

function LineChart({ values, valueKey, label, color = 'var(--lime)', unit = '', height = 190, baseline }: { values: Daily[]; valueKey: keyof Daily; label: string; color?: string; unit?: string; height?: number; baseline?: number | null }) {
  const points = values.map((point, index) => ({ x: values.length < 2 ? 50 : 8 + index * 84 / (values.length - 1), y: point[valueKey] as number | null, date: point.date })).filter(point => point.y !== null);
  if (points.length < 2) return <div className="chart-empty">A few more days of data will fill in this trend.</div>;
  const low = Math.min(...points.map(point => point.y!)); const high = Math.max(...points.map(point => point.y!)); const pad = Math.max((high - low) * .22, 2);
  const min = low - pad; const max = high + pad; const yAt = (n: number) => 91 - (n - min) / (max - min || 1) * 82;
  const d = points.map((point, i) => `${i ? 'L' : 'M'} ${point.x} ${yAt(point.y!)}`).join(' ');
  const fill = `${d} L ${points.at(-1)!.x} 100 L ${points[0].x} 100 Z`;
  return <div className="chart-wrap">
    <div className="chart-top"><span>{label}</span><strong>{points.at(-1)!.y}{unit}</strong></div>
    <svg viewBox="0 0 100 104" preserveAspectRatio="none" style={{ height }} aria-label={`${label} trend chart`} role="img">
      {[18, 42, 66, 90].map(y => <line key={y} x1="0" x2="100" y1={y} y2={y} className="chart-grid" />)}
      {baseline !== undefined && baseline !== null && <line x1="0" x2="100" y1={yAt(baseline)} y2={yAt(baseline)} className="chart-base" />}
      <path d={fill} fill={color} opacity=".10" />
      <path d={d} fill="none" stroke={color} strokeWidth="1.8" vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
      {points.map((point, i) => i === points.length - 1 || i === 0 ? <circle key={point.date} cx={point.x} cy={yAt(point.y!)} r="1.6" fill={color} /> : null)}
    </svg>
    <div className="chart-foot"><span>{fmtDate(points[0].date, { month: 'short', day: 'numeric' })}</span><span>{points.length} observations</span><span>{fmtDate(points.at(-1)?.date, { month: 'short', day: 'numeric' })}</span></div>
  </div>;
}

function App() {
  const [page, setPage] = useState<'primary' | 'data'>('primary');
  const [connection, setConnection] = useState<Connection | null>(null);
  const [summary, setSummary] = useState<{ daily: Daily[]; insights: Insight; events: EventItem[]; experiments: Experiment[]; totalRecords: number } | null>(null);
  const [records, setRecords] = useState<{ records: RecordItem[]; total: number; limit: number; offset: number } | null>(null);
  const [typeFilter, setTypeFilter] = useState(''); const [search, setSearch] = useState(''); const [startFilter, setStartFilter] = useState(''); const [endFilter, setEndFilter] = useState(''); const [offset, setOffset] = useState(0);
  const [fieldOffset, setFieldOffset] = useState(0); const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [loading, setLoading] = useState(true); const [syncing, setSyncing] = useState(false); const [error, setError] = useState(''); const [notice, setNotice] = useState('');
  const [eventForm, setEventForm] = useState(false); const [experimentForm, setExperimentForm] = useState(false); const [eventBusy, setEventBusy] = useState(false); const [expanded, setExpanded] = useState('');
  const [entryFor, setEntryFor] = useState('');

  const loadConnection = useCallback(async () => setConnection(await request<Connection>('/api/connection')), []);
  const loadSummary = useCallback(async () => setSummary(await request('/api/summary')), []);
  const loadRecords = useCallback(async () => {
    const query = new URLSearchParams({ limit: '100', offset: String(offset) });
    if (typeFilter) query.set('type', typeFilter); if (search) query.set('search', search);
    if (startFilter) query.set('start', new Date(`${startFilter}T00:00:00`).toISOString());
    if (endFilter) query.set('end', new Date(`${endFilter}T23:59:59`).toISOString());
    setRecords(await request(`/api/records?${query}`));
  }, [offset, typeFilter, search, startFilter, endFilter]);
  const reload = useCallback(async () => {
    setLoading(true); setError('');
    try { await Promise.all([loadConnection(), loadSummary(), loadRecords()]); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not load local data.'); }
    finally { setLoading(false); }
  }, [loadConnection, loadSummary, loadRecords]);

  useEffect(() => { void reload(); }, [reload]);
  useEffect(() => { const timer = window.setTimeout(() => { if (page === 'data') void loadRecords().catch(err => setError(err.message)); }, 180); return () => window.clearTimeout(timer); }, [page, loadRecords]);

  const sync = async () => {
    setSyncing(true); setError(''); setNotice('');
    try {
      const result = await request<{ synced: Record<string, number> }>('/api/sync', { method: 'POST' });
      await reload();
      const count = Object.values(result.synced).reduce((a, b) => a + b, 0);
      setNotice(`Sync finished · ${count.toLocaleString()} records checked`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Sync did not finish.'); }
    finally { setSyncing(false); }
  };
  const connect = async () => {
    setError('');
    try { const result = await request<{ url: string }>('/api/auth/url'); window.location.assign(result.url); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not start WHOOP authorization.'); }
  };
  const disconnect = async () => {
    try { await request('/api/disconnect', { method: 'POST' }); await reload(); setNotice('WHOOP disconnected. Synced data is still saved on this Mac.'); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not disconnect.'); }
  };

  const saveEvent = async (form: HTMLFormElement) => {
    const values = Object.fromEntries(new FormData(form).entries()); setEventBusy(true);
    try { await request('/api/events', { method: 'POST', body: JSON.stringify({ label: values.label, category: values.category, startAt: new Date(String(values.startAt)).toISOString(), endAt: values.endAt ? new Date(String(values.endAt)).toISOString() : undefined, notes: values.notes }) }); setEventForm(false); await reload(); setNotice('Event note saved.'); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not save event.'); }
    finally { setEventBusy(false); }
  };
  const saveExperiment = async (form: HTMLFormElement) => {
    const values = Object.fromEntries(new FormData(form).entries()); setEventBusy(true);
    try { await request('/api/experiments', { method: 'POST', body: JSON.stringify({ name: values.name, hypothesis: values.hypothesis }) }); setExperimentForm(false); await reload(); setNotice('Experiment created.'); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not create experiment.'); }
    finally { setEventBusy(false); }
  };
  const saveEntry = async (experimentId: string, form: HTMLFormElement) => {
    const values = Object.fromEntries(new FormData(form).entries()); setEventBusy(true);
    try { await request(`/api/experiments/${experimentId}/entries`, { method: 'POST', body: JSON.stringify({ date: values.date, condition: values.condition, notes: values.notes }) }); setEntryFor(''); await reload(); setNotice('Experiment entry saved.'); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not save experiment entry.'); }
    finally { setEventBusy(false); }
  };
  const removeEvent = async (id: string) => { await request(`/api/events/${id}`, { method: 'DELETE' }); await reload(); };
  const removeExperiment = async (id: string) => { await request(`/api/experiments/${id}`, { method: 'DELETE' }); await reload(); };

  const daily = summary?.daily ?? [];
  const latest = summary?.insights?.latest;
  const rowCount = records?.total ?? 0;
  const fieldRows = useMemo(() => (records?.records ?? []).flatMap(record => flattenValues(record.raw).map(field => ({ ...field, record }))), [records]);
  const visibleFieldRows = fieldRows.slice(fieldOffset, fieldOffset + 200);
  const metricOptions = useMemo(() => {
    const options = new Map<string, { key: string; label: string; category: string; path: string; count: number }>();
    for (const record of records?.records ?? []) for (const field of flattenValues(record.raw)) {
      if (typeof field.value !== 'number' || !field.path.startsWith('score.')) continue;
      const key = `${record.category}|${field.path}`;
      const item = options.get(key) ?? { key, label: `${titleCase(record.category)} · ${field.path}`, category: record.category, path: field.path, count: 0 };
      item.count += 1; options.set(key, item);
    }
    return [...options.values()].filter(item => item.count > 1).sort((a,b) => a.category.localeCompare(b.category) || a.path.localeCompare(b.path)).slice(0,24);
  }, [records]);
  const overlaySeries = useMemo(() => selectedMetrics.map(key => {
    const option = metricOptions.find(item => item.key === key);
    if (!option) return null;
    const values = (records?.records ?? []).filter(record => record.category === option.category).map(record => {
      const value = flattenValues(record.raw).find(field => field.path === option.path)?.value;
      return { record, value: typeof value === 'number' ? value : null };
    }).filter(point => point.value !== null).reverse();
    if (!values.length) return null;
    const nums = values.map(point => point.value!); const min = Math.min(...nums); const max = Math.max(...nums);
    return { ...option, values: values.map(point => ({ date: point.record.start_at, value: point.value!, index: max === min ? 50 : Math.round((point.value! - min) / (max - min) * 100) })) };
  }).filter(Boolean) as Array<{ key:string; label:string; category:string; path:string; values:Array<{date:string|null;value:number;index:number}> }>, [selectedMetrics, metricOptions, records]);
  const exportParams = new URLSearchParams({ format: 'csv' }); if (typeFilter) exportParams.set('type', typeFilter); if (startFilter) exportParams.set('start', new Date(`${startFilter}T00:00:00`).toISOString()); if (endFilter) exportParams.set('end', new Date(`${endFilter}T23:59:59`).toISOString());
  const exportJsonParams = new URLSearchParams(exportParams); exportJsonParams.set('format', 'json');
  const categoryOptions = useMemo(() => connection?.counts ?? [], [connection]);
  const complete = connection?.connected ?? false;

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Activity size={19} strokeWidth={2.4} /></div><div><strong>recovery lab</strong><span>PERSONAL DATA STUDIO</span></div></div>
      <div className="side-label">WORKSPACE</div>
      <button className={`nav-link ${page === 'primary' ? 'active' : ''}`} onClick={() => setPage('primary')}><Sparkles size={17} /><span>Primary</span><span className="nav-hint">01</span></button>
      <button className={`nav-link ${page === 'data' ? 'active' : ''}`} onClick={() => setPage('data')}><Database size={17} /><span>Max data</span><span className="nav-hint">02</span></button>
      <div className="side-bottom">
        <div className="privacy-card"><Shield size={17} /><div><strong>Private by design</strong><p>Runs only on this Mac. Your health data stays in local storage.</p></div></div>
        <div className="whoop-state"><span className={`status-dot ${complete ? 'online' : ''}`} /><span>{complete ? 'WHOOP connected' : 'WHOOP not connected'}</span><button aria-label="Connection details" onClick={() => setPage('primary')}><CircleHelp size={15} /></button></div>
        <div className="side-foot">PERSONAL RECOVERY LAB <span>v0.1</span></div>
      </div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="breadcrumbs"><span>MY LAB</span><ChevronRight size={13} /> <strong>{page === 'primary' ? 'PRIMARY' : 'MAX DATA'}</strong></div>
        <div className="top-actions"><div className="sync-stamp"><span className={`status-dot ${complete ? 'online' : ''}`} />{connection?.lastSync ? `SYNCED ${fmtDate(connection.lastSync, { month: 'short', day: 'numeric' })}` : 'LOCAL DATA'}</div>
          {complete ? <button className="button button-subtle" onClick={disconnect}><Unplug size={15} /> Disconnect</button> : <button className="button button-outline" onClick={connect} disabled={!connection?.configured}><Link2 size={15} /> Connect WHOOP</button>}
          <button className="button button-primary" onClick={() => void sync()} disabled={!complete || syncing}><RefreshCw size={15} className={syncing ? 'spin' : ''} />{syncing ? 'Syncing' : 'Sync data'}</button>
        </div>
      </header>

      {(error || notice) && <div className={`toast ${error ? 'toast-error' : ''}`}><span>{error || notice}</span><button onClick={() => { setError(''); setNotice(''); }}><X size={15} /></button></div>}
      {connection?.setupHint && <div className="setup-banner"><Info size={17} /><div><strong>WHOOP OAuth setup is incomplete</strong><p>{connection.setupHint}</p></div></div>}
      {!connection?.connected && connection?.configured && <div className="setup-banner"><Info size={17} /><div><strong>Finish connecting your WHOOP account</strong><p>Authorize access to cycles, recovery, sleep, workouts, and body measurements. Data will remain on this Mac.</p></div><button className="button button-outline" onClick={connect}>Connect account <ArrowRight size={14} /></button></div>}

      <div className="page-content">
        {page === 'primary' ? <>
          <div className="page-intro"><div><div className="eyebrow"><span className="eyebrow-mark" /> PERSONAL RECOVERY LAB <span className="eyebrow-date">{fmtDate(today)}</span></div><h1>How are you <em>adjusting?</em></h1><p>Look for patterns in the signals your body leaves behind.</p></div><div className="intro-note"><div className="note-icon"><Info size={15} /></div><span>Wearable signals can show changes, but not explain their cause.</span></div></div>
          {loading ? <Loading /> : !summary?.totalRecords ? <EmptyState onConnect={connect} configured={Boolean(connection?.configured)} onSync={sync} /> : <>
            <section className="metric-grid">
              <MetricCard label="LATEST RECOVERY" icon={<HeartPulse size={16} />} value={latest?.recovery === null || latest?.recovery === undefined ? '—' : `${latest.recovery}%`} sub={latest ? `Cycle · ${fmtDate(latest.date)}` : 'No scored recovery yet'} accent="lime" />
              <MetricCard label="PERSONAL BASELINE" icon={<BarChart3 size={16} />} value={summary.insights.baseline === null ? 'Building' : `${summary.insights.baseline}%`} sub={summary.insights.baselineSamples >= 5 ? `${summary.insights.baselineSamples} recent observations` : 'Need 5+ scored days'} accent="blue" />
              <MetricCard label="HRV · LAST RECORDED" icon={<Activity size={16} />} value={latest?.hrv === null || latest?.hrv === undefined ? '—' : `${latest.hrv.toFixed(1)} ms`} sub={latest?.date ? `From ${fmtDate(latest.date)}` : 'No recovery data'} accent="purple" />
              <MetricCard label="SYNCED RECORDS" icon={<Database size={16} />} value={(summary.totalRecords || 0).toLocaleString()} sub="WHOOP records stored locally" accent="orange" />
            </section>

            <div className="primary-grid">
              <section className="panel trend-panel"><div className="panel-heading"><div><span className="section-kicker">YOUR SIGNALS</span><h2>Recovery over time</h2></div><span className="panel-range">LAST 180 DAYS</span></div>
                {summary.insights.shift ? <div className={`signal-callout ${summary.insights.shift.delta < 0 ? 'down' : 'up'}`}><div className="signal-icon"><Activity size={16} /></div><div><strong>Possible shift on {fmtDate(summary.insights.shift.date)}</strong><span>Recovery was {Math.abs(summary.insights.shift.delta)} points {summary.insights.shift.delta < 0 ? 'below' : 'above'} your recent median. {summary.insights.shift.confidence} confidence.</span></div><span className="signal-value">{summary.insights.shift.delta > 0 ? '+' : ''}{summary.insights.shift.delta}</span></div> : <p className="baseline-note">{summary.insights.note}</p>}
                <LineChart values={daily.slice(-35)} valueKey="recovery" label="Recovery score" unit="%" baseline={summary.insights.baseline} />
                <div className="chart-duo"><LineChart values={daily.slice(-35)} valueKey="hrv" label="HRV · RMSSD" color="var(--blue)" unit=" ms" height={152} /><LineChart values={daily.slice(-35)} valueKey="restingHr" label="Resting heart rate" color="var(--purple)" unit=" bpm" height={152} /></div>
                    <div className="data-limit"><Info size={14} /><span>Day-level WHOOP summaries can suggest a change; they can’t pinpoint a stressful moment.</span></div>
              </section>
              <section className="panel events-panel"><div className="panel-heading"><div><span className="section-kicker">CONTEXT MATTERS</span><h2>Event journal</h2></div><button className="icon-button add-button" onClick={() => setEventForm(true)} aria-label="Log an event"><Plus size={18} /></button></div>
                <p className="panel-subtitle">Log moments that might help explain a change. Your notes remain separate from wearable measurements.</p>
                {!summary.events.length ? <div className="empty-inset"><div className="inset-icon"><CalendarClock size={17} /></div><div><strong>No events logged yet</strong><span>Add a stressful period, travel day, or other context to compare against your trends.</span></div><button onClick={() => setEventForm(true)}>Log an event <ArrowRight size={14} /></button></div> : <div className="event-list">{summary.events.slice(0, 6).map(event => <div className="event-row" key={event.id}><div className="event-marker" /><div className="event-copy"><div className="event-title"><strong>{event.label}</strong><span className="event-tag">{titleCase(event.category)}</span></div><span>{fmtTime(event.start_at)}{event.end_at ? ` – ${fmtTime(event.end_at)}` : ''}</span>{event.nearestDays?.length ? <small>{event.nearestDays.length} nearby daily observation{event.nearestDays.length > 1 ? 's' : ''} · association only</small> : null}{event.recoveryResponse?.status === 'observed' ? <small>Recovery returned near baseline after {event.recoveryResponse.days} day{event.recoveryResponse.days === 1 ? '' : 's'} · descriptive pattern</small> : event.recoveryResponse?.status === 'insufficient_baseline' ? <small>Need 5 scored pre-event days to estimate a recovery window.</small> : event.recoveryResponse?.status === 'waiting' ? <small>Still observing the recovery window after this event.</small> : event.recoveryResponse?.status === 'not_observed' ? <small>Return to the recent baseline band was not observed within 21 days.</small> : null}</div><button className="row-delete" title="Remove event" onClick={() => void removeEvent(event.id)}><Trash2 size={14} /></button></div>)}</div>}
                <div className="panel-divider" /><div className="panel-heading compact-heading"><div><span className="section-kicker">SMALL EXPERIMENTS</span><h2>Test a change</h2></div><button className="text-button" onClick={() => setExperimentForm(true)}><Plus size={15} /> New test</button></div>
                {!summary.experiments.length ? <div className="experiment-empty">Try one change at a time and log days you did or didn’t follow it.</div> : <div className="experiment-list">{summary.experiments.map(experiment => <div className="experiment-item" key={experiment.id}><button className="experiment-toggle" onClick={() => setExpanded(expanded === experiment.id ? '' : experiment.id)}><span><strong>{experiment.name}</strong><small>{experiment.entries.length} logged day{experiment.entries.length === 1 ? '' : 's'}</small></span>{expanded === experiment.id ? <ChevronDown size={16} /> : <ChevronRight size={16} />}</button><button className="row-delete" title="Delete experiment" onClick={() => void removeExperiment(experiment.id)}><Trash2 size={14} /></button>{expanded === experiment.id && <div className="experiment-details">{experiment.hypothesis && <p>{experiment.hypothesis}</p>}{(() => { const comparison = experimentComparison(experiment, daily); return comparison.yes.count >= 3 && comparison.no.count >= 3 ? <div className="comparison-note">Recovery average · Yes {comparison.yes.mean!.toFixed(0)}% (n={comparison.yes.count}) · No {comparison.no.mean!.toFixed(0)}% (n={comparison.no.count}) · descriptive, not causal</div> : <div className="comparison-note">Add at least 3 scored Yes days and 3 scored No days to compare recovery descriptively.</div>; })()}<div className="entry-list">{experiment.entries.slice(0, 8).map(entry => <span key={entry.id}>{fmtDate(entry.date)} <b>{entry.condition}</b></span>)}</div>{entryFor === experiment.id ? <form className="entry-form" onSubmit={e => { e.preventDefault(); void saveEntry(experiment.id, e.currentTarget); }}><input name="date" type="date" defaultValue={today} required /><select name="condition"><option>Yes</option><option>No</option><option>Partial</option></select><input name="notes" placeholder="Optional note" /><button className="button button-primary" disabled={eventBusy}>Save day</button></form> : <button className="text-button" onClick={() => setEntryFor(experiment.id)}><Plus size={14} /> Log a day</button>}</div>}</div>)}</div>}
              </section>
            </div>
            <div className="bottom-context"><div className="context-item"><span className="context-icon sleep"><Moon size={16} /></span><div><small>RECENT SLEEP</small><strong>{latest?.sleepPerformance === null || latest?.sleepPerformance === undefined ? '—' : `${latest.sleepPerformance}% performance`}</strong></div></div><div className="context-item"><span className="context-icon strain"><Activity size={16} /></span><div><small>DAILY STRAIN</small><strong>{latest?.strain === null || latest?.strain === undefined ? '—' : latest.strain.toFixed(1)}</strong></div></div><div className="context-copy"><Info size={15} /><span>These views are exploratory, not medical guidance. Missing WHOOP data stays missing.</span></div></div>
          </>}
        </> : <>
          <div className="page-intro data-intro"><div><div className="eyebrow"><span className="eyebrow-mark blue-mark" /> COMPLETE DATA CATALOG</div><h1>Max <em>data.</em></h1><p>Every field and record the connected WHOOP API returns, preserved as received.</p></div><div className="export-group"><a className="button button-outline" href={`/api/export?${exportParams}`}><ArrowDownToLine size={15} /> Export CSV</a><a className="button button-outline" href={`/api/export?${exportJsonParams}`}><Download size={15} /> Export JSON</a></div></div>
          <div className="scope-banner"><Info size={16} /><span>“Everything” means fields exposed through WHOOP’s authorized API. Raw sensor streams and hidden app-only data aren’t available through this connection.</span><span className="scope-count">{summary?.totalRecords?.toLocaleString() ?? '—'} RECORDS</span></div>
          <section className="panel explorer-panel"><div className="explorer-toolbar"><div className="search-box"><Search size={16} /><input value={search} onChange={e => { setSearch(e.target.value); setOffset(0); setFieldOffset(0); }} placeholder="Search fields or values..." /></div><select className="filter-select" value={typeFilter} onChange={e => { setTypeFilter(e.target.value); setOffset(0); setFieldOffset(0); }}><option value="">All data types</option>{categoryOptions.map(item => <option key={item.category} value={item.category}>{titleCase(item.category)} ({item.count})</option>)}</select><label className="date-filter"><span>FROM</span><input type="date" value={startFilter} onChange={e => { setStartFilter(e.target.value); setOffset(0); setFieldOffset(0); }} /></label><label className="date-filter"><span>TO</span><input type="date" value={endFilter} onChange={e => { setEndFilter(e.target.value); setOffset(0); setFieldOffset(0); }} /></label><button className="icon-button refresh-small" onClick={() => void loadRecords()} title="Refresh results"><RefreshCw size={15} /></button></div>
            {!complete && <div className="explorer-empty"><Database size={28} /><strong>Connect WHOOP to populate the data catalog</strong><span>The catalog will include all returned raw fields and record timestamps.</span></div>}
            {complete && !rowCount ? <div className="explorer-empty"><Database size={28} /><strong>No records match these filters</strong><span>Adjust the filters or sync your account to load records.</span></div> : complete && <>
              <div className="field-summary"><div><span className="section-kicker">FIELD LEVEL VIEW</span><strong>{fieldRows.length.toLocaleString()} values in these records · {Math.floor(fieldOffset / 200) + 1} of {Math.max(1, Math.ceil(fieldRows.length / 200))} field pages</strong></div><span>Records {offset + 1}–{Math.min(offset + (records?.records.length ?? 0), rowCount)} of {rowCount.toLocaleString()}</span></div>
              <div className="overlay-box"><div className="overlay-head"><div><span className="section-kicker">METRIC OVERLAYS</span><strong>Compare numeric signals</strong></div><span>Pick up to 4 · indexed 0–100 across the current records</span></div><div className="metric-chips">{metricOptions.map(option => <button key={option.key} className={`metric-chip ${selectedMetrics.includes(option.key) ? 'selected' : ''}`} onClick={() => setSelectedMetrics(current => current.includes(option.key) ? current.filter(key => key !== option.key) : current.length < 4 ? [...current,option.key] : current)}>{selectedMetrics.includes(option.key) ? <Check size={12} /> : <Plus size={12} />}{option.label}</button>)}</div>
                {overlaySeries.length > 0 && <OverlayChart series={overlaySeries} />}
              </div>
              <div className="table-scroll"><table className="data-table"><thead><tr><th>DATA TYPE</th><th>TIMESTAMP</th><th>FIELD PATH</th><th>VALUE</th><th>UNIT</th><th>RECORD ID</th><th></th></tr></thead><tbody>
                {visibleFieldRows.map((field, index) => <tr key={`${field.record.category}-${field.record.record_id}-${field.path}-${index}`}><td><span className={`type-pill ${field.record.category}`}>{titleCase(field.record.category)}</span></td><td className="mono-cell">{fmtTime(field.record.start_at || field.record.updated_at || field.record.synced_at)}</td><td className="field-path" title={fieldDefinition(field.path)}>{field.path}</td><td className="value-cell">{typeof field.value === 'object' ? JSON.stringify(field.value) : String(field.value)}</td><td className="unit-cell">{fieldUnit(field.path)}</td><td className="id-cell">{field.record.record_id}</td><td><button className="expand-row" aria-label="Inspect original record" onClick={() => setExpanded(expanded === `${field.record.category}:${field.record.record_id}` ? '' : `${field.record.category}:${field.record.record_id}`)}><ChevronDown size={15} /></button></td></tr>)}
              </tbody></table></div>
              {records?.records.map(record => expanded === `${record.category}:${record.record_id}` && <div className="raw-drawer" key={`raw-${record.category}-${record.record_id}`}><div className="drawer-head"><span>ORIGINAL PROVIDER RECORD · {record.category} · {record.record_id}</span><button onClick={() => setExpanded('')}><X size={15} /></button></div><pre>{JSON.stringify(record.raw, null, 2)}</pre></div>)}
              <div className="table-pagination"><span>Fields {fieldOffset + 1}–{Math.min(fieldOffset + visibleFieldRows.length, fieldRows.length)} of {fieldRows.length} · record page {Math.floor(offset / 100) + 1}</span><div><button className="button button-subtle" disabled={fieldOffset <= 0} onClick={() => setFieldOffset(Math.max(0, fieldOffset - 200))}><ArrowLeft size={14} /> Fields</button><button className="button button-subtle" disabled={fieldOffset + 200 >= fieldRows.length} onClick={() => setFieldOffset(fieldOffset + 200)}>Fields <ArrowRight size={14} /></button><button className="button button-subtle" disabled={offset + 100 >= rowCount} onClick={() => { setOffset(offset + 100); setFieldOffset(0); }}>More records <ArrowRight size={14} /></button></div></div>
            </>}
          </section>
          <div className="catalog-foot"><Shield size={14} /><span>Raw records are retained locally. Exports contain health data; store them carefully.</span><span className="catalog-updated">Last sync {fmtDate(connection?.lastSync)}</span></div>
        </>}
      </div>
      <footer className="app-footer"><span>RECOVERY LAB <i>·</i> BUILT FOR CURIOSITY</span><span><Shield size={13} /> LOCAL ONLY <i>·</i> DATA IS NOT MEDICAL ADVICE</span></footer>
    </main>

    {eventForm && <Modal title="Log an event" subtitle="Add context to compare with your own physiological trends." onClose={() => setEventForm(false)}><form className="modal-form" onSubmit={e => { e.preventDefault(); void saveEvent(e.currentTarget); }}><label>What happened?<input name="label" placeholder="e.g. Difficult work presentation" maxLength={160} required autoFocus /></label><label>Category<select name="category"><option value="stress">Stressful period</option><option value="travel">Travel</option><option value="illness">Felt unwell</option><option value="personal">Personal context</option><option value="other">Other</option></select></label><div className="form-row"><label>Started<input name="startAt" type="datetime-local" defaultValue={`${today}T12:00`} required /></label><label>Ended (optional)<input name="endAt" type="datetime-local" /></label></div><label>Note (optional)<textarea name="notes" rows={3} placeholder="What would be useful to remember?" maxLength={1000} /></label><div className="modal-actions"><button type="button" className="button button-subtle" onClick={() => setEventForm(false)}>Cancel</button><button className="button button-primary" disabled={eventBusy}>{eventBusy ? 'Saving…' : 'Save event'}</button></div></form></Modal>}
    {experimentForm && <Modal title="Start a personal experiment" subtitle="Track one small change over time. This records your inputs; it won’t claim causation." onClose={() => setExperimentForm(false)}><form className="modal-form" onSubmit={e => { e.preventDefault(); void saveExperiment(e.currentTarget); }}><label>Experiment name<input name="name" placeholder="e.g. No caffeine after 12 pm" maxLength={160} required autoFocus /></label><label>What do you expect? (optional)<textarea name="hypothesis" rows={3} placeholder="A prediction to compare with the data later" maxLength={600} /></label><div className="modal-actions"><button type="button" className="button button-subtle" onClick={() => setExperimentForm(false)}>Cancel</button><button className="button button-primary" disabled={eventBusy}>{eventBusy ? 'Saving…' : 'Create experiment'}</button></div></form></Modal>}
  </div>;
}

function MetricCard({ label, icon, value, sub, accent }: { label: string; icon: React.ReactNode; value: string; sub: string; accent: string }) { return <div className="metric-card"><div className="metric-heading"><span>{label}</span><span className={`metric-icon ${accent}`}>{icon}</span></div><strong className="metric-value">{value}</strong><span className="metric-sub">{sub}</span></div>; }
function OverlayChart({ series }: { series: Array<{ key:string; label:string; category:string; path:string; values:Array<{date:string|null;value:number;index:number}> }> }) {
  const colors = ['#c8f36c','#81b8fa','#c3a0ff','#ffad75'];
  const dates = series.flatMap(item => item.values.map(point => point.date ? Date.parse(point.date) : NaN)).filter(Number.isFinite);
  const minTime = Math.min(...dates); const maxTime = Math.max(...dates);
  const xAt = (value: string | null) => { const time = value ? Date.parse(value) : NaN; return !Number.isFinite(time) || minTime === maxTime ? 50 : 4 + (time - minTime) / (maxTime - minTime) * 92; };
  return <div className="overlay-chart"><div><svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Selected WHOOP metrics indexed on a shared 0 to 100 scale">{[15,35,55,75,95].map(y=><line key={y} x1="0" x2="100" y1={y} y2={y} className="chart-grid" />)}{series.map((item,si)=>{const d=item.values.map((point,i)=>`${i?'L':'M'} ${xAt(point.date)} ${94-point.index*.82}`).join(' ');return <path key={item.key} d={d} fill="none" stroke={colors[si%colors.length]} strokeWidth="1.6" vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" />;})}</svg><div className="chart-foot"><span>{fmtDate(minTime ? new Date(minTime).toISOString() : null,{month:'short',day:'numeric'})}</span><span>relative scale</span><span>{fmtDate(maxTime ? new Date(maxTime).toISOString() : null,{month:'short',day:'numeric'})}</span></div></div><div className="overlay-legend">{series.map((item,index)=>{const nums=item.values.map(point=>point.value);return <span key={item.key} title={fieldDefinition(item.path)}><i style={{background:colors[index%colors.length]}} />{item.label}<small>{Math.min(...nums).toPrecision(4)}–{Math.max(...nums).toPrecision(4)} {fieldUnit(item.path)}</small></span>;})}</div></div>;
}
function Loading() { return <div className="loading-panel"><LoaderCircle className="spin" size={22} /> Loading your local data…</div>; }
function EmptyState({ onConnect, configured, onSync }: { onConnect: () => void; configured: boolean; onSync: () => void }) {
  return <div className="welcome-panel"><div className="welcome-orbit"><div className="orbit-core"><HeartPulse size={25} /></div><span className="orbit-dot dot-one" /><span className="orbit-dot dot-two" /><span className="orbit-dot dot-three" /></div><div className="section-kicker">START WITH YOUR OWN SIGNALS</div><h2>Your data deserves a <em>closer look.</em></h2><p>Connect WHOOP and sync your history to see personal recovery patterns, log meaningful context, and browse every available data field.</p><div className="welcome-steps"><div><span>01</span><strong>Connect</strong><small>Authorize WHOOP data access</small></div><ArrowRight size={15} /><div><span>02</span><strong>Sync</strong><small>Bring your history into local storage</small></div><ArrowRight size={15} /><div><span>03</span><strong>Explore</strong><small>Review patterns and all fields</small></div></div><div className="welcome-actions">{configured ? <button className="button button-primary" onClick={onConnect}><Link2 size={16} /> Connect WHOOP</button> : <button className="button button-primary" disabled><Link2 size={16} /> Configure WHOOP credentials</button>}<button className="button button-subtle" onClick={onSync} disabled>Sync requires a connection <RefreshCw size={14} /></button></div><div className="welcome-private"><Shield size={14} /> Your health data stays on this Mac.</div></div>;
}
function Modal({ title, subtitle, children, onClose }: { title: string; subtitle: string; children: React.ReactNode; onClose: () => void }) { return <div className="modal-scrim" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}><section className="modal-card" role="dialog" aria-modal="true" aria-label={title}><div className="modal-head"><div><span className="section-kicker">PERSONAL CONTEXT</span><h2>{title}</h2><p>{subtitle}</p></div><button className="icon-button" onClick={onClose} aria-label="Close dialog"><X size={18} /></button></div>{children}</section></div>; }

export default App;
