import { useState } from 'react'
import type { World } from '../../sim'
import type { ChronicleArticle } from '../../sim/chronicle/types'

interface Props {
  world: World;
  onClose: () => void;
}

const BIAS_LABEL: Record<ChronicleArticle['bias'], string> = {
  pro: 'Pro',
  neutral: 'Neutral',
  anti: 'Anti',
}

const BIAS_CLASS: Record<ChronicleArticle['bias'], string> = {
  pro: 'bias-pro',
  neutral: 'bias-neutral',
  anti: 'bias-anti',
}

type FilterMode = 'all' | 'faction' | 'period'

export function ChroniclePanel({ world, onClose }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [filterMode, setFilterMode] = useState<FilterMode>('all')
  const [selectedFaction, setSelectedFaction] = useState<string>('')
  const [selectedPeriod, setSelectedPeriod] = useState<string>('')

  const articles = world.chronicleArticles ?? []

  // Derive unique factions and periods for filter dropdowns
  const factions = [...new Set(articles.map((a) => a.authorFaction))].sort()
  const periods = [...new Set(articles.map((a) => a.period))].sort()

  const filtered = articles.filter((a) => {
    if (filterMode === 'faction' && selectedFaction && a.authorFaction !== selectedFaction) return false
    if (filterMode === 'period' && selectedPeriod && a.period !== selectedPeriod) return false
    return true
  })

  // Show newest first
  const sorted = [...filtered].reverse()

  return (
    <div className="chronicle-panel">
      <div className="chronicle-header">
        <h2>📜 The Chronicle</h2>
        <button className="close-btn" onClick={onClose}>
          ✕
        </button>
      </div>

      <div className="chronicle-filters">
        <button
          className={`chronicle-filter-tab ${filterMode === 'all' ? 'active' : ''}`}
          onClick={() => setFilterMode('all')}
        >
          All
        </button>
        <button
          className={`chronicle-filter-tab ${filterMode === 'faction' ? 'active' : ''}`}
          onClick={() => setFilterMode('faction')}
        >
          By Faction
        </button>
        <button
          className={`chronicle-filter-tab ${filterMode === 'period' ? 'active' : ''}`}
          onClick={() => setFilterMode('period')}
        >
          By Period
        </button>

        {filterMode === 'faction' && (
          <select
            className="chronicle-select"
            value={selectedFaction}
            onChange={(e) => setSelectedFaction(e.target.value)}
          >
            <option value="">All factions</option>
            {factions.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        )}

        {filterMode === 'period' && (
          <select
            className="chronicle-select"
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
          >
            <option value="">All periods</option>
            {periods.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        )}
      </div>

      {sorted.length === 0 ? (
        <div className="chronicle-empty">
          <span className="chronicle-empty-icon">📜</span>
          <p>The chronicle awaits its first story...</p>
          <span className="chronicle-empty-hint">
            Notable events will be recorded here as the simulation runs.
          </span>
        </div>
      ) : (
        <ul className="chronicle-list">
          {sorted.map((article) => {
            const expanded = expandedId === article.id
            return (
              <li key={article.id} className={`chronicle-article ${expanded ? 'expanded' : ''}`}>
                <button
                  className="chronicle-article-head"
                  onClick={() => setExpandedId(expanded ? null : article.id)}
                >
                  <span className="chronicle-title">{article.title}</span>
                  <div className="chronicle-meta">
                    <span className={`chronicle-bias ${BIAS_CLASS[article.bias]}`}>
                      {BIAS_LABEL[article.bias]}
                    </span>
                    <span className="chronicle-period">{article.period}</span>
                    <span className="chronicle-faction">{article.authorFaction}</span>
                  </div>
                </button>
                {expanded && (
                  <div className="chronicle-body">
                    <div className="chronicle-content">{article.content}</div>
                    {article.sourceEventIds.length > 0 && (
                      <div className="chronicle-sources">
                        <h3>Source Events</h3>
                        <ul>
                          {article.sourceEventIds.map((eid) => {
                            const ev = world.events.find((e) => e.id === eid)
                            return (
                              <li key={eid} className="chronicle-source-event">
                                {ev ? `Day ${ev.day} · ${ev.type}` : eid}
                              </li>
                            )
                          })}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
