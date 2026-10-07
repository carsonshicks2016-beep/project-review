import { useEffect, useRef, useState } from 'react'
import { createWorld, deserializeWorld, serializeWorld, stepDay, stepPhase } from '../sim'
import type { SimEvent, World } from '../sim'
import { currentPhase, DEFAULT_CHRONICLE_CONFIG } from '../sim'
import type { ChronicleConfig } from '../sim/chronicle/types'
import { PHASE_EMOJI } from './palette'
import { WorldMap } from './components/WorldMap'
import { Timeline } from './components/Timeline'
import { StatsBar } from './components/StatsBar'
import { EventTicker } from './components/EventTicker'
import { BuildingCard } from './components/BuildingCard'
import { SocialGraph } from './components/SocialGraph'
import { AgentDrawer } from './components/AgentDrawer'
import { PhraseCard } from './components/PhraseCard'
import type { PhraseSelection } from './components/PhraseCard'
import { ChroniclePanel } from './components/ChroniclePanel'
import { ChronicleConfigPanel } from './components/ChronicleConfig'
import { OllamaSimulator } from './components/OllamaSimulator'

const DEFAULT_SEED = 'ashvale-1'
const SAVE_KEY = 'civilia-save'

const SPEEDS = [
  { label: '1×', ms: 2600 },
  { label: '2×', ms: 1300 },
  { label: '4×', ms: 650 },
] as const

type Tab = 'town' | 'people' | 'log' | 'chronicle' | 'ollama'

export function App() {
  const [seed, setSeed] = useState(DEFAULT_SEED)
  const [seedDraft, setSeedDraft] = useState(DEFAULT_SEED)
  const worldRef = useRef<World>(createWorld(DEFAULT_SEED))
  // The world mutates in place; this counter tells React when to re-render.
  const [version, setVersion] = useState(0)
  const [running, setRunning] = useState(false)
  const [speedIdx, setSpeedIdx] = useState(0)
  const [tab, setTab] = useState<Tab>('town')
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)
  const [selectedPlaceId, setSelectedPlaceId] = useState<string | null>(null)
  const [selectedPhrase, setSelectedPhrase] = useState<PhraseSelection | null>(null)
  const [recentEvents, setRecentEvents] = useState<SimEvent[]>([])
  const [chronicleConfig, setChronicleConfig] = useState<ChronicleConfig>(DEFAULT_CHRONICLE_CONFIG)
  const [showChronicleConfig, setShowChronicleConfig] = useState(false)
  const [saveNote, setSaveNote] = useState<string | null>(null)

  const world = worldRef.current

  // Keep world.chronicleConfig in sync with UI state
  useEffect(() => {
    worldRef.current.chronicleConfig = chronicleConfig
  }, [chronicleConfig])

  /** Step the sim and capture the new events for map bubbles. */
  const advance = (fn: (w: World) => void) => {
    const before = worldRef.current.events.length
    fn(worldRef.current)
    setRecentEvents(worldRef.current.events.slice(before))
    setVersion((v) => v + 1)
  }

  const resetSelections = () => {
    setSelectedAgentId(null)
    setSelectedPlaceId(null)
    setSelectedPhrase(null)
    setRecentEvents([])
  }

  const onNewWorld = () => {
    setRunning(false)
    const w = createWorld(seedDraft)
    w.chronicleConfig = chronicleConfig
    worldRef.current = w
    setSeed(seedDraft)
    resetSelections()
    setVersion((v) => v + 1)
  }

  const flashNote = (note: string) => {
    setSaveNote(note)
    window.setTimeout(() => setSaveNote(null), 2200)
  }

  const onSave = () => {
    try {
      localStorage.setItem(SAVE_KEY, serializeWorld(worldRef.current))
      flashNote(`Saved day ${worldRef.current.day}`)
    } catch (err) {
      flashNote('Save failed')
      console.error(err)
    }
  }

  const onLoad = () => {
    const json = localStorage.getItem(SAVE_KEY)
    if (!json) {
      flashNote('No save found')
      return
    }
    try {
      const w = deserializeWorld(json)
      setRunning(false)
      worldRef.current = w
      setSeed(w.seed)
      setSeedDraft(w.seed)
      resetSelections()
      setVersion((v) => v + 1)
      flashNote(`Loaded day ${w.day}`)
    } catch (err) {
      flashNote('Load failed')
      console.error(err)
    }
  }

  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => advance(stepPhase), SPEEDS[speedIdx].ms)
    return () => clearInterval(timer)
  }, [running, speedIdx])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-name">Civilia</span>
          <span className="brand-town">{world.name}</span>
        </div>
        <div className="clock">
          {PHASE_EMOJI[currentPhase(world)]} Day {world.day} · <span className="phase">{currentPhase(world)}</span>
        </div>
        <div className="controls">
          <button onClick={() => advance(stepPhase)} disabled={running}>
            Step phase
          </button>
          <button onClick={() => advance(stepDay)} disabled={running}>
            Step day
          </button>
          <button className={running ? 'danger' : 'primary'} onClick={() => setRunning(!running)}>
            {running ? '⏸ Pause' : '▶ Play'}
          </button>
          <button
            className="speed-btn"
            title="playback speed"
            onClick={() => setSpeedIdx((speedIdx + 1) % SPEEDS.length)}
          >
            {SPEEDS[speedIdx].label}
          </button>
          <button title="Save world to this browser" onClick={onSave}>
            💾
          </button>
          <button title="Load saved world" onClick={onLoad}>
            📂
          </button>
          {saveNote && <span className="save-note">{saveNote}</span>}
        </div>
        <div className="tabs">
          <button className={tab === 'town' ? 'tab active' : 'tab'} onClick={() => setTab('town')}>
            Map
          </button>
          <button className={tab === 'people' ? 'tab active' : 'tab'} onClick={() => setTab('people')}>
            People
          </button>
          <button className={tab === 'log' ? 'tab active' : 'tab'} onClick={() => setTab('log')}>
            Log
          </button>
          <button className={tab === 'chronicle' ? 'tab active' : 'tab'} onClick={() => setTab('chronicle')}>
            📜 Chronicle
          </button>
          <button className={tab === 'ollama' ? 'tab active' : 'tab'} onClick={() => setTab('ollama')}>
            🤖 Ollama Chat
          </button>
          <button
            className="tab chronicle-settings-btn"
            title="Chronicle settings"
            onClick={() => setShowChronicleConfig(!showChronicleConfig)}
          >
            ⚙
          </button>
        </div>
        <div className="seed-controls">
          <input
            value={seedDraft}
            onChange={(e) => setSeedDraft(e.target.value)}
            placeholder="world seed"
            title="World seed"
          />
          <button onClick={onNewWorld}>New world</button>
          <span className="seed-current" title="active seed">
            {seed}
          </span>
        </div>
      </header>

      <StatsBar world={world} />

      <div className="stage">
        {tab === 'town' ? (
          <>
            <WorldMap
              world={world}
              version={version}
              recentEvents={recentEvents}
              selectedAgentId={selectedAgentId}
              onSelectAgent={(id) => {
                setSelectedAgentId(id)
                if (id) setSelectedPlaceId(null)
              }}
              onSelectPlace={(id) => {
                setSelectedPlaceId(id)
                if (id) setSelectedAgentId(null)
              }}
              onSelectPhrase={setSelectedPhrase}
            />
            <EventTicker world={world} onSelectAgent={setSelectedAgentId} />
            {selectedPlaceId && (
              <BuildingCard
                world={world}
                placeId={selectedPlaceId}
                onSelectAgent={(id) => {
                  setSelectedAgentId(id)
                  setSelectedPlaceId(null)
                }}
                onClose={() => setSelectedPlaceId(null)}
              />
            )}
            {selectedPhrase && (
              <PhraseCard
                world={world}
                selection={selectedPhrase}
                onSelectAgent={(id) => {
                  setSelectedAgentId(id)
                  setSelectedPhrase(null)
                }}
                onClose={() => setSelectedPhrase(null)}
              />
            )}
          </>
        ) : tab === 'people' ? (
          <div className="log-pane">
            <SocialGraph world={world} selectedAgentId={selectedAgentId} onSelectAgent={setSelectedAgentId} />
          </div>
        ) : tab === 'log' ? (
          <div className="log-pane">
            <Timeline world={world} selectedAgentId={selectedAgentId} />
          </div>
        ) : tab === 'chronicle' ? (
          <div className="log-pane">
            <ChroniclePanel world={world} onClose={() => setTab('town')} />
          </div>
        ) : (
          <div className="log-pane">
            <OllamaSimulator />
          </div>
        )}
        <AgentDrawer world={world} agentId={selectedAgentId} onClose={() => setSelectedAgentId(null)} />
        {showChronicleConfig && (
          <ChronicleConfigPanel
            config={chronicleConfig}
            onChange={setChronicleConfig}
            onClose={() => setShowChronicleConfig(false)}
          />
        )}
      </div>
    </div>
  )
}
