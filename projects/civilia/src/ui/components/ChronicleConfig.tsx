import type { ChronicleConfig } from '../../sim/chronicle/types'

interface Props {
  config: ChronicleConfig;
  onChange: (config: ChronicleConfig) => void;
  onClose: () => void;
}

export function ChronicleConfigPanel({ config, onChange, onClose }: Props) {
  const update = <K extends keyof ChronicleConfig>(key: K, value: ChronicleConfig[K]) => {
    onChange({ ...config, [key]: value })
  }

  return (
    <div className="chronicle-config">
      <div className="chronicle-config-head">
        <strong>⚙ Chronicle Settings</strong>
        <button className="close-btn" onClick={onClose}>
          ✕
        </button>
      </div>

      <label className="chronicle-config-row">
        <span>Enabled</span>
        <input
          type="checkbox"
          checked={config.enabled}
          onChange={(e) => update('enabled', e.target.checked)}
        />
      </label>

      <label className="chronicle-config-row">
        <span>Cadence</span>
        <select
          className="chronicle-select"
          value={config.cadence}
          onChange={(e) => update('cadence', e.target.value as ChronicleConfig['cadence'])}
        >
          <option value="weekly">Weekly</option>
          <option value="monthly">Monthly</option>
          <option value="seasonal">Seasonal</option>
        </select>
      </label>

      <label className="chronicle-config-row">
        <span>Consequence threshold</span>
        <input
          type="number"
          min={0}
          max={3}
          value={config.consequenceThreshold}
          onChange={(e) => update('consequenceThreshold', Number(e.target.value))}
          className="chronicle-number-input"
        />
      </label>

      <label className="chronicle-config-row">
        <span>Max articles per period</span>
        <input
          type="number"
          min={1}
          max={10}
          value={config.maxArticlesPerPeriod}
          onChange={(e) => update('maxArticlesPerPeriod', Number(e.target.value))}
          className="chronicle-number-input"
        />
      </label>
    </div>
  )
}
