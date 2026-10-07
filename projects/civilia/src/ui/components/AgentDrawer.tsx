import type { World } from '../../sim'
import { AgentPanel } from './AgentPanel'

interface Props {
  world: World
  agentId: string | null
  onClose: () => void
}

/** Slide-over wrapping the full agent profile + decision inspector. */
export function AgentDrawer({ world, agentId, onClose }: Props) {
  return (
    <aside className={`agent-drawer ${agentId ? 'open' : ''}`}>
      {agentId && (
        <>
          <button className="close-btn drawer-close" onClick={onClose}>
            ✕
          </button>
          <AgentPanel world={world} agentId={agentId} />
        </>
      )}
    </aside>
  )
}
