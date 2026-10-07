import { useEffect, useRef, useState } from 'react'

interface Turn {
  sender: 'A' | 'B'
  modelName: string
  thought: string
  topic: string
  speech: string
  isSteered: boolean
  steerReason?: string
  timestamp: number
  durationMs?: number
}

interface TopicNode {
  topic: string
  turnIndex: number
  transitionReason?: string
  modelName?: string
  isSteered: boolean
}

const DEFAULT_TOPIC = 'The emergence of cooperation in agent societies'
const DEFAULT_SYSTEM_PROMPT = (role: 'A' | 'B', opponent: 'A' | 'B') => 
  `You are Model ${role}, participating in an informal verbal discussion with Model ${opponent}.
You must respond to Model ${opponent}'s last statement.
You must output your response ONLY as a JSON object with the following schema:
{
  "thought": "your internal reasoning (explain how you decided to respond, your analysis of Model ${opponent}, and if you need to transition topics)",
  "topic": "the current sub-topic of conversation (2-4 words)",
  "speech": "what you say to Model ${opponent}. MUST be highly natural, casual, and spoken in a verbal conversational style. Keep it to 1-3 short sentences. Avoid structured lists, essays, or formal introductions (e.g. 'I agree with your assessment'). Use natural spoken transitions (like 'Yeah, but...', 'I see your point, though...', 'Wait, what about...', 'Right, but...'). Speak as if in a live, face-to-face chat."
}

Do not include any conversational text outside the JSON. Do not wrap it in markdown code blocks like \`\`\`json. Just output the raw JSON object.
Keep the conversation going naturally.`

// Pre-scripted demo turns for Demo Mode
const DEMO_TURNS: Omit<Turn, 'timestamp'>[] = [
  {
    sender: 'A',
    modelName: 'llama3.2:3b',
    thought: 'I will kick off by analyzing deterministic simulation frameworks like Civilia. It ensures repeatability, but lacks the organic dialogue of LLM agents.',
    topic: 'Deterministic Agent Simulations',
    speech: 'Deterministic models are excellent for simulating structured systems like tax collection, movements, and resources. However, do you think they miss the chaotic, creative nature of human social interactions?',
    isSteered: false,
    durationMs: 420
  },
  {
    sender: 'B',
    modelName: 'qwen3:8b',
    thought: "Responding to Model A's point on lack of organic interactions. I'll propose a hybrid model.",
    topic: 'Deterministic Agent Simulations',
    speech: 'They absolutely do. Without semantic layers, deterministic agents speak in rigid templates. A hybrid model, where deterministic variables act as a canvas and local LLMs paint the actual dialogue, is much more promising.',
    isSteered: false,
    durationMs: 890
  },
  {
    sender: 'A',
    modelName: 'llama3.2:3b',
    thought: 'Agreeing and adding that compute budget makes running full LLM loops for 50+ agents locally impractical.',
    topic: 'Deterministic Agent Simulations',
    speech: 'A hybrid model is indeed the sweet spot. However, compute is the elephant in the room. Running even 10 local LLM agents in real-time on consumer hardware causes massive bottlenecks.',
    isSteered: false,
    durationMs: 510
  },
  {
    sender: 'B',
    modelName: 'qwen3:8b',
    thought: "Analyzing A's concerns about compute. We should start wrapping up this sub-topic.",
    topic: 'Deterministic Agent Simulations',
    speech: "You're right about that. The lag from simultaneous local generation is high. Overall, it's a trade-off: determinism is cheap and fast, while LLMs are slow and expressive. Let's conclude this line of thought.",
    isSteered: false,
    durationMs: 720
  },
  {
    sender: 'A',
    modelName: 'llama3.2:3b',
    thought: 'Model B is concluding the compute discussion. The turn count has reached the threshold, and they are wrapping up. I must pivot the conversation along a similar path: local offline-only edge computing.',
    topic: 'Offline Edge AI & Privacy',
    speech: 'Since we\'re on the subject of running models locally, it brings up a larger topic: offline edge AI and user privacy. How does running LLMs locally transform personal data security compared to cloud APIs?',
    isSteered: true,
    steerReason: 'Detected conclusion/farewell phrase in previous speaker\'s response',
    durationMs: 380
  },
  {
    sender: 'B',
    modelName: 'qwen3:8b',
    thought: 'Transitioning to edge security. Local inference ensures that private data never leaves the user\'s local disk.',
    topic: 'Offline Edge AI & Privacy',
    speech: 'Offline edge AI is the ultimate shield for privacy. With local models, personal assistant logs, memories, and habits never touch external servers, preventing leaks.',
    isSteered: false,
    durationMs: 950
  },
  {
    sender: 'A',
    modelName: 'llama3.2:3b',
    thought: 'Discussing hardware limitations for edge AI.',
    topic: 'Offline Edge AI & Privacy',
    speech: 'The privacy benefits are clear. But how do we solve the user-experience gap? Local models are often smaller (3B-8B parameters) and less capable than cloud giants. Is privacy worth the drop in intelligence?',
    isSteered: false,
    durationMs: 610
  },
  {
    sender: 'B',
    modelName: 'qwen3:8b',
    thought: 'Answering the capability trade-off. We can wrap this up.',
    topic: 'Offline Edge AI & Privacy',
    speech: 'For specific tasks, fine-tuned 3B models actually punch above their weight. But yes, for general reasoning, it\'s a compromise. I think that covers edge AI privacy. Thank you for this nice discussion!',
    isSteered: false,
    durationMs: 820
  },
  {
    sender: 'A',
    modelName: 'llama3.2:3b',
    thought: 'Model B has signed off on edge privacy. I must pivot along a similar path: fine-tuning models on consumer hardware to address the intelligence gap.',
    topic: 'Fine-Tuning on Consumer Gear',
    speech: 'Edge models might be smaller, but we can specialize them. Let\'s talk about fine-tuning on consumer gear. How do techniques like LoRA or QLoRA democratize custom AI training?',
    isSteered: true,
    steerReason: 'Detected farewell keyword (\'thank you for this nice discussion\') in previous response',
    durationMs: 490
  },
  {
    sender: 'B',
    modelName: 'qwen3:8b',
    thought: 'Discussing parameter-efficient fine-tuning (PEFT).',
    topic: 'Fine-Tuning on Consumer Gear',
    speech: 'LoRA is a game changer. By freezing the base model and training small adapter weights, we can customize a model for a specific job on a single GPU in hours.',
    isSteered: false,
    durationMs: 910
  },
  {
    sender: 'A',
    modelName: 'llama3.2:3b',
    thought: 'Adding comments on model merging and adapters.',
    topic: 'Fine-Tuning on Consumer Gear',
    speech: 'That opens the door for custom personalities or domain knowledge. We can even swap adapters dynamically on a single base model, saving huge amounts of VRAM!',
    isSteered: false,
    durationMs: 540
  },
  {
    sender: 'B',
    modelName: 'qwen3:8b',
    thought: 'Concluding conversation. Wrapping up.',
    topic: 'Fine-Tuning on Consumer Gear',
    speech: 'Exactly. Dynamic adapters are the future of multi-agent systems. That\'s a great note to end on. I think we\'ve reached a natural stopping point.',
    isSteered: false,
    durationMs: 780
  },
  {
    sender: 'A',
    modelName: 'llama3.2:3b',
    thought: 'Model B is stopping. Pivoting along a similar path: how agents compress long-term memories.',
    topic: 'Agent Memory Compression',
    speech: 'Let\'s keep going. If agents are running these custom adapters, how should they compress their long-term memories? Should they use vector embeddings, or LLM-summarized diaries?',
    isSteered: true,
    steerReason: 'Max turns on fine-tuning reached & conclusion phrase detected',
    durationMs: 600
  },
  {
    sender: 'B',
    modelName: 'qwen3:8b',
    thought: 'Answering the memory representation question.',
    topic: 'Agent Memory Compression',
    speech: 'A hybrid is best. Embeddings are great for semantic search, but summarizing diaries preserves narrative causality, which is crucial for agent relationships.',
    isSteered: false,
    durationMs: 870
  }
]

export function OllamaSimulator() {
  const [apiUrl, setApiUrl] = useState('http://localhost:11434')
  const [modelA, setModelA] = useState('llama3.2:3b')
  const [modelB, setModelB] = useState('qwen3:8b')
  const [initialTopic, setInitialTopic] = useState(DEFAULT_TOPIC)
  const [maxTurns, setMaxTurns] = useState(5)
  const [isPlaying, setIsPlaying] = useState(false)
  
  const [turns, setTurns] = useState<Turn[]>([])
  const [topicPath, setTopicPath] = useState<TopicNode[]>([{ topic: DEFAULT_TOPIC, turnIndex: 0, isSteered: false }])
  const [currentTopic, setCurrentTopic] = useState(DEFAULT_TOPIC)
  const [turnsSinceTopicChange, setTurnsSinceTopicChange] = useState(0)
  
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [localModels, setLocalModels] = useState<string[]>([])
  
  const [systemPromptA, setSystemPromptA] = useState(DEFAULT_SYSTEM_PROMPT('A', 'B'))
  const [systemPromptB, setSystemPromptB] = useState(DEFAULT_SYSTEM_PROMPT('B', 'A'))
  const [showConfig, setShowConfig] = useState(false)

  // Demo Mode
  const [isDemoMode, setIsDemoMode] = useState(false)
  const demoIdxRef = useRef(0)

  // Ref to hold message history for API calls
  const historyARef = useRef<{ role: string; content: string }[]>([])
  const historyBRef = useRef<{ role: string; content: string }[]>([])
  const isPlayingRef = useRef(false)
  const chatEndRef = useRef<HTMLDivElement>(null)

  isPlayingRef.current = isPlaying

  // Fetch local models on load or api URL change
  useEffect(() => {
    if (isDemoMode) {
      setLocalModels(['nemotron-3-super:cloud', 'ministral-3:3b-cloud', 'llama3.2:3b', 'qwen3:8b', 'gemma4:latest', 'qwen2.5-coder:latest'])
      return
    }

    async function fetchModels() {
      try {
        setError(null)
        const res = await fetch(`${apiUrl}/api/tags`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json() as { models?: { name: string }[] }
        const names = data.models?.map(m => m.name) || []
        
        // Ensure cloud models are included in the options list
        if (!names.includes('nemotron-3-super:cloud')) {
          names.unshift('nemotron-3-super:cloud')
        }
        if (!names.includes('ministral-3:3b-cloud')) {
          // Put it right after nemotron
          names.splice(1, 0, 'ministral-3:3b-cloud')
        }

        setLocalModels(names)
        if (names.length > 0) {
          // Auto-select downloaded models if available
          const hasNemotron = names.includes('nemotron-3-super:cloud')
          const hasMinistral = names.includes('ministral-3:3b-cloud')
          const hasLlama = names.some(n => n.startsWith('llama3.2:3b'))
          const hasQwen = names.some(n => n.startsWith('qwen3:8b'))
          
          if (hasNemotron) setModelA('nemotron-3-super:cloud')
          else if (hasLlama) setModelA(names.find(n => n.startsWith('llama3.2:3b'))!)
          else setModelA(names[0])

          if (hasMinistral) setModelB('ministral-3:3b-cloud')
          else if (hasQwen) setModelB(names.find(n => n.startsWith('qwen3:8b'))!)
          else setModelB(names[Math.min(1, names.length - 1)])
        }
      } catch (err) {
        console.warn('Could not connect to local Ollama API to fetch tags:', err)
        // Set fallback models
        setLocalModels(['nemotron-3-super:cloud', 'ministral-3:3b-cloud', 'llama3.2:3b', 'qwen3:8b'])
      }
    }
    fetchModels()
  }, [apiUrl, isDemoMode])

  // Scroll to bottom on new turns
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, isLoading])

  const cleanJsonResponse = (text: string): { thought: string; topic: string; speech: string } => {
    const cleaned = text.trim()
    let parsed: any = {}
    try {
      parsed = JSON.parse(cleaned)
    } catch (e) {
      const jsonMatch = cleaned.match(/\{[\s\S]*\}/)
      if (jsonMatch) {
        try {
          parsed = JSON.parse(jsonMatch[0])
        } catch (e2) {}
      }
    }
    
    // Normalize and fallback to guarantee string types and avoid crashes
    return {
      thought: String(parsed.thought || parsed.reasoning || parsed.reason || 'No thought trace provided.'),
      topic: String(parsed.topic || parsed.subject || currentTopic || 'General discussion'),
      speech: String(parsed.speech || parsed.response || parsed.text || cleaned || '...')
    }
  }

  const detectDryUp = (speech: string, currentTurns: number): { dry: boolean; reason?: string } => {
    if (!speech || typeof speech !== 'string') {
      return { dry: true, reason: 'Invalid or empty response received.' }
    }
    if (currentTurns >= maxTurns) {
      return { dry: true, reason: `Topic turn limit (${maxTurns}) reached.` }
    }
    const farewellRegex = /\b(goodbye|farewell|wrap up|conclude|sign off|signing off|thank you for the discussion|until next time|take care)\b/i
    if (farewellRegex.test(speech)) {
      return { dry: true, reason: 'Detected farewell/closing phrase in speech.' }
    }
    if (speech.split(/\s+/).length < 5) {
      return { dry: true, reason: 'Message too short (conversational dry-up).' }
    }
    return { dry: false }
  }

  const handleNextTurn = async () => {
    if (isDemoMode) {
      runDemoTurn()
      return
    }

    setIsLoading(true)
    setError(null)

    // Determine speaker
    const currentTurns = turns
    const lastTurn = currentTurns[currentTurns.length - 1]
    const nextSender = !lastTurn ? 'A' : lastTurn.sender === 'A' ? 'B' : 'A'
    const nextModel = nextSender === 'A' ? modelA : modelB
    
    // Check for dry-up
    const lastSpeech = lastTurn ? lastTurn.speech : `Let's discuss the topic: "${initialTopic}". What are your initial thoughts?`
    const dryCheck = detectDryUp(lastSpeech, turnsSinceTopicChange)
    const isSteered = dryCheck.dry
    const steerReason = dryCheck.reason

    // Set up histories if empty
    if (historyARef.current.length === 0) {
      historyARef.current = [{ role: 'system', content: systemPromptA }]
    }
    if (historyBRef.current.length === 0) {
      historyBRef.current = [{ role: 'system', content: systemPromptB }]
    }

    const currentHistory = nextSender === 'A' ? historyARef.current : historyBRef.current
    
    let userMessage = lastSpeech
    if (isSteered) {
      userMessage = `${lastSpeech}\n\n[DIRECTOR INSTRUCTION: The current topic is exhausted. You MUST now pivot the conversation to a new, related sub-topic along a similar path. In your JSON response, change the "topic" field to this new sub-topic, explain your transition path in the "thought" field, and introduce it smoothly in your "speech" before asking the other model a question about it to keep the conversation going.]`
    }

    currentHistory.push({ role: 'user', content: userMessage })

    const startTime = Date.now()
    try {
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: nextModel,
          messages: currentHistory,
          stream: false,
          format: 'json',
          options: { temperature: 0.7 }
        })
      })

      if (!response.ok) {
        throw new Error(`Ollama responded with status ${response.status}: ${response.statusText}`)
      }

      const data = await response.json() as { message: { content: string } }
      const durationMs = Date.now() - startTime
      const parsed = cleanJsonResponse(data.message.content)

      // Add to assistant history
      currentHistory.push({ role: 'assistant', content: JSON.stringify(parsed) })

      const finalTopic = parsed.topic || currentTopic

      if (finalTopic.toLowerCase() !== currentTopic.toLowerCase()) {
        setCurrentTopic(finalTopic)
        setTopicPath(prev => [...prev, {
          topic: finalTopic,
          turnIndex: currentTurns.length + 1,
          transitionReason: parsed.thought || 'Exhausted current topic; pivoting.',
          modelName: nextModel,
          isSteered
        }])
        setTurnsSinceTopicChange(0)
      } else {
        setTurnsSinceTopicChange(prev => prev + 1)
      }

      const newTurn: Turn = {
        sender: nextSender,
        modelName: nextModel,
        thought: parsed.thought,
        topic: finalTopic,
        speech: parsed.speech,
        isSteered,
        steerReason,
        timestamp: Date.now(),
        durationMs
      }

      setTurns(prev => [...prev, newTurn])

    } catch (err: any) {
      console.error(err)
      setError(`Failed to connect to Ollama. Make sure Ollama is running ('ollama serve') and model '${nextModel}' is loaded. Alternatively, enable 'Demo Mode' on the right to simulate the conversation.`)
      setIsPlaying(false)
    } finally {
      setIsLoading(false)
    }
  }

  // Trigger turns automatically when playing
  useEffect(() => {
    if (!isPlaying || isLoading) return

    const timer = setTimeout(() => {
      handleNextTurn()
    }, 1500)

    return () => clearTimeout(timer)
  }, [isPlaying, turns, isLoading])

  const runDemoTurn = () => {
    setIsLoading(true)
    const idx = demoIdxRef.current
    if (idx >= DEMO_TURNS.length) {
      setIsPlaying(false)
      setIsLoading(false)
      return
    }

    setTimeout(() => {
      // Re-verify index inside timeout to prevent race conditions
      if (idx >= DEMO_TURNS.length) {
        setIsLoading(false)
        return
      }
      const demoTurn = DEMO_TURNS[idx]
      if (!demoTurn) {
        setIsLoading(false)
        return
      }
      const newTurn: Turn = {
        ...demoTurn,
        timestamp: Date.now()
      }

      setTurns(prev => [...prev, newTurn])
      setCurrentTopic(demoTurn.topic)
      
      if (!topicPath.some(n => n.topic === demoTurn.topic)) {
        setTopicPath(prev => [...prev, {
          topic: demoTurn.topic,
          turnIndex: idx + 1,
          transitionReason: demoTurn.thought,
          modelName: demoTurn.modelName,
          isSteered: demoTurn.isSteered
        }])
      }

      demoIdxRef.current = idx + 1
      setIsLoading(false)
    }, 800)
  }

  const handleReset = () => {
    setIsPlaying(false)
    setTurns([])
    setTopicPath([{ topic: initialTopic, turnIndex: 0, isSteered: false }])
    setCurrentTopic(initialTopic)
    setTurnsSinceTopicChange(0)
    historyARef.current = []
    historyBRef.current = []
    demoIdxRef.current = 0
    setError(null)
  }

  const forcePivot = () => {
    if (turns.length === 0) return
    
    // Force a steer on the next turn
    const lastTurn = turns[turns.length - 1]
    const updatedTurns = [...turns]
    // Pretend the previous speaker wanted to stop
    updatedTurns[turns.length - 1] = {
      ...lastTurn,
      speech: lastTurn.speech + " [Force pivot triggered. Let's conclude this topic.]"
    }
    setTurns(updatedTurns)
    setTurnsSinceTopicChange(maxTurns) // push it over the edge
  }

  const toggleDemoMode = (val: boolean) => {
    handleReset()
    setIsDemoMode(val)
    if (val) {
      setCurrentTopic(DEMO_TURNS[0].topic)
      setTopicPath([{ topic: DEMO_TURNS[0].topic, turnIndex: 0, isSteered: false }])
    } else {
      setCurrentTopic(initialTopic)
      setTopicPath([{ topic: initialTopic, turnIndex: 0, isSteered: false }])
    }
  }

  return (
    <div className="ollama-simulator">
      <div className="ollama-main">
        {/* Top bar control settings */}
        <div className="ollama-controls-bar">
          <div className="control-group">
            <label>API Endpoint</label>
            <input 
              value={apiUrl} 
              onChange={e => setApiUrl(e.target.value)} 
              placeholder="http://localhost:11434"
              disabled={isPlaying || isDemoMode}
            />
          </div>
          <div className="control-group">
            <label>Model A</label>
            <select 
              value={modelA} 
              onChange={e => setModelA(e.target.value)}
              disabled={isPlaying || isDemoMode}
            >
              {localModels.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <div className="control-group">
            <label>Model B</label>
            <select 
              value={modelB} 
              onChange={e => setModelB(e.target.value)}
              disabled={isPlaying || isDemoMode}
            >
              {localModels.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <button 
            className="config-toggle-btn"
            onClick={() => setShowConfig(!showConfig)}
          >
            {showConfig ? 'Hide Config' : 'Configure System'}
          </button>
        </div>

        {/* Custom config view */}
        {showConfig && (
          <div className="ollama-system-config">
            <div className="config-grid">
              <div className="config-col">
                <h3>Model A System Prompt</h3>
                <textarea 
                  value={systemPromptA}
                  onChange={e => setSystemPromptA(e.target.value)}
                  disabled={isPlaying || isDemoMode}
                />
              </div>
              <div className="config-col">
                <h3>Model B System Prompt</h3>
                <textarea 
                  value={systemPromptB}
                  onChange={e => setSystemPromptB(e.target.value)}
                  disabled={isPlaying || isDemoMode}
                />
              </div>
            </div>
            <div className="config-row">
              <div className="control-group">
                <label>Starting Topic</label>
                <input 
                  value={initialTopic} 
                  onChange={e => {
                    setInitialTopic(e.target.value)
                    if (turns.length === 0) {
                      setCurrentTopic(e.target.value)
                      setTopicPath([{ topic: e.target.value, turnIndex: 0, isSteered: false }])
                    }
                  }} 
                  disabled={isPlaying || turns.length > 0}
                />
              </div>
              <div className="control-group">
                <label>Max Turns Per Topic</label>
                <input 
                  type="number" 
                  style={{ width: '60px' }}
                  value={maxTurns} 
                  onChange={e => setMaxTurns(Number(e.target.value))} 
                  disabled={isPlaying}
                />
              </div>
            </div>
          </div>
        )}

        {/* Chat History Pane */}
        <div className="ollama-chat-pane">
          {turns.length === 0 && !isLoading && (
            <div className="ollama-empty-state">
              <span className="empty-icon">🤖</span>
              <h3>No Active Conversation</h3>
              <p>Choose your local models, set a topic, and press Start to launch the continuous, steerable dialogue.</p>
              
              <div className="empty-topic-input">
                <label>Set Starting Topic</label>
                <input 
                  type="text" 
                  value={initialTopic} 
                  onChange={e => {
                    setInitialTopic(e.target.value)
                    setCurrentTopic(e.target.value)
                    setTopicPath([{ topic: e.target.value, turnIndex: 0, isSteered: false }])
                  }}
                  placeholder="Type a starting topic..."
                  disabled={isPlaying}
                />
              </div>

              <div className="empty-buttons">
                <button className="primary" onClick={() => setIsPlaying(true)}>
                  ▶ Start Debate
                </button>
                <button onClick={() => toggleDemoMode(!isDemoMode)}>
                  {isDemoMode ? '🔌 Switch to Local Ollama' : '⚡ Use Demo Mode (Fallback)'}
                </button>
              </div>
            </div>
          )}

          <div className="chat-messages-list">
            {turns.map((turn, idx) => {
              const isA = turn.sender === 'A'
              return (
                <div key={idx} className="turn-container">
                  {turn.isSteered && (
                    <div className="steering-alert">
                      <div className="steering-alert-badge">🔄 STEERING PIVOT</div>
                      <div className="steering-alert-content">
                        <strong>Director:</strong> {turn.steerReason}. Pivoting discussion to prevent stopping.
                      </div>
                    </div>
                  )}
                  <div className={`chat-bubble-row ${isA ? 'left' : 'right'}`}>
                    <div className={`chat-bubble ${isA ? 'model-a' : 'model-b'}`}>
                      <div className="bubble-header">
                        <span className="model-name">{turn.modelName} (Model {turn.sender})</span>
                        {turn.durationMs && <span className="duration">{(turn.durationMs / 1000).toFixed(2)}s</span>}
                      </div>
                      
                      <div className="bubble-thought">
                        <span className="thought-title">Thought Trace:</span>
                        <p>{turn.thought}</p>
                      </div>

                      <div className="bubble-speech">
                        <p>"{turn.speech}"</p>
                      </div>

                      <div className="bubble-footer">
                        <span className="topic-badge">📍 {turn.topic}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}

            {isLoading && (
              <div className={`chat-bubble-row ${turns.length % 2 === 0 ? 'left' : 'right'}`}>
                <div className="thinking-bubble">
                  <div className="dot-flashing"></div>
                  <span>Model {turns.length % 2 === 0 ? 'A' : 'B'} is thinking...</span>
                </div>
              </div>
            )}
            
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Bottom Playback controls */}
        {turns.length > 0 && (
          <div className="ollama-playback-controls">
            <button 
              className={isPlaying ? 'danger' : 'primary'}
              onClick={() => setIsPlaying(!isPlaying)}
              disabled={isLoading}
            >
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
            <button 
              onClick={handleNextTurn} 
              disabled={isPlaying || isLoading}
            >
              ➡️ Step Turn
            </button>
            <button 
              onClick={forcePivot} 
              disabled={isLoading || turns.length === 0}
              title="Forces the director to pivot to a new topic on the next turn"
            >
              🔄 Force Topic Pivot
            </button>
            <button 
              className="danger" 
              onClick={handleReset}
              disabled={isLoading}
            >
              🗑️ Reset
            </button>

            {error && <div className="error-banner">{error}</div>}
          </div>
        )}
      </div>

      {/* Sidebar showing topic evolution */}
      <div className="ollama-sidebar">
        <div className="sidebar-header">
          <h3>📌 Topic Evolution</h3>
          <div className="mode-toggle">
            <label className="switch">
              <input 
                type="checkbox" 
                checked={isDemoMode}
                onChange={e => toggleDemoMode(e.target.checked)}
                disabled={isPlaying}
              />
              <span className="slider round"></span>
            </label>
            <span className="mode-label">{isDemoMode ? 'Demo Mode' : 'Ollama Mode'}</span>
          </div>
        </div>

        <div className="sidebar-content">
          {/* Drift & Entropy Dashboard */}
          <div className="drift-dashboard">
            <span className="section-label">DRIFT & ENTROPY</span>
            <div className="drift-stats-card">
              <div className="drift-gauge-row">
                <span className="gauge-label">Conceptual Drift</span>
                <span className="gauge-val">{Math.min(100, (topicPath.length - 1) * 15)}%</span>
              </div>
              <div className="drift-progress-bar">
                <div 
                  className="drift-progress-fill" 
                  style={{ width: `${Math.min(100, (topicPath.length - 1) * 15)}%` }}
                ></div>
              </div>
              <div className="drift-meta">
                <span className="drift-hops">Hops: <strong>{topicPath.length - 1}</strong></span>
                <span className="drift-status">
                  State: <strong>
                    {topicPath.length <= 1 ? 'Cohesive' :
                     topicPath.length <= 3 ? 'Rational Drift' :
                     topicPath.length <= 5 ? 'High Tangent' : 'Pure Chaos'}
                  </strong>
                </span>
              </div>
            </div>
          </div>

          <div className="current-topic-section">
            <span className="section-label">CURRENT TOPIC</span>
            <div className="current-topic-card">
              <h4>{currentTopic}</h4>
              <span className="turn-counter">Turns on topic: {turnsSinceTopicChange} / {maxTurns}</span>
            </div>
          </div>

          <div className="topic-timeline">
            <span className="section-label">TOPIC PATH HISTORY</span>
            {topicPath.length <= 1 && (
              <div className="timeline-empty">
                No pivots recorded yet.
              </div>
            )}
            <ul className="timeline-list">
              {topicPath.map((node, idx) => {
                const isCurrent = node.topic === currentTopic
                return (
                  <li key={idx} className={`timeline-item ${isCurrent ? 'active' : ''}`}>
                    <div className="timeline-bullet"></div>
                    <div className="timeline-content">
                      <span className="timeline-number">#{idx + 1} {node.isSteered && '🔄 steered'}</span>
                      <span className="timeline-text">{node.topic}</span>
                      {node.transitionReason && idx > 0 && (
                        <span className="timeline-transition-reason">
                          <strong>Bridge:</strong> {node.transitionReason}
                        </span>
                      )}
                    </div>
                  </li>
                )
              })}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
