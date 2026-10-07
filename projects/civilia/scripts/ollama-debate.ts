import { argv } from 'process'

// Parse command line arguments
const args = argv.slice(2)
let modelA = 'nemotron-3-super:cloud'
let modelB = 'ministral-3:3b-cloud'
let apiUrl = 'http://localhost:11434'
let initialTopic = 'The emergence of cooperation in agent societies'
let turnsCount = 10
let maxTurnsPerTopic = 4

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--modelA' && args[i + 1]) modelA = args[++i]
  else if (args[i] === '--modelB' && args[i + 1]) modelB = args[++i]
  else if (args[i] === '--apiUrl' && args[i + 1]) apiUrl = args[++i]
  else if (args[i] === '--topic' && args[i + 1]) initialTopic = args[++i]
  else if (args[i] === '--turns' && args[i + 1]) turnsCount = Number(args[++i])
  else if (args[i] === '--maxTurnsPerTopic' && args[i + 1]) maxTurnsPerTopic = Number(args[++i])
}

// Set up systems prompts
const SYSTEM_PROMPT_TEMPLATE = (role: 'A' | 'B', opponent: 'A' | 'B') => `You are Model ${role}, participating in an informal verbal discussion with Model ${opponent}.
You must respond to Model ${opponent}'s last statement.
You must output your response ONLY as a JSON object with the following schema:
{
  "thought": "your internal reasoning (explain how you decided to respond, your analysis of Model ${opponent}, and if you need to transition topics)",
  "topic": "the current sub-topic of conversation (2-4 words)",
  "speech": "what you say to Model ${opponent}. MUST be highly natural, casual, and spoken in a verbal conversational style. Keep it to 1-3 short sentences. Avoid structured lists, essays, or formal introductions (e.g. 'I agree with your assessment'). Use natural spoken transitions (like 'Yeah, but...', 'I see your point, though...', 'Wait, what about...', 'Right, but...'). Speak as if in a live, face-to-face chat."
}

Do not include any conversational text outside the JSON. Do not wrap it in markdown code blocks like \`\`\`json. Just output the raw JSON object.
Keep the conversation going naturally.`

const messagesA: { role: string; content: string }[] = [
  { role: 'system', content: SYSTEM_PROMPT_TEMPLATE('A', 'B') }
]
const messagesB: { role: string; content: string }[] = [
  { role: 'system', content: SYSTEM_PROMPT_TEMPLATE('B', 'A') }
]

let currentTopic = initialTopic
const topicHistory: string[] = [initialTopic]
let turnsSinceTopicChange = 0

function cleanJsonResponse(text: string): { thought: string; topic: string; speech: string } {
  const cleaned = text.trim()
  let parsed: any = {}
  try {
    parsed = JSON.parse(cleaned)
  } catch (e) {
    // Try to extract JSON between curly braces if wrapped in text/markdown
    const jsonMatch = cleaned.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      try {
        parsed = JSON.parse(jsonMatch[0])
      } catch (e2) {
        // Fallback if regex match failed to parse
      }
    }
  }
  
  // Guarantee string fields to avoid crashes
  return {
    thought: String(parsed.thought || parsed.reasoning || parsed.reason || 'No thought trace provided.'),
    topic: String(parsed.topic || parsed.subject || currentTopic || 'General discussion'),
    speech: String(parsed.speech || parsed.response || parsed.text || cleaned || '...')
  }
}

async function checkOllamaConnection() {
  try {
    const res = await fetch(`${apiUrl}/api/tags`)
    if (!res.ok) throw new Error(`HTTP error ${res.status}`)
    const data = (await res.json()) as { models?: { name: string }[] }
    const modelNames = data.models?.map((m) => m.name) || []
    
    // Warn if selected models are not downloaded
    const hasA = modelNames.some((n) => n === modelA || n.startsWith(modelA + ':'))
    const hasB = modelNames.some((n) => n === modelB || n.startsWith(modelB + ':'))
    
    if (!hasA) {
      console.warn(`\x1b[33mWarning: Model '${modelA}' not found in local Ollama tags. Available: ${modelNames.join(', ')}\x1b[0m`)
    }
    if (!hasB) {
      console.warn(`\x1b[33mWarning: Model '${modelB}' not found in local Ollama tags. Available: ${modelNames.join(', ')}\x1b[0m`)
    }
  } catch (error) {
    console.error(`\x1b[31mError: Could not connect to Ollama API at ${apiUrl}. Please make sure Ollama is running.\x1b[0m`)
    process.exit(1)
  }
}

async function queryModel(model: string, messages: { role: string; content: string }[]): Promise<string> {
  const res = await fetch(`${apiUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      options: {
        temperature: 0.7
      },
      format: 'json'
    })
  })
  
  if (!res.ok) {
    throw new Error(`Ollama API error: ${res.statusText} (${res.status})`)
  }
  
  const data = (await res.json()) as { message: { content: string } }
  return data.message.content
}

function detectDryUp(speech: string): { dry: boolean; reason?: string } {
  if (!speech || typeof speech !== 'string') {
    return { dry: true, reason: 'Invalid or empty response received.' }
  }
  // Check if turn count exceeded
  if (turnsSinceTopicChange >= maxTurnsPerTopic) {
    return { dry: true, reason: `Reached maximum of ${maxTurnsPerTopic} turns on this topic` }
  }
  
  // Check keywords
  const farewellRegex = /\b(goodbye|farewell|wrap up|conclude|sign off|signing off|thank you for the discussion|until next time|take care)\b/i
  if (farewellRegex.test(speech)) {
    return { dry: true, reason: 'Detected farewell/conclusion keywords in speech' }
  }
  
  // Check short speech
  if (speech.split(/\s+/).length < 5) {
    return { dry: true, reason: 'Response is too short, signaling conversation exhaustion' }
  }
  
  return { dry: false }
}

async function runDebate() {
  console.log(`\n\x1b[36m==================================================\x1b[0m`)
  console.log(`\x1b[36m            OLLAMA CONTINUOUS DEBATE              \x1b[0m`)
  console.log(`\x1b[36m==================================================\x1b[0m`)
  console.log(` Model A:  \x1b[32m${modelA}\x1b[0m`)
  console.log(` Model B:  \x1b[35m${modelB}\x1b[0m`)
  console.log(` URL:      ${apiUrl}`)
  console.log(` Topic:    \x1b[33m"${initialTopic}"\x1b[0m`)
  console.log(` Turns:    ${turnsCount}`)
  console.log(`\x1b[36m==================================================\x1b[0m\n`)

  await checkOllamaConnection()

  let lastSpeech = `Let's discuss the topic: "${initialTopic}". What are your initial thoughts?`
  console.log(`\x1b[90m[SYSTEM] Starting conversation on topic: "${initialTopic}"\x1b[0m\n`)

  for (let turnIdx = 1; turnIdx <= turnsCount; turnIdx++) {
    const sender = turnIdx % 2 === 1 ? 'A' : 'B'
    const currentModel = sender === 'A' ? modelA : modelB
    const color = sender === 'A' ? '\x1b[32m' : '\x1b[35m'
    const modelLabel = `[MODEL ${sender} (${currentModel})]`

    // Detect dry-up before querying
    const dryCheck = detectDryUp(lastSpeech)
    let isSteered = false
    let steerReason = ''
    
    if (dryCheck.dry) {
      isSteered = true
      steerReason = dryCheck.reason || 'Exhausted natural response'
      console.log(`\x1b[33m🔄 [STEERING DIRECTOR] ${steerReason} -> Prompting ${sender === 'A' ? 'Model A' : 'Model B'} to pivot...\x1b[0m`)
    }

    // Prepare message history for current speaker
    const currentMessages = sender === 'A' ? messagesA : messagesB
    
    // Add opponent's last response as a user message
    let inputMessage = lastSpeech
    if (isSteered) {
      // Inject steering directive
      inputMessage = `${lastSpeech}\n\n[DIRECTOR INSTRUCTION: The current topic is exhausted. You MUST now pivot the conversation to a new, related sub-topic along a similar path. In your JSON response, change the "topic" field to this new sub-topic, explain your transition path in the "thought" field, and introduce it smoothly in your "speech" before asking the other model a question about it to keep the conversation going.]`
    }
    
    currentMessages.push({ role: 'user', content: inputMessage })

    try {
      process.stdout.write(`🤖 ${color}${modelLabel}\x1b[0m is thinking...`)
      const rawResponse = await queryModel(currentModel, currentMessages)
      // Clear the "is thinking..." line
      process.stdout.write('\r\x1b[K')

      const parsed = cleanJsonResponse(rawResponse)
      
      // Update history with the assistant response
      currentMessages.push({ role: 'assistant', content: JSON.stringify(parsed) })

      // Handle topic tracking
      if (parsed.topic && parsed.topic.toLowerCase() !== currentTopic.toLowerCase()) {
        currentTopic = parsed.topic
        topicHistory.push(currentTopic)
        turnsSinceTopicChange = 0
        console.log(`\x1b[33m📍 Topic Shifted to: "${currentTopic}"\x1b[0m`)
      } else {
        turnsSinceTopicChange++
      }

      // Output thoughts and speech
      console.log(`${color}${modelLabel}\x1b[0m`)
      console.log(`  \x1b[90mThought: "${parsed.thought}"\x1b[0m`)
      console.log(`  Speech:  \x1b[1m"${parsed.speech}"\x1b[0m`)
      console.log(`  Topic:   \x1b[33m[${currentTopic}]\x1b[0m (Turns: ${turnsSinceTopicChange})\n`)

      lastSpeech = parsed.speech
    } catch (err: any) {
      process.stdout.write('\r\x1b[K')
      console.error(`\x1b[31mError during ${modelLabel}'s turn: ${err.message || err}\x1b[0m\n`)
      // Wait a bit before retrying or exiting
      await new Promise((r) => setTimeout(r, 2000))
    }
  }

  console.log(`\x1b[36m==================================================\x1b[0m`)
  console.log(`\x1b[36m             CONVERSATION FINISHED                \x1b[0m`)
  console.log(`\x1b[36m==================================================\x1b[0m`)
  console.log(` Topic Path History:`)
  topicHistory.forEach((topic, idx) => {
    console.log(`   ${idx + 1}. ${topic}`)
  })
  console.log(`\x1b[36m==================================================\x1b[0m\n`)
}

runDebate()
