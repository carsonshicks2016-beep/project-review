export interface ChronicleArticle {
  id: string                      // UUID
  title: string                   // Generated headline
  authorFaction: string           // Faction that produced the article
  bias: 'pro' | 'neutral' | 'anti' // Narrative bias relative to the faction
  period: string                  // e.g. "week-12"
  sourceEventIds: string[]        // IDs of events that informed this article
  content: string                 // Full LLM-generated text
  generatedAt: number             // Unix timestamp (ms)
}

export interface ChronicleConfig {
  enabled: boolean
  cadence: 'weekly' | 'monthly' | 'seasonal'
  consequenceThreshold: number    // min consequenceLevel to be "notable" (default 1)
  maxArticlesPerPeriod: number    // cap (default 5)
}

export const DEFAULT_CHRONICLE_CONFIG: ChronicleConfig = {
  enabled: true,
  cadence: 'weekly',
  consequenceThreshold: 1,
  maxArticlesPerPeriod: 5,
}
