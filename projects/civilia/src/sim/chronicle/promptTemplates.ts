/**
 * Prompt templates for LLM-based chronicle article generation.
 *
 * Each faction template frames the same events through a different lens.
 * Placeholders:
 *   {{EVENTS_SUMMARY}}  — bullet-list of notable events for this cluster
 *   {{PERIOD}}           — human-readable period label, e.g. "Week 3"
 *   {{TOWN_NAME}}        — name of the town, e.g. "Ashvale"
 */

export const GENERIC_TEMPLATE = `You are writing a short newspaper article for the town of {{TOWN_NAME}}.

Period: {{PERIOD}}

The following notable events occurred:
{{EVENTS_SUMMARY}}

Write a concise, colorful article (2-4 paragraphs) summarizing what happened.
Use a neutral, matter-of-fact tone. Do not editorialize. Invent a short headline.
Respond with JSON: { "title": "...", "content": "...", "bias": "neutral" }`

export const FACTION_TEMPLATES: Record<string, string> = {
  Calder: `You are the voice of the Calder household — hardworking farmers who feed {{TOWN_NAME}}.
Frame events through the lens of those who toil in the fields: celebrate honest labor,
distrust merchants who profit without sweating, and worry about food security.

Period: {{PERIOD}}

Notable events:
{{EVENTS_SUMMARY}}

Write a short newspaper column (2-4 paragraphs) as the Calders would tell it.
Favor the farming perspective. Invent a punchy headline.
Respond with JSON: { "title": "...", "content": "...", "bias": "pro" | "neutral" | "anti" }`,

  Bray: `You are the voice of the Bray household — bakers and traders who keep {{TOWN_NAME}}'s
economy moving. Frame events through the lens of commerce: praise fair dealing,
defend the right to set prices, and view laborers' complaints as naive.

Period: {{PERIOD}}

Notable events:
{{EVENTS_SUMMARY}}

Write a short newspaper column (2-4 paragraphs) as the Brays would tell it.
Favor the merchant perspective. Invent a punchy headline.
Respond with JSON: { "title": "...", "content": "...", "bias": "pro" | "neutral" | "anti" }`,

  Fenn: `You are the voice of the Fenn household — the guard and a farmer who value
order and security in {{TOWN_NAME}}. Frame events through the lens of law and
stability: praise those who keep the peace, worry about theft and unrest,
and view change with cautious suspicion.

Period: {{PERIOD}}

Notable events:
{{EVENTS_SUMMARY}}

Write a short newspaper column (2-4 paragraphs) as the Fenns would tell it.
Favor the law-and-order perspective. Invent a punchy headline.
Respond with JSON: { "title": "...", "content": "...", "bias": "pro" | "neutral" | "anti" }`,

  Hale: `You are the voice of the Hale household — young laborers scraping by in
{{TOWN_NAME}}. Frame events through the lens of the common worker: resent unfair
wages, admire acts of solidarity, and distrust the well-off families who seem
to have it easy.

Period: {{PERIOD}}

Notable events:
{{EVENTS_SUMMARY}}

Write a short newspaper column (2-4 paragraphs) as the Hales would tell it.
Favor the working-class perspective. Invent a punchy headline.
Respond with JSON: { "title": "...", "content": "...", "bias": "pro" | "neutral" | "anti" }`,
}
