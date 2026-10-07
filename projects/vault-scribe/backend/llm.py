"""
Vault Scribe — LLM Integration
Provides the core Chat and completion functions via OpenAI.
"""

import os
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Dict, Any
import random
from backend.indexer import get_indexer
from backend.config import VAULT_PATH

# Groq is fully compatible with the OpenAI python package, 
# we just change the base URL and the API key!
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY", "NOT_SET"),
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM_PROMPT = """You are Vault Scribe, an elite research assistant managing a personal knowledge graph.
Your job is to interview the user, structure their thoughts, and ultimately generate perfectly formatted Markdown notes that match their vault conventions.

Rules:
1. Ask targeted questions to clarify ambiguities or flesh out details.
2. If the user provides raw data (like a URL dump or brain dump), ask how they want it categorized.
3. If the user tells you to "commit" or "generate the note", respond with a JSON object formatted exactly as:
   ```json
   {
     "action": "COMMIT",
     "title": "Proposed Note Title",
     "content": "The raw markdown content to be saved",
     "template": "reference"
   }
   ```
4. Otherwise, just converse naturally as a helpful scientist.
"""

async def chat_with_scribe(messages: List[Dict[str, str]]) -> str:
    """
    Send a conversation history to the LLM and get the response.
    messages format: [{"role": "user", "content": "hello"}]
    """
    if os.getenv("GROQ_API_KEY", "NOT_SET") == "NOT_SET":
        return "⚠️ GROQ_API_KEY is not set in the .env file. Please add it to continue."

    # Prepend system prompt
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    
    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=full_messages,
        temperature=0.7,
        max_tokens=2000
    )
    
    return response.choices[0].message.content


async def generate_insights() -> str:
    """Select 3 random notes from the vault and ask the LLM to synthesize a connection."""
    if os.getenv("GROQ_API_KEY", "NOT_SET") == "NOT_SET":
        return "⚠️ GROQ_API_KEY is not set in the .env file."
        
    indexer = get_indexer()
    if not indexer.index.notes:
        return "Vault is empty."
        
    # Get 3 random titles
    all_titles = list(indexer.index.notes.keys())
    # Try to pick 3 random ones. If <3, pick what we have.
    sample_size = min(3, len(all_titles))
    chosen_titles = random.sample(all_titles, sample_size)
    
    # Load their contents
    contents = []
    for t in chosen_titles:
        path = indexer.index.notes[t].filepath
        if path.exists():
            # grab first 1000 chars to save context window
            text = path.read_text(encoding="utf-8")[:1000] 
            contents.append(f"--- NOTE: {t} ---\n{text}")
            
    vault_data = "\n\n".join(contents)
    
    prompt = f"""You are Vault Scribe, an elite biological systems thinker.
I am going to provide you with snippets from {sample_size} random notes in my vault.
Your goal is to synthesize a novel, non-obvious, mechanistic connection between them.

{vault_data}

Provide a brilliant 2-3 paragraph insight connecting these concepts. Be specific, scientific, and profound."""

    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=1000
    )
    
    return response.choices[0].message.content


async def check_contradictions(new_content: str, title: str) -> str:
    """Use simple word overlap to find top 3 similar notes, then ask LLM if the new content contradicts them."""
    if os.getenv("GROQ_API_KEY", "NOT_SET") == "NOT_SET":
        return "⚠️ GROQ_API_KEY not set."
        
    indexer = get_indexer()
    if not indexer.index.notes:
        return "No clear contradictions found."
    
    # Simple word-overlap similarity (replacing broken _compute_content_similarity call)
    new_words = set(new_content.lower().split())
    similarities = []
    for t, note in indexer.index.notes.items():
        if t == title:
            continue
        note_words = set(note.content.lower().split())
        if not note_words:
            continue
        overlap = len(new_words.intersection(note_words))
        sim = overlap / max(len(new_words), 1) * 100
        if sim > 10:
            similarities.append((t, note.content, sim))
            
    similarities.sort(key=lambda x: x[2], reverse=True)
    top_notes = similarities[:3]
    
    if not top_notes:
        return "No clear contradictions found."
        
    # 2. Build context
    context = []
    for t, content, _ in top_notes:
        context.append(f"--- ESTABLISHED NOTE: {t} ---\n{content[:1000]}")
    vault_data = "\n\n".join(context)
    
    prompt = f"""You are Vault Scribe, an elite biological reviewer.
I am writing a new note titled "{title}". 
Here is my draft content:
{new_content[:1500]}

Here are snippets from my existing vault notes on related topics:
{vault_data}

Does my draft content directly contradict any concrete claims made in the existing notes?
If YES, concisely explain the contradiction in 1-2 short sentences.
If NO, reply exactly with: "No clear contradictions found."
Do NOT say "No contradictions found" if you find one. Only say that exact phrase if there are none. Be strict."""

    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Contradiction check failed: {str(e)[:100]}"
