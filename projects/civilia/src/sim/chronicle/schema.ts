/**
 * Lightweight manual validator for ChronicleArticle.
 *
 * We avoid pulling in Zod — this covers the required field/type checks
 * plus the bias enum constraint.
 */

import type { ChronicleArticle } from './types'

const VALID_BIASES = ['pro', 'neutral', 'anti'] as const

interface ValidationOk {
  ok: true
  article: ChronicleArticle
}

interface ValidationErr {
  ok: false
  errors: string[]
}

export function validateChronicleArticle(raw: unknown): ValidationOk | ValidationErr {
  const errors: string[] = []

  if (raw === null || raw === undefined || typeof raw !== 'object') {
    return { ok: false, errors: ['expected an object'] }
  }

  const obj = raw as Record<string, unknown>

  // Required string fields
  const requiredStrings: (keyof ChronicleArticle)[] = [
    'id',
    'title',
    'authorFaction',
    'bias',
    'period',
    'content',
  ]
  for (const field of requiredStrings) {
    if (typeof obj[field] !== 'string') {
      errors.push(`"${field}" must be a string`)
    }
  }

  // bias enum check (only if it passed the string check)
  if (
    typeof obj.bias === 'string' &&
    !(VALID_BIASES as readonly string[]).includes(obj.bias)
  ) {
    errors.push(`"bias" must be one of: ${VALID_BIASES.join(', ')}`)
  }

  // sourceEventIds: string[]
  if (!Array.isArray(obj.sourceEventIds)) {
    errors.push('"sourceEventIds" must be an array')
  } else {
    for (let i = 0; i < obj.sourceEventIds.length; i++) {
      if (typeof obj.sourceEventIds[i] !== 'string') {
        errors.push(`"sourceEventIds[${i}]" must be a string`)
      }
    }
  }

  // generatedAt: number
  if (typeof obj.generatedAt !== 'number' || !Number.isFinite(obj.generatedAt)) {
    errors.push('"generatedAt" must be a finite number')
  }

  if (errors.length > 0) {
    return { ok: false, errors }
  }

  return { ok: true, article: obj as unknown as ChronicleArticle }
}
