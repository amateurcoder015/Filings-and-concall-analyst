import { describe, expect, it } from 'vitest'
import { numberClaims, statusLabel } from './citations'
import type { Claim } from '../types'

const claim = (text: string, status: Claim['status']): Claim => ({
  text, doc_id: 'd', page_no: 1, quote: 'q', status,
})

describe('numberClaims', () => {
  it('numbers claims from 1 in order', () => {
    const numbered = numberClaims([claim('a', 'verified'), claim('b', 'failed')])
    expect(numbered.map((n) => n.n)).toEqual([1, 2])
    expect(numbered[1].claim.text).toBe('b')
  })
  it('handles no claims', () => {
    expect(numberClaims([])).toEqual([])
  })
})

describe('statusLabel', () => {
  it('never presents a failed claim as verified', () => {
    expect(statusLabel('verified')).toBe('Verified in source')
    expect(statusLabel('weak')).toBe('Close match, check the source')
    expect(statusLabel('failed')).toBe('Unverified')
  })
})
