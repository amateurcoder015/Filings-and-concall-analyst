import { describe, expect, it } from 'vitest'
import { autoOpenClaim, numberClaims, statusLabel, statusTone, summaryNotice } from './citations'
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
  it('labels a claim whose figure is not in its quote', () => {
    expect(statusLabel('unsupported')).toBe('Figure not in the quoted passage')
  })
})

describe('statusTone', () => {
  it('maps statuses to tones', () => {
    expect(statusTone('verified')).toBe('ok')
    expect(statusTone('weak')).toBe('caution')
    expect(statusTone('failed')).toBe('bad')
  })
  it('shows unsupported claims with the caution tone', () => {
    expect(statusTone('unsupported')).toBe('caution')
  })
})

describe('autoOpenClaim', () => {
  it('opens only the first verified claim, never an unsupported, weak or failed one', () => {
    const claims = [claim('a', 'unsupported'), claim('b', 'weak'), claim('c', 'verified'), claim('d', 'verified')]
    expect(autoOpenClaim(claims)?.text).toBe('c')
    expect(autoOpenClaim([claim('a', 'unsupported'), claim('b', 'failed')])).toBeNull()
    expect(autoOpenClaim([])).toBeNull()
  })
})

describe('summaryNotice', () => {
  it('warns only when summary figures are not confirmed', () => {
    expect(summaryNotice(false)).toBe('Figures in this summary are not confirmed by the cited passages.')
    expect(summaryNotice(true)).toBeNull()
  })
})
