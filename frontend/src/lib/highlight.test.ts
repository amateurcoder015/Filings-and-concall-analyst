import { describe, expect, it } from 'vitest'
import { escapeHtml, normalize, shouldHighlight } from './highlight'

describe('normalize', () => {
  it('lowercases, unifies curly quotes and collapses whitespace', () => {
    expect(normalize('  Clients’   BUDGETS \n tight ')).toBe("clients' budgets tight")
  })
})

describe('shouldHighlight', () => {
  const quote = 'Operating margin was 21.1% in the second quarter'
  it('highlights text items that are part of the quote', () => {
    expect(shouldHighlight('Operating margin was', quote)).toBe(true)
    expect(shouldHighlight('21.1%', quote)).toBe(true)
  })
  it('ignores items that are not in the quote', () => {
    expect(shouldHighlight('Revenue grew', quote)).toBe(false)
  })
  it('ignores tiny fragments that would highlight everywhere', () => {
    expect(shouldHighlight('a', quote)).toBe(false)
    expect(shouldHighlight(' ', quote)).toBe(false)
  })
})

describe('escapeHtml', () => {
  it('escapes markup so PDF text cannot inject HTML', () => {
    expect(escapeHtml('<b>"A&B"</b>')).toBe('&lt;b&gt;&quot;A&amp;B&quot;&lt;/b&gt;')
  })
})
