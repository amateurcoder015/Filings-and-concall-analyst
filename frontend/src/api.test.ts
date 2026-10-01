import { describe, expect, it } from 'vitest'
import { errorMessage } from './api'

describe('errorMessage', () => {
  it('uses a string detail', () => {
    expect(errorMessage({ detail: 'Bad period' }, 400)).toBe('Bad period')
  })
  it('joins msg fields of an array detail', () => {
    expect(errorMessage({ detail: [{ msg: 'too long' }, { msg: 'required' }] }, 422)).toBe('too long; required')
  })
  it('stringifies other detail shapes', () => {
    expect(errorMessage({ detail: { a: 1 } }, 400)).toBe('{"a":1}')
  })
  it('falls back to the status', () => {
    expect(errorMessage({}, 500)).toBe('Request failed (500)')
    expect(errorMessage(null, 502)).toBe('Request failed (502)')
  })
})
