import type { Claim, ClaimStatus } from '../types'

export interface NumberedClaim {
  n: number
  claim: Claim
}

export function numberClaims(claims: Claim[]): NumberedClaim[] {
  return claims.map((claim, i) => ({ n: i + 1, claim }))
}

export function statusLabel(status: ClaimStatus): string {
  switch (status) {
    case 'verified':
      return 'Verified in source'
    case 'weak':
      return 'Close match, check the source'
    case 'unsupported':
      return 'Figure not in the quoted passage'
    case 'failed':
      return 'Unverified'
  }
}

export type StatusTone = 'ok' | 'caution' | 'bad'

export function statusTone(status: ClaimStatus): StatusTone {
  switch (status) {
    case 'verified':
      return 'ok'
    case 'weak':
    case 'unsupported':
      return 'caution'
    case 'failed':
      return 'bad'
  }
}

// Only a claim that passed every check is opened automatically.
export function autoOpenClaim(claims: Claim[]): Claim | null {
  return claims.find((c) => c.status === 'verified') ?? null
}

export function summaryNotice(summarySupported: boolean): string | null {
  return summarySupported ? null : 'Figures in this summary are not confirmed by the cited passages.'
}
