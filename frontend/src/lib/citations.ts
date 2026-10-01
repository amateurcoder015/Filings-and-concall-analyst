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
    case 'failed':
      return 'Unverified'
  }
}
