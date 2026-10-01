export type ClaimStatus = 'verified' | 'weak' | 'failed'

export interface Claim {
  text: string
  doc_id: string
  page_no: number
  quote: string
  status: ClaimStatus
}

export interface AskResponse {
  summary: string
  not_found: boolean
  mostly_unverified: boolean
  disclaimer: string
  claims: Claim[]
}

export interface DocumentInfo {
  doc_id: string
  title: string
  doc_type: string
  period: string
  pages: number
}

export interface DocumentsResponse {
  company: string
  documents: DocumentInfo[]
}
