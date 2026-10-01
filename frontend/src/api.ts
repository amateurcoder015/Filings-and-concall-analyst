import type { AskResponse, DocumentsResponse } from './types'

export async function fetchDocuments(): Promise<DocumentsResponse> {
  const response = await fetch('/api/documents')
  if (!response.ok) throw new Error('Could not load documents. Is the backend running?')
  return response.json()
}

export async function ask(question: string, period: string | null): Promise<AskResponse> {
  const response = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, period }),
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed (${response.status})`)
  }
  return response.json()
}

export const pdfUrl = (docId: string) => `/api/pdf/${encodeURIComponent(docId)}`
