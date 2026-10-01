import type { AskResponse, DocumentsResponse } from './types'

export async function fetchDocuments(): Promise<DocumentsResponse> {
  const response = await fetch('/api/documents')
  if (!response.ok) throw new Error('Could not load documents. Is the backend running?')
  return response.json()
}

export function errorMessage(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (detail === undefined || detail === null) return `Request failed (${status})`
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const msgs = detail.map((d) => (d && typeof d === 'object' && 'msg' in d ? String(d.msg) : JSON.stringify(d)))
    return msgs.join('; ')
  }
  return JSON.stringify(detail)
}

export async function ask(question: string, period: string | null): Promise<AskResponse> {
  const response = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, period }),
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(errorMessage(body, response.status))
  }
  return response.json()
}

export const pdfUrl = (docId: string) => `/api/pdf/${encodeURIComponent(docId)}`
