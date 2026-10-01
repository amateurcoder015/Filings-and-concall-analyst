import { useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/TextLayer.css'
import { pdfUrl } from '../api'
import { escapeHtml, shouldHighlight } from '../lib/highlight'
import { statusLabel } from '../lib/citations'
import { StatusBadge } from './ChatPane'
import type { Claim } from '../types'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

const NOTICE: Record<'weak' | 'failed', string> = {
  failed: 'Unverified: the quote could not be matched to this page',
  weak: 'Close match: check the page',
}

export function SourceViewer({ claim }: { claim: Claim | null }) {
  const [retry, setRetry] = useState(0)

  if (!claim) {
    return (
      <aside className="p-5 text-sm text-[var(--muted)]">
        Click a citation number to see the source page here, with the quoted passage highlighted.
      </aside>
    )
  }
  return (
    <aside className="flex min-h-0 flex-col">
      <div className="border-b border-[var(--line)] p-4 text-sm">
        <div className="mb-2 text-xs uppercase tracking-wide text-[var(--muted)]">
          {claim.doc_id} · page {claim.page_no}
          <StatusBadge status={claim.status} />
        </div>
        {claim.status !== 'verified' && (
          <p
            role="note"
            className={`mb-2 rounded px-3 py-2 ${
              claim.status === 'failed' ? 'bg-red-100 text-red-900' : 'bg-amber-100 text-amber-900'
            }`}
          >
            {NOTICE[claim.status]}
          </p>
        )}
        <blockquote className="border-l-2 border-[var(--accent)] pl-3">{claim.quote}</blockquote>
        <span className="sr-only">{statusLabel(claim.status)}</span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-4">
        <PdfPage
          key={`${claim.doc_id}:${claim.page_no}:${retry}`}
          claim={claim}
          onRetry={() => setRetry((r) => r + 1)}
        />
      </div>
    </aside>
  )
}

function PdfPage({ claim, onRetry }: { claim: Claim; onRetry: () => void }) {
  const [error, setError] = useState<string | null>(null)

  if (error) {
    return (
      <div className="space-y-2 text-sm text-red-700" role="alert">
        <p>{error}</p>
        <button
          className="rounded border border-[var(--line)] bg-white px-3 py-1 text-[var(--ink)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
          onClick={onRetry}
        >
          Retry
        </button>
      </div>
    )
  }
  const pageError = () => setError(`Page ${claim.page_no} could not be shown. The cited passage is above.`)
  return (
    <Document
      file={pdfUrl(claim.doc_id)}
      onLoadError={() => setError('Could not load the PDF. The cited passage is above.')}
    >
      <Page
        pageNumber={claim.page_no}
        width={520}
        onLoadError={pageError}
        onRenderError={pageError}
        customTextRenderer={({ str }) =>
          shouldHighlight(str, claim.quote) ? `<mark>${escapeHtml(str)}</mark>` : escapeHtml(str)
        }
      />
    </Document>
  )
}
