import { useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/TextLayer.css'
import { pdfUrl } from '../api'
import { escapeHtml, shouldHighlight } from '../lib/highlight'
import { statusLabel } from '../lib/citations'
import type { Claim } from '../types'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

export function SourceViewer({ claim }: { claim: Claim | null }) {
  const [loadError, setLoadError] = useState(false)

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
        <div className="mb-1 text-xs uppercase tracking-wide text-[var(--muted)]">
          {claim.doc_id} · page {claim.page_no} · {statusLabel(claim.status)}
        </div>
        <blockquote className="border-l-2 border-[var(--accent)] pl-3">{claim.quote}</blockquote>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-4">
        {loadError ? (
          <p className="text-sm text-red-700">Could not load the PDF. The quote above is the cited passage.</p>
        ) : (
          <Document
            key={claim.doc_id}
            file={pdfUrl(claim.doc_id)}
            onLoadError={() => setLoadError(true)}
            onLoadSuccess={() => setLoadError(false)}
          >
            <Page
              pageNumber={claim.page_no}
              width={520}
              customTextRenderer={({ str }) =>
                shouldHighlight(str, claim.quote) ? `<mark>${escapeHtml(str)}</mark>` : escapeHtml(str)
              }
            />
          </Document>
        )}
      </div>
    </aside>
  )
}
