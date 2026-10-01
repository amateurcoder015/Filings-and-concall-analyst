import { useEffect, useState } from 'react'
import { fetchDocuments } from './api'
import { BriefingPane } from './components/BriefingPane'
import { ChatPane } from './components/ChatPane'
import { SourceViewer } from './components/SourceViewer'
import { TopBar } from './components/TopBar'
import type { Claim, DocumentsResponse } from './types'

export default function App() {
  const [docs, setDocs] = useState<DocumentsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [period, setPeriod] = useState<string | null>(null)
  const [active, setActive] = useState<Claim | null>(null)

  useEffect(() => {
    fetchDocuments().then(setDocs).catch((e: Error) => setError(e.message))
  }, [])

  if (error) return <p className="p-6 text-red-700">{error}</p>
  if (!docs) return <p className="p-6 text-[var(--muted)]">Loading…</p>

  const periods = [...new Set(docs.documents.map((d) => d.period))]
  return (
    <div className="flex h-screen flex-col">
      <TopBar company={docs.company} periods={periods} period={period} onPeriod={setPeriod} />
      <div className="grid min-h-0 flex-1 grid-cols-[260px_minmax(0,1fr)_minmax(0,1fr)]">
        <BriefingPane documents={docs.documents} />
        <ChatPane period={period} onCite={setActive} />
        <SourceViewer claim={active} />
      </div>
    </div>
  )
}
