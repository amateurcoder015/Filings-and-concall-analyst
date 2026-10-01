import type { DocumentInfo } from '../types'

export function BriefingPane({ documents }: { documents: DocumentInfo[] }) {
  return (
    <aside className="flex flex-col gap-6 overflow-y-auto border-r border-[var(--line)] p-5 text-sm">
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">Loaded filings</h2>
        <ul className="flex flex-col gap-2">
          {documents.map((d) => (
            <li key={d.doc_id}>
              <div className="font-medium">{d.title}</div>
              <div className="text-[var(--muted)]">{d.period} · {d.pages} pages</div>
            </li>
          ))}
        </ul>
      </section>
      <section className="text-[var(--muted)]">
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide">Briefing</h2>
        <p>What changed, red flags and management tone will appear here in a later version.</p>
      </section>
    </aside>
  )
}
