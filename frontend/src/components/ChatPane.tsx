import { useState } from 'react'
import { ask } from '../api'
import { autoOpenClaim, numberClaims, statusLabel, statusTone, summaryNotice } from '../lib/citations'
import type { AskResponse, Claim, ClaimStatus } from '../types'

interface Turn {
  question: string
  answer?: AskResponse
  error?: string
}

const SUGGESTIONS = [
  'What drove the change in operating margin this quarter?',
  'What did management say about the demand outlook?',
  'What is the guidance for the full year?',
]

interface Props {
  period: string | null
  onCite: (claim: Claim) => void
}

export function ChatPane({ period, onCite }: Props) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(question: string) {
    const trimmed = question.trim()
    if (!trimmed || busy) return
    setDraft('')
    setBusy(true)
    setTurns((t) => [...t, { question: trimmed }])
    try {
      const answer = await ask(trimmed, period)
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, answer } : turn)))
      const firstVerified = autoOpenClaim(answer.claims)
      if (firstVerified) onCite(firstVerified)
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Something went wrong.'
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, error } : turn)))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="flex min-h-0 flex-col border-r border-[var(--line)]">
      <div className="flex-1 space-y-6 overflow-y-auto p-5">
        {turns.length === 0 && (
          <div className="space-y-2 text-sm">
            <p className="text-[var(--muted)]">Ask about the loaded filings. Try:</p>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                className="block text-left text-[var(--accent)] hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
                onClick={() => submit(s)}
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn, i) => (
          <div key={i} className="space-y-2">
            <p className="font-medium">{turn.question}</p>
            {!turn.answer && !turn.error && <p role="status" className="text-sm text-[var(--muted)]">Reading the filings…</p>}
            {turn.error && <p className="text-sm text-red-700">{turn.error}</p>}
            {turn.answer && <AnswerView answer={turn.answer} onCite={onCite} />}
          </div>
        ))}
      </div>
      <form
        className="border-t border-[var(--line)] p-4"
        onSubmit={(e) => {
          e.preventDefault()
          submit(draft)
        }}
      >
        <input
          aria-label="Ask a question about the filings"
          className="w-full rounded border border-[var(--line)] bg-white px-3 py-2 text-sm focus:border-[var(--accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
          placeholder="Ask about this filing…"
          value={draft}
          maxLength={1000}
          onChange={(e) => setDraft(e.target.value)}
          disabled={busy}
        />
        <p className="mt-2 text-xs text-[var(--muted)]">
          For research and education only. Not investment advice.
        </p>
      </form>
    </main>
  )
}

function AnswerView({ answer, onCite }: { answer: AskResponse; onCite: (claim: Claim) => void }) {
  if (answer.not_found) {
    return <p className="text-sm text-[var(--muted)]">{answer.summary}</p>
  }
  return (
    <div className="space-y-3 text-sm">
      {answer.mostly_unverified && (
        <p role="alert" className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900">
          Most citations below could not be verified against the source. Treat this answer with caution.
        </p>
      )}
      <div className="space-y-1">
        <p className="text-xs uppercase tracking-wide text-[var(--muted)]">Model-written summary</p>
        {summaryNotice(answer.summary_supported) && (
          <p role="status" className="rounded bg-amber-100 px-3 py-1 text-xs text-amber-900">
            {summaryNotice(answer.summary_supported)}
          </p>
        )}
        <p>{answer.summary}</p>
      </div>
      <ol className="space-y-2">
        {numberClaims(answer.claims).map(({ n, claim }) => (
          <li key={n} className={claim.status === 'failed' ? 'text-[var(--muted)]' : ''}>
            <button
              className="mr-2 rounded bg-[var(--accent)] px-1.5 text-xs text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
              onClick={() => onCite(claim)}
              aria-label={`Open source for claim ${n}`}
            >
              {n}
            </button>
            {claim.text}
            <StatusBadge status={claim.status} />
          </li>
        ))}
      </ol>
    </div>
  )
}

const TONE_CLASS = {
  ok: 'border border-[var(--line)] text-[var(--muted)]',
  caution: 'bg-amber-100 text-amber-900',
  bad: 'bg-red-100 text-red-900',
} as const

export function StatusBadge({ status }: { status: ClaimStatus }) {
  const text = status === 'verified' ? 'Verified' : statusLabel(status)
  return (
    <span className={`ml-2 whitespace-nowrap rounded px-1.5 py-0.5 text-xs ${TONE_CLASS[statusTone(status)]}`}>
      {text}
    </span>
  )
}
