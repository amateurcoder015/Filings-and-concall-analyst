import { useState } from 'react'
import { ask } from '../api'
import { numberClaims, statusLabel } from '../lib/citations'
import type { AskResponse, Claim } from '../types'

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
      const first = answer.claims[0]
      if (first) onCite(first)
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
                className="block text-left text-[var(--accent)] hover:underline"
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
            {!turn.answer && !turn.error && <p className="text-sm text-[var(--muted)]">Reading the filings…</p>}
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
          className="w-full rounded border border-[var(--line)] bg-white px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
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
        <p className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900">
          Most citations below could not be verified against the source. Treat this answer with caution.
        </p>
      )}
      <p>{answer.summary}</p>
      <ol className="space-y-2">
        {numberClaims(answer.claims).map(({ n, claim }) => (
          <li key={n} className={claim.status === 'failed' ? 'opacity-60' : ''}>
            <button
              className="mr-2 rounded bg-[var(--accent)] px-1.5 text-xs text-white"
              onClick={() => onCite(claim)}
              aria-label={`Open source for claim ${n}`}
            >
              {n}
            </button>
            {claim.text}
            <span className="ml-2 text-xs text-[var(--muted)]">{statusLabel(claim.status)}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
