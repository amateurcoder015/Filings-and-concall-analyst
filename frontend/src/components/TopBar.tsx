interface Props {
  company: string
  periods: string[]
  period: string | null
  onPeriod: (period: string | null) => void
}

export function TopBar({ company, periods, period, onPeriod }: Props) {
  return (
    <header className="flex items-center gap-4 border-b border-[var(--line)] px-5 py-3">
      <h1 className="text-base font-semibold tracking-tight">{company}</h1>
      <label className="flex items-center gap-2 text-sm text-[var(--muted)]">
        Period
        <select
          className="rounded border border-[var(--line)] bg-white px-2 py-1 text-[var(--ink)]"
          value={period ?? ''}
          onChange={(e) => onPeriod(e.target.value || null)}
        >
          <option value="">All periods</option>
          {periods.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </label>
    </header>
  )
}
