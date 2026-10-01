export function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/\s+/g, ' ')
    .trim()
}

// PDF text layers split a line into many small items, so an item is highlighted when it
// is a substring of the quote. Items under 4 characters are skipped to avoid noise.
export function shouldHighlight(item: string, quote: string): boolean {
  const text = normalize(item)
  return text.length >= 4 && normalize(quote).includes(text)
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}
