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

// Short FNV-1a hash, so a second claim on the same page gets a fresh PdfPage and re-highlights.
export function quoteKey(quote: string): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < quote.length; i++) {
    hash ^= quote.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16)
}
