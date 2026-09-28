/** Formats an amount in cents, e.g. formatPrice(1250, "EUR", "en-US") → "€12.50". */
export function formatPrice(cents: number, currency: string, locale = "en-US"): string {
  if (!Number.isInteger(cents)) {
    throw new RangeError(`cents must be an integer, got ${cents}`);
  }

  return new Intl.NumberFormat(locale, { style: "currency", currency }).format(cents / 100);
}

/** A URL slug: lower case, words joined by single dashes. */
export function slugify(text: string): string {
  return text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

/** Shortens text to at most `max` characters, ending with "…" when cut. */
export function truncate(text: string, max: number): string {
  if (max < 1) {
    throw new RangeError("max must be at least 1");
  }

  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}
