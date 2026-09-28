export type ValidationResult = { valid: true } | { valid: false; errors: string[] };

const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function isEmail(value: string): boolean {
  return emailPattern.test(value);
}

/** At least 8 characters with a digit and a letter. */
export function validatePassword(password: string): ValidationResult {
  const errors: string[] = [];
  if (password.length < 8) errors.push("at least 8 characters");
  if (!/\d/.test(password)) errors.push("a digit");
  if (!/[a-z]/i.test(password)) errors.push("a letter");
  return errors.length === 0 ? { valid: true } : { valid: false, errors };
}
