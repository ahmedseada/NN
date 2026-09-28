export interface CartItem {
  readonly sku: string;
  readonly name: string;
  /** Price of one unit in cents. */
  readonly unitPrice: number;
  readonly quantity: number;
}

/** A new list with `item` added; an item with the same SKU gets its quantity increased. */
export function addItem(items: readonly CartItem[], item: CartItem): CartItem[] {
  if (item.quantity <= 0) {
    throw new RangeError("quantity must be positive");
  }

  const existing = items.find((i) => i.sku === item.sku);
  if (!existing) {
    return [...items, item];
  }

  return items.map((i) => (i.sku === item.sku ? { ...i, quantity: i.quantity + item.quantity } : i));
}

export function removeItem(items: readonly CartItem[], sku: string): CartItem[] {
  return items.filter((i) => i.sku !== sku);
}

/** Total in cents. */
export function cartTotal(items: readonly CartItem[]): number {
  return items.reduce((sum, i) => sum + i.unitPrice * i.quantity, 0);
}
