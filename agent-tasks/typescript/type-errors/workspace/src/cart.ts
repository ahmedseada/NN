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

  const index = items.findIndex((i) => i.sku === item.sku);
  if (index < 0) {
    return [...items, item];
  }

  const copy = [...items];
  copy[index] = { ...items[index], quantity: items[index].quantity + item.quantity };
  return copy;
}

export function removeItem(items: readonly CartItem[], sku: string): CartItem[] {
  return items.filter((i) => i.sku !== sku);
}

/** Total in cents. */
export function cartTotal(items: readonly CartItem[]): number {
  return items.reduce((sum, i) => sum + i.unitPrice * i.quantity);
}

/** The most expensive line (by line total), or undefined for an empty cart. */
export function mostExpensive(items: readonly CartItem[]): CartItem | undefined {
  let best: CartItem;
  for (const item of items) {
    if (!best || item.unitPrice * item.quantity > best.unitPrice * best.quantity) {
      best = item;
    }
  }

  return best;
}
