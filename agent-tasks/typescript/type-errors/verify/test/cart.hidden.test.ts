import { test } from "node:test";
import assert from "node:assert/strict";
import { addItem, cartTotal, mostExpensive, type CartItem } from "../src/cart.ts";

const mug: CartItem = { sku: "MUG", name: "Mug", unitPrice: 850, quantity: 2 };
const pan: CartItem = { sku: "PAN", name: "Pan", unitPrice: 1500, quantity: 1 };

test("mostExpensive picks the largest line total", () => {
  assert.equal(mostExpensive([mug, pan])?.sku, "MUG");
  assert.equal(mostExpensive([]), undefined);
});

test("totals and merging still work", () => {
  assert.equal(cartTotal([mug, pan]), 3200);
  assert.equal(cartTotal([]), 0);
  assert.equal(addItem([mug, pan], { ...pan, quantity: 2 })[1]?.quantity, 3);
});
