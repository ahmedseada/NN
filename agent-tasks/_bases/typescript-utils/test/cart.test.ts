import { test } from "node:test";
import assert from "node:assert/strict";
import { addItem, cartTotal, removeItem, type CartItem } from "../src/cart.ts";

const mug: CartItem = { sku: "MUG-1", name: "Mug", unitPrice: 850, quantity: 1 };

test("addItem merges the same SKU and does not mutate", () => {
  const one = addItem([], mug);
  const two = addItem(one, { ...mug, quantity: 2 });
  assert.equal(one[0]?.quantity, 1);
  assert.equal(two.length, 1);
  assert.equal(two[0]?.quantity, 3);
});

test("cartTotal sums in cents", () => {
  assert.equal(cartTotal(addItem([mug], { sku: "PAN", name: "Pan", unitPrice: 3000, quantity: 2 })), 6850);
  assert.equal(cartTotal(removeItem([mug], "MUG-1")), 0);
});
