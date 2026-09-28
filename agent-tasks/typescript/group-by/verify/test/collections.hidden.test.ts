import { test } from "node:test";
import assert from "node:assert/strict";
import { groupBy, sumBy } from "../src/collections.ts";

type Order = { id: number; status: "open" | "paid"; total: number };
const orders: Order[] = [
  { id: 1, status: "paid", total: 10 },
  { id: 2, status: "open", total: 5 },
  { id: 3, status: "paid", total: 2.5 },
];

test("groupBy keeps first-seen group order and item order", () => {
  const groups: Partial<Record<"open" | "paid", Order[]>> = groupBy(orders, (o) => o.status);
  assert.deepEqual(Object.keys(groups), ["paid", "open"]);
  assert.deepEqual(groups.paid?.map((o) => o.id), [1, 3]);
  assert.deepEqual(groupBy([], (x: number) => x), {});
});

test("groupBy works with number keys", () => {
  assert.deepEqual(groupBy(["a", "bb", "cc"], (s) => s.length), { 1: ["a"], 2: ["bb", "cc"] });
});

test("sumBy", () => {
  assert.equal(sumBy(orders, (o) => o.total), 17.5);
  assert.equal(sumBy([], (x: number) => x), 0);
});
