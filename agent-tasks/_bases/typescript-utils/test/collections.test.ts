import { test } from "node:test";
import assert from "node:assert/strict";
import { chunk, uniqueBy } from "../src/collections.ts";

test("chunk splits into fixed sizes", () => {
  assert.deepEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
  assert.throws(() => chunk([1], 0), RangeError);
});

test("uniqueBy keeps first occurrences", () => {
  assert.deepEqual(uniqueBy([{ id: 1, v: "a" }, { id: 1, v: "b" }, { id: 2, v: "c" }], (x) => x.id).map((x) => x.v), ["a", "c"]);
});
