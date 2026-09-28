import { test } from "node:test";
import assert from "node:assert/strict";
import { formatPrice, slugify, truncate } from "../src/format.ts";

test("formatPrice formats cents in the currency", () => {
  assert.equal(formatPrice(1250, "EUR", "en-US"), "€12.50");
  assert.throws(() => formatPrice(1.5, "EUR"), RangeError);
});

test("slugify joins lower-case words with dashes", () => {
  assert.equal(slugify("  Hello, World!  "), "hello-world");
});

test("truncate cuts with an ellipsis", () => {
  assert.equal(truncate("abcdef", 4), "abc…");
  assert.equal(truncate("abc", 4), "abc");
});
