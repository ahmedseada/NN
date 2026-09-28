import { test } from "node:test";
import assert from "node:assert/strict";
import { LruCache } from "../src/lru-cache.ts";

test("evicts the least recently used entry", () => {
  const cache = new LruCache<string, number>(2);
  cache.set("a", 1).set("b", 2);
  assert.equal(cache.get("a"), 1);
  cache.set("c", 3);
  assert.equal(cache.has("b"), false);
  assert.deepEqual(cache.keys(), ["a", "c"]);
  assert.equal(cache.size, 2);
});

test("has does not refresh; set of an existing key does", () => {
  const cache = new LruCache<string, number>(2);
  cache.set("a", 1).set("b", 2);
  cache.has("a");
  cache.set("c", 3);
  assert.deepEqual(cache.keys(), ["b", "c"]);
  cache.set("b", 20);
  assert.deepEqual(cache.keys(), ["c", "b"]);
  assert.equal(cache.get("b"), 20);
});

test("stores undefined values and deletes", () => {
  const cache = new LruCache<number, string | undefined>(3);
  cache.set(1, undefined);
  assert.equal(cache.has(1), true);
  assert.equal(cache.delete(1), true);
  assert.equal(cache.delete(1), false);
  assert.equal(cache.get(1), undefined);
  assert.equal(cache.size, 0);
});

test("rejects bad capacities", () => {
  assert.throws(() => new LruCache(0), RangeError);
  assert.throws(() => new LruCache(1.5), RangeError);
});
