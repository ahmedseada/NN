import { test } from "node:test";
import assert from "node:assert/strict";
import { retry } from "../src/timing.ts";

test("retries with exponential back-off until it succeeds", async () => {
  const times: number[] = [];
  const start = Date.now();
  const result = await retry(async (attempt) => {
    times.push(Date.now() - start);
    if (attempt < 3) throw new Error(`fail ${attempt}`);
    return attempt * 10;
  }, { retries: 5, delayMs: 20 });
  assert.equal(result, 30);
  assert.equal(times.length, 3);
  assert.ok((times[1] ?? 0) >= 15 && (times[2] ?? 0) - (times[1] ?? 0) >= 35, `waits ${times.join(", ")}`);
});

test("rejects with the last error after the retries", async () => {
  let calls = 0;
  await assert.rejects(retry(async (attempt) => { calls++; throw new Error(`fail ${attempt}`); }, { retries: 2, delayMs: 1 }), /fail 3/);
  assert.equal(calls, 3);
});

test("stops when aborted, before or while waiting", async () => {
  const early = new AbortController();
  early.abort(new Error("stopped early"));
  let calls = 0;
  await assert.rejects(retry(async () => { calls++; return 1; }, { retries: 1, delayMs: 1, signal: early.signal }), /stopped early/);
  assert.equal(calls, 0);

  const later = new AbortController();
  const started = Date.now();
  setTimeout(() => later.abort(new Error("stopped")), 30);
  await assert.rejects(retry(async () => { throw new Error("fail"); }, { retries: 5, delayMs: 10_000, signal: later.signal }), /stopped/);
  assert.ok(Date.now() - started < 2000);
});
