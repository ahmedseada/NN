import { test, mock } from "node:test";
import assert from "node:assert/strict";
import { debounce } from "../src/timing.ts";

test("calls once with the latest arguments after the wait", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const fn = mock.fn((a: number, b: string) => `${a}${b}`);
  const d = debounce(fn, 100);
  d(1, "a");
  t.mock.timers.tick(50);
  d(2, "b");
  t.mock.timers.tick(99);
  assert.equal(fn.mock.callCount(), 0);
  t.mock.timers.tick(1);
  assert.equal(fn.mock.callCount(), 1);
  assert.deepEqual(fn.mock.calls[0]?.arguments, [2, "b"]);
});

test("cancel drops the pending call; flush runs it now", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const fn = mock.fn((_: string) => {});
  const d = debounce(fn, 100);
  d("x");
  d.cancel();
  t.mock.timers.tick(200);
  assert.equal(fn.mock.callCount(), 0);
  d("y");
  d.flush();
  assert.deepEqual(fn.mock.calls.map((c) => c.arguments), [["y"]]);
  t.mock.timers.tick(200);
  d.flush();
  assert.equal(fn.mock.callCount(), 1);
});
