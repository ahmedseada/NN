import { test } from "node:test";
import assert from "node:assert/strict";
import { Emitter } from "../src/events.ts";

test("emitter delivers typed payloads and unsubscribes", () => {
  const emitter = new Emitter<{ added: { sku: string }; cleared: undefined }>();
  const seen: string[] = [];
  const off = emitter.on("added", (p) => seen.push(p.sku));
  emitter.emit("added", { sku: "A" });
  off();
  emitter.emit("added", { sku: "B" });
  assert.deepEqual(seen, ["A"]);
  assert.equal(emitter.listenerCount("added"), 0);
});
