import { test } from "node:test";
import assert from "node:assert/strict";
import { slugify } from "../src/format.ts";

test("slugify keeps accented letters as ASCII", () => {
  assert.equal(slugify("Crème Brûlée"), "creme-brulee");
  assert.equal(slugify("Ärger über"), "arger-uber");
  assert.equal(slugify("  Mañana, señor! "), "manana-senor");
  assert.equal(slugify("Ångström"), "angstrom");
});

test("slugify still joins words with single dashes", () => {
  assert.equal(slugify("--Hello   World--"), "hello-world");
  assert.equal(slugify("C# & TypeScript 5.9"), "c-typescript-5-9");
});
