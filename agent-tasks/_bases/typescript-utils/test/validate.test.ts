import { test } from "node:test";
import assert from "node:assert/strict";
import { isEmail, validatePassword } from "../src/validate.ts";

test("isEmail", () => {
  assert.ok(isEmail("a@b.co"));
  assert.ok(!isEmail("a@b"));
});

test("validatePassword lists what is missing", () => {
  assert.deepEqual(validatePassword("abcdefgh1"), { valid: true });
  assert.deepEqual(validatePassword("abc"), { valid: false, errors: ["at least 8 characters", "a digit"] });
});
