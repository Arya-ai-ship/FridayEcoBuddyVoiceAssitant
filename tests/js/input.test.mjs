// Tests for the pure Chat_Input validator (Req 1.2, 1.4, 1.7, 1.9).

import assert from "node:assert/strict";
import { test } from "node:test";

import { validateSubmission } from "../../src/friday/static/js/input.js";
import { MAX_TEXT_CHARS } from "../../src/friday/static/js/constants.js";

test("empty text is ignored", () => {
  assert.deepEqual(validateSubmission("", false), { action: "ignore", text: "" });
});

test("whitespace-only text is ignored and trimmed", () => {
  assert.deepEqual(validateSubmission("   \n\t ", false), { action: "ignore", text: "" });
});

test("a request already waiting ignores Enter (Req 1.9)", () => {
  assert.deepEqual(validateSubmission("hello", true), { action: "ignore", text: "hello" });
});

test("ordinary text is sent trimmed", () => {
  assert.deepEqual(validateSubmission("  pull inflation  ", false), {
    action: "send",
    text: "pull inflation",
  });
});

test("exactly MAX_TEXT_CHARS after trim is sent", () => {
  const text = "a".repeat(MAX_TEXT_CHARS);
  assert.deepEqual(validateSubmission(text, false), { action: "send", text });
});

test("MAX_TEXT_CHARS + 1 after trim is too_long (Req 1.7)", () => {
  const text = "a".repeat(MAX_TEXT_CHARS + 1);
  assert.deepEqual(validateSubmission(text, false), { action: "too_long", text });
});

test("surrounding whitespace does not count toward the limit", () => {
  const core = "a".repeat(MAX_TEXT_CHARS);
  const result = validateSubmission(`   ${core}   `, false);
  assert.equal(result.action, "send");
  assert.equal(result.text, core);
});
