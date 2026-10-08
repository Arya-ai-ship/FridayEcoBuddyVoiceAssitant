// Pure Chat_Input validation (Req 1.2, 1.3, 1.4, 1.7, 1.9, 3.4).
//
// `validateSubmission(text, waiting)` decides what an Enter press should do. It is pure:
// no DOM, no state. app.js trims nothing itself; it passes the raw textarea value and the
// returned `text` (already trimmed) is what gets sent and appended.
//
// Rules:
//   - Enter while a request is already in flight is ignored (Req 1.9).
//   - Text that is empty or only whitespace is ignored (Req 1.2, 1.4).
//   - Text longer than MAX_TEXT_CHARS after trimming is rejected as "too_long" (Req 1.7).
//   - Otherwise the trimmed text is sent (Req 1.2, 1.3).

import { MAX_TEXT_CHARS } from "./constants.js";

/** @typedef {{action: "send"|"ignore"|"too_long", text: string}} Submission */

/**
 * Classify an attempted Chat_Input submission.
 * @param {string} text the raw Chat_Input value
 * @param {boolean} waiting whether a Backend request is already in flight (Req 1.9)
 * @returns {Submission} the action to take and the trimmed text
 */
export function validateSubmission(text, waiting) {
  const trimmed = String(text ?? "").trim();
  if (waiting) return { action: "ignore", text: trimmed };
  if (trimmed.length === 0) return { action: "ignore", text: trimmed };
  if (trimmed.length > MAX_TEXT_CHARS) return { action: "too_long", text: trimmed };
  return { action: "send", text: trimmed };
}
