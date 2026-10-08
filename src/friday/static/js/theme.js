/**
 * Theme cycling: Light → Dawn → Dark → Light …
 *
 * The active theme is stored as data-theme on <html> and persisted in
 * localStorage so it survives page reloads. "dark" is the default when no
 * preference is saved (matches the :root defaults in styles.css).
 *
 * Each theme is a plain object so this module stays pure (no DOM side-effects
 * at import time) and is easy to unit-test.
 */

/** @typedef {"dark" | "dawn" | "light"} Theme */

const STORAGE_KEY = "friday-theme";

/** Ordered cycle: pressing the button advances to the next entry. */
const CYCLE = /** @type {Theme[]} */ (["light", "dawn", "dark"]);

/**
 * Human label and icon shown ON the button (describes the mode you'll switch
 * TO, matching the convention used by most apps).
 * @type {Record<Theme, { label: string; icon: string }>}
 */
const BUTTON_META = {
  light: { icon: "🌅", label: "Dawn mode" },
  dawn:  { icon: "🌙", label: "Dark mode" },
  dark:  { icon: "☀️",  label: "Light mode" },
};

/** Return the next theme in the cycle. */
export function nextTheme(current) {
  const idx = CYCLE.indexOf(current);
  return CYCLE[(idx + 1) % CYCLE.length];
}

/** Load the saved theme, falling back to "dark". */
export function loadTheme() {
  const saved = globalThis.localStorage?.getItem(STORAGE_KEY);
  return CYCLE.includes(/** @type {Theme} */ (saved)) ? /** @type {Theme} */ (saved) : "dark";
}

/** Persist and apply a theme to the document. */
export function applyTheme(theme, doc = globalThis.document) {
  doc.documentElement.setAttribute("data-theme", theme);
  globalThis.localStorage?.setItem(STORAGE_KEY, theme);
}

/** Update the toggle button's icon and accessible label. */
export function updateButton(btn, currentTheme) {
  const meta = BUTTON_META[currentTheme];
  btn.setAttribute("aria-label", `Switch to ${meta.label}`);
  btn.title = `Switch to ${meta.label}`;
  const icon = btn.querySelector(".theme-icon");
  if (icon) icon.textContent = meta.icon;
}
