// Chat_Window rendering: user and Friday messages, status chips, Preview_Table with its
// CSV link, stats table, inline chart, and the "Audio unavailable" notice
// (Req 1.5, 1.6, 4.8, 7.1-7.4, 7.7-7.9, 9.4, 11.6).
//
// Every node is built with createElement and textContent (never innerHTML), so model and
// Backend text cannot inject markup and the CSP holds. CSV links are intercepted: a 200
// triggers a Blob download; a 404 appends the "no longer available" message (Req 7.8).

const CSV_UNAVAILABLE = "That dataset is no longer available — please fetch it again, Boss.";
const AUDIO_UNAVAILABLE = "Audio unavailable — showing the text instead.";
const EMPTY_PREVIEW = (indicator) => `No data rows were returned for ${indicator}.`;

export class ChatView {
  /**
   * @param {HTMLElement} logEl the chat `<section role="log">`
   * @param {Object} [deps]
   * @param {typeof fetch} [deps.fetchImpl]
   * @param {Document} [deps.doc]
   */
  constructor(logEl, deps = {}) {
    this._log = logEl;
    this._fetch = deps.fetchImpl ?? globalThis.fetch?.bind(globalThis);
    this._doc = deps.doc ?? globalThis.document;
    /** @type {Map<string, HTMLElement>} the current Friday message block per turn */
    this._friday = null;
  }

  _el(tag, className, text) {
    const node = this._doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  _append(node) {
    this._log.appendChild(node);
    node.scrollIntoView?.({ block: "end" });
  }

  /** Append a user message and clear any open Friday block (Req 1.5). */
  addUserMessage(text) {
    const msg = this._el("div", "message user");
    msg.appendChild(this._el("span", "label", "Boss"));
    msg.appendChild(this._el("div", "text", text));
    this._append(msg);
    this._friday = null;
  }

  _fridayBlock() {
    if (!this._friday) {
      const msg = this._el("div", "message friday");
      msg.appendChild(this._el("span", "label", "Friday"));
      this._friday = msg;
      this._append(msg);
    }
    return this._friday;
  }

  /** Render one NDJSON event (Req 1.6, 4.x, 7.x, 9.4, 11.6). */
  handleEvent(event) {
    switch (event.type) {
      case "status":
        this._status(event);
        break;
      case "dataset_preview":
        this._preview(event);
        break;
      case "stats_table":
        this._stats(event);
        break;
      case "chart":
        this._chart(event);
        break;
      case "final":
        this._final(event);
        break;
      default:
        break;
    }
  }

  _status(event) {
    const block = this._fridayBlock();
    const chip = this._el("div", "status-chip", event.text);
    if (event.phase === "tool_error") chip.classList.add("error");
    if (event.audio_error) this._noticeInline(block);
    block.appendChild(chip);
    this._scroll();
  }

  _preview(event) {
    const block = this._fridayBlock();
    if (event.empty) {
      block.appendChild(this._el("div", "text", EMPTY_PREVIEW(event.indicator)));
      this._scroll();
      return;
    }
    const wrap = this._el("div", "preview");
    const table = this._el("table", "data-table");
    const head = this._doc.createElement("tr");
    for (const col of event.columns) head.appendChild(this._el("th", null, col));
    table.appendChild(head);
    for (const row of event.rows) {
      const tr = this._doc.createElement("tr");
      for (const cell of row) tr.appendChild(this._el("td", null, cell));
      table.appendChild(tr);
    }
    wrap.appendChild(table);
    wrap.appendChild(this._csvLink(event));
    block.appendChild(wrap);
    this._scroll();
  }

  _csvLink(event) {
    const link = this._el("a", "csv-link", "Download CSV");
    link.href = event.csv_url;
    link.addEventListener("click", (e) => {
      e.preventDefault();
      this._downloadCsv(event);
    });
    return link;
  }

  async _downloadCsv(event) {
    try {
      const response = await this._fetch(event.csv_url);
      if (response.status === 404) {
        this._fridayBlock().appendChild(this._el("div", "text", CSV_UNAVAILABLE));
        this._scroll();
        return;
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = this._doc.createElement("a");
      anchor.href = url;
      anchor.download = `${event.series_id}_${event.dataset_id}.csv`;
      this._doc.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch {
      this._fridayBlock().appendChild(this._el("div", "text", CSV_UNAVAILABLE));
      this._scroll();
    }
  }

  _stats(event) {
    const block = this._fridayBlock();
    const table = this._el("table", "data-table");
    for (const cell of event.rows) {
      const tr = this._doc.createElement("tr");
      tr.appendChild(this._el("th", null, cell.label));
      tr.appendChild(this._el("td", null, cell.value));
      table.appendChild(tr);
    }
    block.appendChild(table);
    this._scroll();
  }

  _chart(event) {
    const block = this._fridayBlock();
    const img = this._el("img", "chart-image");
    img.src = event.image;
    img.alt = event.alt;
    block.appendChild(img);
    this._scroll();
  }

  _final(event) {
    const block = this._fridayBlock();
    if (event.text) block.appendChild(this._el("div", "text", event.text));
    if (event.audio_error) this._noticeInline(block);
    this._scroll();
    this._friday = null; // the turn is done
  }

  _noticeInline(block) {
    if (block.querySelector(".audio-notice")) return;
    block.appendChild(this._el("div", "audio-notice status-chip", AUDIO_UNAVAILABLE));
  }

  _scroll() {
    this._log.scrollTop = this._log.scrollHeight;
  }
}

export const MESSAGES = { CSV_UNAVAILABLE, AUDIO_UNAVAILABLE, EMPTY_PREVIEW };
