(() => {
  "use strict";

  const api = () => window.pywebview.api;

  const $ = (id) => document.getElementById(id);

  let state = null;
  let autoplayTimer = null;
  let watchKindChoiceCache = null;
  let bpFieldChoices = null;

  function escapeHtml(text) {
    return text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  // -- Rendering helpers -------------------------------------------------

  function renderCircuitText(text, activeSpan) {
    const lines = text.split("\n");
    if (!activeSpan) {
      $("circuit-text").innerHTML = escapeHtml(text);
      return;
    }
    const [start, end] = activeSpan;
    const before = lines.slice(0, start - 1).join("\n");
    const active = lines.slice(start - 1, end).join("\n");
    const after = lines.slice(end).join("\n");
    const parts = [];
    if (before) parts.push(escapeHtml(before) + "\n");
    parts.push(`<span class="active-span">${escapeHtml(active)}</span>`);
    if (after) parts.push("\n" + escapeHtml(after));
    $("circuit-text").innerHTML = parts.join("");
  }

  const KET_LINE_RE = /^(\|[01]+⟩)(\s+)([+-][\d.]+[+-][\d.]+j)(\s+p=)([\d.]+)$/;

  function renderInfoText(text) {
    const html = text
      .split("\n")
      .map((line) => {
        const m = line.match(KET_LINE_RE);
        if (!m) return escapeHtml(line);
        return (
          `<span class="ket">${escapeHtml(m[1])}</span>${m[2]}` +
          `<span class="amp">${escapeHtml(m[3])}</span>${m[4]}` +
          `<span class="prob">${escapeHtml(m[5])}</span>`
        );
      })
      .join("\n");
    $("info-text").innerHTML = `<pre class="mono">${html}</pre>`;
  }

  function highlightSpans(text, spans, cls) {
    if (!spans || !spans.length) return escapeHtml(text);
    const sorted = [...spans].sort((a, b) => a[0] - b[0]);
    let out = "";
    let cursor = 0;
    for (const [start, end] of sorted) {
      if (end <= start || start < cursor) continue;
      out += escapeHtml(text.slice(cursor, start));
      out += `<span class="${cls}">${escapeHtml(text.slice(start, end))}</span>`;
      cursor = end;
    }
    out += escapeHtml(text.slice(cursor));
    return out;
  }

  function highlightDiffLines(text) {
    return text
      .split("\n")
      .map((line) => {
        if (line.startsWith("  - ")) return `<span class="diff-line-removed">${escapeHtml(line)}</span>`;
        if (line.startsWith("  + ")) return `<span class="diff-line-added">${escapeHtml(line)}</span>`;
        return escapeHtml(line);
      })
      .join("\n");
  }

  function fillSelect(select, values, selected) {
    select.innerHTML = "";
    for (const v of values) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      if (v === selected) opt.selected = true;
      select.appendChild(opt);
    }
  }

  // -- Full render ---------------------------------------------------

  function render(next) {
    state = next;

    $("header").textContent = state.header;
    $("header").title = state.header;
    $("show-all-circuits").checked = state.show_all_circuits;
    document.documentElement.style.setProperty("--circuit-size", `${state.circuit_zoom}px`);

    renderCircuitText(state.circuit_text, state.active_span);
    renderInfoText(state.info_text);

    const extra = state.extra_visual;
    if (extra) {
      $("info-text").innerHTML +=
        `<div class="hint" style="margin-top:8px">— ${escapeHtml(extra.label)} — (click to enlarge)</div>` +
        `<img src="${extra.data_uri}" class="info-figure" data-label="${escapeHtml(extra.label)}" style="max-width:100%" />`;
    }

    // Always keep this line in the layout (never toggle it away with
    // `hidden`) -- it has a fixed reserved height precisely so appearing or
    // disappearing text here can't shift the sticky controls below it.
    $("other-circuits").textContent = state.other_circuits.length
      ? `+${state.other_circuits.length} more not shown: ${state.other_circuits.join(", ")}`
      : "";

    $("btn-first").disabled = state.at_first;
    $("btn-prev").disabled = state.at_first;
    $("btn-next").disabled = state.at_last;
    $("btn-last").disabled = state.at_last;
    $("btn-play").textContent = autoplayTimer ? "⏸ Pause" : "⏵ Play";

    $("slider").max = state.count;
    $("slider").value = state.index + 1;

    renderWatches(state.watches);
    renderMarks(state.marks);
    renderBreakpoints(state.breakpoints);
    $("hit-status").textContent = state.hit_status || "";

    if (state.breakpoint_error) {
      $("bp-error").hidden = false;
      $("bp-error").textContent = state.breakpoint_error;
    }

    if (state.at_last && autoplayTimer) {
      clearInterval(autoplayTimer);
      autoplayTimer = null;
      $("btn-play").textContent = "⏵ Play";
    }
  }

  function renderWatches(watches) {
    const host = $("watch-cards");
    host.innerHTML = "";
    for (const w of watches.list) {
      const card = document.createElement("div");
      card.className = "card";

      const head = document.createElement("div");
      head.className = "card-head";
      const badge = document.createElement("span");
      badge.className = `badge badge-${w.kind}`;
      badge.textContent = w.kind;
      const title = document.createElement("span");
      title.className = "card-title";
      title.textContent = w.label;
      const remove = document.createElement("button");
      remove.className = "card-remove";
      remove.title = "Remove this watch";
      remove.textContent = "✕";
      remove.addEventListener("click", async () => render(await api().remove_watch(w.index)));
      head.append(badge, title, remove);
      card.appendChild(head);

      if (w.lines.length) {
        const body = document.createElement("div");
        body.className = "card-body";
        for (const line of w.lines) {
          const row = document.createElement("div");
          row.className = "card-body-line";
          row.textContent = line;
          body.appendChild(row);
        }
        card.appendChild(body);
      }
      host.appendChild(card);
    }
    if (watchKindChoiceCache === null) {
      fillSelect($("watch-kind"), watches.kind_choices, "qubit");
      watchKindChoiceCache = watches.kind_choices;
      refreshWatchRefChoices();
    }
  }

  function renderMarks(marks) {
    $("mark-status").textContent = `A: ${marks.label_a}   B: ${marks.label_b}`;
    $("btn-show-diff").hidden = !marks.both_marked;
    $("diff-tab-button").hidden = !marks.both_marked;
    if (!marks.both_marked && activeTab === "diff") setTab("timeline");

    const labelFor = (el) => marks.track_label[el] || el;
    fillSelect(
      $("diff-element-a"),
      marks.choices_a.map((e) => e),
      marks.selected_a
    );
    fillSelect(
      $("diff-element-b"),
      marks.choices_b.map((e) => e),
      marks.selected_b
    );
    [...$("diff-element-a").options].forEach((o) => (o.textContent = labelFor(o.value)));
    [...$("diff-element-b").options].forEach((o) => (o.textContent = labelFor(o.value)));

    if (!marks.diff) return;
    if (marks.diff.message) {
      $("diff-message").hidden = false;
      $("diff-message").textContent = marks.diff.message;
      $("diff-label-a").textContent = "";
      $("diff-label-b").textContent = "";
      $("diff-text-a").innerHTML = "";
      $("diff-text-b").innerHTML = "";
      $("stage-diff-text").innerHTML = "";
      $("watch-diff-text").innerHTML = "";
      return;
    }
    $("diff-message").hidden = true;
    $("diff-label-a").textContent = marks.diff.side_a.label;
    $("diff-label-b").textContent = marks.diff.side_b.label;
    $("diff-text-a").innerHTML = highlightSpans(
      marks.diff.side_a.text,
      marks.diff.side_a.spans,
      "diff-removed"
    );
    $("diff-text-b").innerHTML = highlightSpans(
      marks.diff.side_b.text,
      marks.diff.side_b.spans,
      "diff-added"
    );
    $("stage-diff-text").innerHTML = highlightDiffLines(marks.diff.stage_diff);
    $("watch-diff-text").innerHTML = highlightDiffLines(marks.diff.watch_diff);
  }

  function renderBreakpoints(breakpoints) {
    const host = $("breakpoint-rows");
    host.innerHTML = "";
    for (const bp of breakpoints.list) {
      const row = document.createElement("div");
      row.className = "bp-row";
      const label = document.createElement("span");
      label.className = "bp-row-label";
      label.textContent = `[${bp.handle}] ${bp.label}`;
      const remove = document.createElement("button");
      remove.className = "card-remove";
      remove.title = "Remove this breakpoint";
      remove.textContent = "✕";
      remove.addEventListener("click", async () => render(await api().remove_breakpoint(bp.handle)));
      row.append(label, remove);
      host.appendChild(row);
    }
  }

  // -- Tabs ------------------------------------------------------------

  let activeTab = "timeline";
  function setTab(name) {
    activeTab = name;
    document.querySelectorAll(".tab-button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  }

  // -- Watch ref choices (depends on kind) ------------------------------

  async function refreshWatchRefChoices() {
    const kind = $("watch-kind").value;
    const refs = await api().watch_ref_choices(kind);
    fillSelect($("watch-ref"), refs);
  }

  // -- Breakpoint condition builder --------------------------------------

  // "list" is a read-only IDL property on <input> (it resolves to the
  // associated <datalist> element) -- assigning it via Object.assign throws
  // a TypeError, so it must go through setAttribute instead like any other
  // HTML attribute that has no writable JS property.
  function bpField(tag, attrs) {
    const el = document.createElement(tag);
    const { list, ...rest } = attrs;
    Object.assign(el, rest);
    if (list) el.setAttribute("list", list);
    return el;
  }

  function renderBreakpointFields(kind) {
    const c = bpFieldChoices;
    const host = $("bp-fields");
    host.innerHTML = "";
    if (!c) return;

    if (kind === "gate") {
      const nameInput = bpField("input", { type: "text", placeholder: "gate name, e.g. cx", id: "bp-f-gate-name", list: "bp-gate-names" });
      const dl = bpField("datalist", { id: "bp-gate-names" });
      dl.innerHTML = c.gate_names.map((n) => `<option value="${n}">`).join("");
      const qubits = bpField("input", { type: "text", placeholder: "qubits, e.g. 0 1 (optional)", id: "bp-f-gate-qubits" });
      host.append(nameInput, dl, qubits);
    } else if (kind === "seq") {
      host.append(comparatorSelect(c), bpField("input", { type: "number", id: "bp-f-seq-value", min: 0, value: 0 }));
    } else if (kind === "watch") {
      const kindSel = bpField("select", { id: "bp-f-watch-kind" });
      fillSelect(kindSel, c.watch_kinds, "qubit");
      const refSel = bpField("select", { id: "bp-f-watch-ref" });
      const populateRef = () => {
        const k = kindSel.value;
        fillSelect(refSel, k === "qubit" ? c.qubits : k === "clbit" ? c.clbits : c.registers);
      };
      kindSel.addEventListener("change", populateRef);
      host.append(kindSel, refSel, comparatorSelect(c), thresholdInput());
      populateRef();
    } else if (kind === "expect") {
      const pauli = bpField("select", { id: "bp-f-expect-pauli" });
      fillSelect(pauli, c.paulis, "Z");
      const qubit = bpField("select", { id: "bp-f-expect-qubit" });
      fillSelect(qubit, c.qubits);
      host.append(pauli, qubit, comparatorSelect(c), thresholdInput());
    } else if (kind === "prob") {
      const outcome = bpField("input", { type: "text", placeholder: "outcome, e.g. 01", id: "bp-f-prob-outcome", list: "bp-outcomes" });
      const dl = bpField("datalist", { id: "bp-outcomes" });
      dl.innerHTML = c.outcomes.map((n) => `<option value="${n}">`).join("");
      host.append(outcome, dl, comparatorSelect(c), thresholdInput());
    } else if (kind === "tvd") {
      const expected = bpField("input", { type: "text", placeholder: "expected, e.g. 00=0.5 11=0.5", id: "bp-f-tvd-expected" });
      host.append(expected, comparatorSelect(c), thresholdInput());
    } else if (kind === "anomaly") {
      const sev = bpField("select", { id: "bp-f-anomaly-severity" });
      fillSelect(sev, c.severities, "warning");
      host.append(sev);
    }
  }

  function comparatorSelect(c) {
    const sel = bpField("select", { id: "bp-f-cmp" });
    fillSelect(sel, c.comparators, ">=");
    return sel;
  }

  function thresholdInput() {
    return bpField("input", { type: "number", id: "bp-f-threshold", step: "0.0001", value: 0.5 });
  }

  function collectBreakpointTokens(kind) {
    const v = (id) => (document.getElementById(id) || {}).value || "";
    if (kind === "gate") return [v("bp-f-gate-name"), ...v("bp-f-gate-qubits").split(/\s+/).filter(Boolean)];
    if (kind === "seq") return [v("bp-f-cmp"), v("bp-f-seq-value")];
    if (kind === "watch") return [v("bp-f-watch-kind"), v("bp-f-watch-ref"), v("bp-f-cmp"), v("bp-f-threshold")];
    if (kind === "expect") return [v("bp-f-expect-pauli"), v("bp-f-expect-qubit"), v("bp-f-cmp"), v("bp-f-threshold")];
    if (kind === "prob") return [v("bp-f-prob-outcome"), v("bp-f-cmp"), v("bp-f-threshold")];
    if (kind === "tvd") return [...v("bp-f-tvd-expected").split(/\s+/).filter(Boolean), v("bp-f-cmp"), v("bp-f-threshold")];
    const severity = v("bp-f-anomaly-severity");
    return severity ? [severity] : [];
  }

  async function initBreakpointBuilder() {
    bpFieldChoices = await api().breakpoint_field_choices();
    fillSelect($("bp-kind"), bpFieldChoices.kinds, "gate");
    fillSelect(
      $("bp-scope"),
      bpFieldChoices.scopes.map((s) => s.value)
    );
    [...$("bp-scope").options].forEach((o, i) => (o.textContent = bpFieldChoices.scopes[i].label));
    renderBreakpointFields($("bp-kind").value);
    $("bp-kind").addEventListener("change", () => renderBreakpointFields($("bp-kind").value));
  }

  // -- Playback ----------------------------------------------------------

  async function goto(index) {
    render(await api().goto(index));
  }

  function stopAutoplay() {
    if (autoplayTimer) {
      clearInterval(autoplayTimer);
      autoplayTimer = null;
      $("btn-play").textContent = "⏵ Play";
    }
  }

  function togglePlay() {
    if (autoplayTimer) {
      stopAutoplay();
      return;
    }
    if (state.at_last) return;
    $("btn-play").textContent = "⏸ Pause";
    autoplayTimer = setInterval(async () => {
      if (state.at_last) {
        stopAutoplay();
        return;
      }
      await goto(state.index + 1);
    }, Number($("interval-ms").value) || 800);
  }

  function zoom(delta) {
    if (delta > 0) api().zoom_in().then(render);
    else api().zoom_out().then(render);
  }

  // -- Wiring --------------------------------------------------------

  function wire() {
    document.querySelectorAll(".tab-button").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));

    $("btn-first").addEventListener("click", () => goto(0));
    $("btn-prev").addEventListener("click", () => goto(state.index - 1));
    $("btn-next").addEventListener("click", () => goto(state.index + 1));
    $("btn-last").addEventListener("click", () => goto(state.count - 1));
    $("btn-play").addEventListener("click", togglePlay);
    $("slider").addEventListener("input", (e) => goto(Number(e.target.value) - 1));
    $("show-all-circuits").addEventListener("change", async (e) => {
      render(await api().set_show_all_circuits(e.target.checked));
    });

    $("circuit-text").addEventListener(
      "wheel",
      (e) => {
        if (!(e.ctrlKey || e.metaKey)) return;
        e.preventDefault();
        zoom(e.deltaY < 0 ? 1 : -1);
      },
      { passive: false }
    );

    $("watch-kind").addEventListener("change", refreshWatchRefChoices);
    $("btn-add-watch").addEventListener("click", async () => {
      render(await api().add_watch($("watch-kind").value, $("watch-ref").value));
    });

    $("btn-mark-a").addEventListener("click", async () => render(await api().mark("a")));
    $("btn-mark-b").addEventListener("click", async () => render(await api().mark("b")));
    $("btn-clear-marks").addEventListener("click", async () => render(await api().clear_marks()));
    $("btn-show-diff").addEventListener("click", () => setTab("diff"));
    $("diff-element-a").addEventListener("change", async (e) => render(await api().set_diff_element("a", e.target.value)));
    $("diff-element-b").addEventListener("change", async (e) => render(await api().set_diff_element("b", e.target.value)));

    $("btn-add-breakpoint").addEventListener("click", async () => {
      $("bp-error").hidden = true;
      const kind = $("bp-kind").value;
      const tokens = collectBreakpointTokens(kind);
      const scope = $("bp-scope").value;
      render(await api().add_breakpoint(kind, tokens, scope));
    });
    $("btn-hit-first").addEventListener("click", async () => render(await api().goto_hit("first")));
    $("btn-hit-prev").addEventListener("click", async () => render(await api().goto_hit("prev")));
    $("btn-hit-next").addEventListener("click", async () => render(await api().goto_hit("next")));
    $("btn-hit-last").addEventListener("click", async () => render(await api().goto_hit("last")));

    $("btn-help").addEventListener("click", openHelp);
    $("btn-close-help").addEventListener("click", () => $("help-dialog").close());
    wireLightbox();

    document.addEventListener("keydown", (e) => {
      const inField = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
      if (e.key === "?") { openHelp(); return; }
      if (inField) return;
      if (e.key === "ArrowLeft") goto(state.index - 1);
      else if (e.key === "ArrowRight") goto(state.index + 1);
      else if (e.key === " ") { e.preventDefault(); togglePlay(); }
      else if (e.key === "+" || e.key === "=") zoom(1);
      else if (e.key === "-") zoom(-1);
      else if (e.key === "0") api().zoom_reset().then(render);
    });
  }

  // -- Image lightbox: click a histogram/Bloch-sphere figure to open it at
  // full size with wheel/button zoom and drag-to-pan -- the inline copy is
  // shrunk to fit a narrow info pane column, which makes a Bloch sphere in
  // particular too small to read on a laptop screen. --

  const lightbox = { scale: 1, panX: 0, panY: 0, dragging: false, startX: 0, startY: 0, baseX: 0, baseY: 0 };

  function applyLightboxTransform() {
    $("lightbox-img").style.transform = `translate(${lightbox.panX}px, ${lightbox.panY}px) scale(${lightbox.scale})`;
  }

  function lightboxSetScale(scale) {
    lightbox.scale = Math.min(6, Math.max(0.5, scale));
    applyLightboxTransform();
  }

  function lightboxReset() {
    lightbox.scale = 1;
    lightbox.panX = 0;
    lightbox.panY = 0;
    applyLightboxTransform();
  }

  function openLightbox(src, label) {
    $("lightbox-img").src = src;
    $("lightbox-title").textContent = label || "";
    lightboxReset();
    $("image-lightbox").showModal();
  }

  function wireLightbox() {
    $("info-text").addEventListener("click", (e) => {
      if (e.target.tagName === "IMG" && e.target.classList.contains("info-figure")) {
        openLightbox(e.target.src, e.target.dataset.label);
      }
    });
    $("btn-close-lightbox").addEventListener("click", () => $("image-lightbox").close());
    $("lightbox-zoom-in").addEventListener("click", () => lightboxSetScale(lightbox.scale * 1.25));
    $("lightbox-zoom-out").addEventListener("click", () => lightboxSetScale(lightbox.scale / 1.25));
    $("lightbox-zoom-reset").addEventListener("click", lightboxReset);

    const viewport = $("lightbox-viewport");
    viewport.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        lightboxSetScale(lightbox.scale * (e.deltaY < 0 ? 1.12 : 0.89));
      },
      { passive: false }
    );
    viewport.addEventListener("mousedown", (e) => {
      lightbox.dragging = true;
      lightbox.startX = e.clientX;
      lightbox.startY = e.clientY;
      lightbox.baseX = lightbox.panX;
      lightbox.baseY = lightbox.panY;
      viewport.classList.add("dragging");
    });
    window.addEventListener("mousemove", (e) => {
      if (!lightbox.dragging) return;
      lightbox.panX = lightbox.baseX + (e.clientX - lightbox.startX);
      lightbox.panY = lightbox.baseY + (e.clientY - lightbox.startY);
      applyLightboxTransform();
    });
    window.addEventListener("mouseup", () => {
      lightbox.dragging = false;
      viewport.classList.remove("dragging");
    });

    // The click target is the <dialog> itself only when the click lands
    // outside every child (they fill it via flex layout) -- i.e. the
    // backdrop area within the dialog's own box.
    $("image-lightbox").addEventListener("click", (e) => {
      if (e.target.id === "image-lightbox") $("image-lightbox").close();
    });
  }

  async function openHelp() {
    const entries = await api().help_entries();
    $("help-table").innerHTML = entries.map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(v)}</td></tr>`).join("");
    $("help-dialog").showModal();
  }

  async function boot() {
    wire();
    render(await api().goto(0));
    try {
      await initBreakpointBuilder();
    } catch (err) {
      // The breakpoint builder's field choices are secondary to the
      // always-visible timeline view -- a failure here should never blank
      // the whole window the way it did before this was split out.
      console.error("breakpoint builder failed to initialize", err);
    }
  }

  window.addEventListener("pywebviewready", boot);
})();
