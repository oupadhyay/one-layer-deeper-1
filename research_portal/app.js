(() => {
  "use strict";
  const root = document.querySelector("#experiments");
  const groupFilter = document.querySelector("#group-filter");
  const verdictFilter = document.querySelector("#verdict-filter");
  const rungFilter = document.querySelector("#rung-filter");
  const tierControls = document.querySelector("#tier-controls");
  const runDialog = document.querySelector("#run-dialog");
  const dialogTitle = document.querySelector("#run-dialog-title");
  const dialogContent = document.querySelector("#run-dialog-content");
  const metricLabels = {train:"Train", trainT1:"Train T1", test:"Aggregate test", testT1:"Test T1", seenT1:"Seen-N T1", ood:"Aggregate OOD", oodT6:"OOD T6", oodNT1:"OOD-N T1"};
  const tiers = ["easy", "medium", "hard"];
  const svgNamespace = "http://www.w3.org/2000/svg";
  const node = (tag, className, text) => { const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined) el.textContent = text; return el; };
  const svgNode = (tag, attributes = {}) => { const el = document.createElementNS(svgNamespace, tag); Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, String(value))); return el; };

  function exactFraction(metric) {
    if (!metric || typeof metric.exact !== "string") return null;
    const match = metric.exact.match(/^(\d+)\/(\d+)$/);
    if (!match || Number(match[2]) === 0) return null;
    return { correct:Number(match[1]), total:Number(match[2]), accuracy:Number(match[1]) / Number(match[2]) };
  }

  function jointT1(exp) {
    const seen = exactFraction(exp.metrics.seenT1 || exp.metrics.depthT1);
    const ood = exactFraction(exp.metrics.oodNT1);
    if (!seen || !ood) return null;
    const balanced = Math.min(seen.accuracy, ood.accuracy);
    const transferRatio = Math.min(seen.accuracy / (4 / 512), ood.accuracy / (2 / 512));
    return {
      seen, ood, balanced, transferRatio,
      publicPass: transferRatio > 1,
      breakthroughPass: seen.accuracy >= .10 && ood.accuracy >= .10,
    };
  }

  function verdictCategory(exp) {
    const verdict = exp.verdict.toLowerCase();
    if (exp.status || /abort|evaluation failed|resource-gate/.test(verdict)) return "Aborted / invalid";
    const joint = jointT1(exp);
    if (joint && joint.publicPass) return "Transfer success";
    if (/inconclusive|diagnostic|control/.test(verdict)) return "Diagnostic / inconclusive";
    if (/pass/.test(verdict) && /fail|reject/.test(verdict)) return "Partial success";
    if (/fail|reject/.test(verdict)) return "Failed gate";
    if (/pass|success|accepted/.test(verdict)) return "Partial success";
    return "Diagnostic / inconclusive";
  }

  function metricsTable(exp) {
    const table = node("table", "metrics");
    const head = node("thead");
    const headingRow = node("tr");
    ["Split", "Exact", "CE", "Evidence"].forEach(value => headingRow.append(node("th", "", value)));
    head.append(headingRow);
    table.append(head);
    const body = node("tbody");
    Object.entries(exp.metrics).forEach(([key, value]) => {
      const row = node("tr");
      const evidence = key.startsWith("train") ? "Training fit" : "Transfer";
      row.append(
        node("td", "", metricLabels[key] || key),
        node("td", "", value.exact ?? "n/a"),
        node("td", "", value.ce === null || value.ce === undefined ? "n/a" : String(value.ce)),
        node("td", evidence.toLowerCase().replace(" ", "-"), evidence),
      );
      body.append(row);
    });
    table.append(body);
    return table;
  }

  function appendDefinition(list, term, value) {
    list.append(node("dt", "", term), node("dd", "", value ?? "n/a"));
  }

  function rungCell(rung) {
    if (!rung) return "n/a";
    return `${rung.correct}/${rung.total} (${(100 * rung.correct / rung.total).toFixed(3)}%)`;
  }

  function rungTable(detail) {
    const table = node("table", "rung-table");
    const head = node("thead");
    const headingRow = node("tr");
    ["Rung", "Seen-N", "OOD-N"].forEach(value => headingRow.append(node("th", "", value)));
    head.append(headingRow);
    table.append(head);
    const body = node("tbody");
    for (const timeSteps of [1, 2, 4, 8, 16, 32, 64]) {
      const seen = detail?.seenRungs?.find(item => item.timeSteps === timeSteps);
      const ood = detail?.oodRungs?.find(item => item.timeSteps === timeSteps);
      const row = node("tr");
      row.append(node("th", "", `T${timeSteps}`), node("td", "", rungCell(seen)), node("td", "", rungCell(ood)));
      body.append(row);
    }
    table.append(body);
    return table;
  }

  function openRunDetails(exp) {
    const detail = RUN_DETAILS[exp.id];
    dialogTitle.textContent = exp.title;
    dialogContent.replaceChildren();
    const facts = node("dl", "run-facts");
    appendDefinition(facts, "Date", detail?.createdAt ? new Date(detail.createdAt).toLocaleString() : "n/a");
    appendDefinition(facts, "Tier / dataset", detail ? `${detail.tier.toUpperCase()} / ${detail.dataset || "n/a"}` : "n/a");
    appendDefinition(facts, "Submission", detail?.submissionId);
    appendDefinition(facts, "Run", detail?.runId);
    appendDefinition(facts, "Model state", exp.modelState?.toLocaleString() || "n/a");
    appendDefinition(facts, "Updates / examples", `${exp.updates ?? "n/a"} / ${exp.examples ?? "n/a"}`);
    dialogContent.append(facts);
    dialogContent.append(node("h3", "", "Architecture"));
    dialogContent.append(node("p", "architecture-note", detail?.architecture || `${exp.group} — ${exp.protocol}`));
    if (detail?.packagePath) dialogContent.append(node("code", "package-path", detail.packagePath));
    dialogContent.append(node("h3", "", "Seen and OOD rung ladder"), rungTable(detail));
    if (Object.keys(exp.metrics).length) dialogContent.append(node("h3", "", "Aggregate metrics"), metricsTable(exp));
    dialogContent.append(
      node("p", "conclusion", `Conclusion: ${exp.conclusion}`),
      node("p", "limitation", `Limitation: ${exp.limitation}`),
    );
    const links = node("div", "links");
    Object.entries(exp.links).forEach(([label, href]) => {
      const link = node("a", "", label === "result" ? "Result JSON" : "Provenance JSON");
      link.href = href;
      links.append(link);
    });
    dialogContent.append(links);
    if (typeof runDialog.showModal === "function") runDialog.showModal(); else runDialog.setAttribute("open", "");
  }

  function chartDatum(exp, profile, rung) {
    const detail = RUN_DETAILS[exp.id];
    if (!detail?.createdAt || !tiers.includes(detail.tier)) return null;
    const rungs = profile === "seen" ? detail.seenRungs : detail.oodRungs;
    const metric = rungs?.find(item => item?.timeSteps === rung);
    if (!metric || !metric.total) return null;
    return {exp, detail, metric, date:new Date(detail.createdAt), accuracy:metric.correct / metric.total};
  }

  function formatDate(date) {
    return new Intl.DateTimeFormat(undefined, {month:"short", day:"numeric", timeZone:"UTC"}).format(date);
  }

  function positionTooltip(tooltip, container, event, fallbackX, fallbackY) {
    if (event?.clientX !== undefined) {
      const bounds = container.getBoundingClientRect();
      tooltip.style.left = `${Math.max(12, Math.min(bounds.width - 220, event.clientX - bounds.left + 12))}px`;
      tooltip.style.top = `${Math.max(8, event.clientY - bounds.top - 72)}px`;
    } else {
      tooltip.style.left = `${fallbackX}px`;
      tooltip.style.top = `${fallbackY}px`;
    }
  }

  function drawChart(container, profile, selectedTiers, rung) {
    container.replaceChildren();
    const width = 760;
    const height = 300;
    const margin = {top:18, right:18, bottom:42, left:58};
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const data = EXPERIMENTS.map(exp => chartDatum(exp, profile, rung)).filter(Boolean).filter(item => selectedTiers.has(item.detail.tier));
    const svg = svgNode("svg", {viewBox:`0 0 ${width} ${height}`, role:"img", "aria-label":`${profile === "seen" ? "Seen-N" : "OOD-N"} T${rung} accuracy by date`});
    const tooltip = node("div", "chart-tooltip");
    tooltip.hidden = true;
    container.append(svg, tooltip);
    if (!data.length) {
      svg.append(svgNode("text", {x:width / 2, y:height / 2, "text-anchor":"middle", class:"chart-empty"}));
      svg.lastChild.textContent = "No dated rung results for this selection";
      return 0;
    }
    let minimumDate = Math.min(...data.map(item => item.date.getTime()));
    let maximumDate = Math.max(...data.map(item => item.date.getTime()));
    if (minimumDate === maximumDate) { minimumDate -= 43200000; maximumDate += 43200000; }
    const maximumAccuracy = Math.max(.01, ...data.map(item => item.accuracy));
    const yMaximum = Math.min(1, Math.max(.01, Math.ceil(maximumAccuracy * 11500) / 10000));
    const x = value => margin.left + (value.getTime() - minimumDate) * plotWidth / (maximumDate - minimumDate);
    const y = value => margin.top + plotHeight - value * plotHeight / yMaximum;
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = yMaximum * tick / 4;
      const py = y(value);
      svg.append(svgNode("line", {x1:margin.left, x2:width - margin.right, y1:py, y2:py, class:"chart-grid"}));
      const label = svgNode("text", {x:margin.left - 9, y:py + 4, "text-anchor":"end", class:"chart-label"});
      label.textContent = `${(100 * value).toFixed(value < .01 ? 2 : 1)}%`;
      svg.append(label);
    }
    [minimumDate, (minimumDate + maximumDate) / 2, maximumDate].forEach((value, index) => {
      const px = margin.left + index * plotWidth / 2;
      const label = svgNode("text", {x:px, y:height - 13, "text-anchor":index === 0 ? "start" : index === 2 ? "end" : "middle", class:"chart-label"});
      label.textContent = formatDate(new Date(value));
      svg.append(label);
    });
    tiers.forEach(tier => {
      const series = data.filter(item => item.detail.tier === tier).sort((left, right) => left.date - right.date);
      if (!series.length) return;
      const path = svgNode("path", {d:series.map((item, index) => `${index ? "L" : "M"}${x(item.date).toFixed(2)},${y(item.accuracy).toFixed(2)}`).join(" "), class:`chart-line tier-${tier}`});
      svg.append(path);
      series.forEach(item => {
        const px = x(item.date);
        const py = y(item.accuracy);
        const point = svgNode("circle", {cx:px, cy:py, r:4, tabindex:0, class:`chart-point tier-${tier}`, role:"button", "aria-label":`${item.exp.title}, ${tier}, ${profile} T${rung}, ${item.metric.correct} of ${item.metric.total}`});
        const tooltipText = `${item.exp.title} · ${tier.toUpperCase()} · ${formatDate(item.date)} · ${item.metric.correct}/${item.metric.total} (${(100 * item.accuracy).toFixed(3)}%)`;
        const show = event => { tooltip.textContent = tooltipText; tooltip.hidden = false; positionTooltip(tooltip, container, event, px + 10, py - 56); };
        point.addEventListener("pointerenter", show);
        point.addEventListener("pointermove", show);
        point.addEventListener("pointerleave", () => { tooltip.hidden = true; });
        point.addEventListener("focus", () => show(null));
        point.addEventListener("blur", () => { tooltip.hidden = true; });
        point.addEventListener("click", () => openRunDetails(item.exp));
        point.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); openRunDetails(item.exp); } });
        svg.append(point);
      });
    });
    return data.length;
  }

  function renderDashboard() {
    const selectedTiers = new Set([...tierControls.querySelectorAll("input:checked")].map(input => input.value));
    const rung = Number(rungFilter.value);
    const seenCount = drawChart(document.querySelector("#seen-chart"), "seen", selectedTiers, rung);
    const oodCount = drawChart(document.querySelector("#ood-chart"), "ood", selectedTiers, rung);
    document.querySelector("#chart-status").textContent = `T${rung} · ${seenCount} Seen-N points · ${oodCount} OOD-N points · ${[...selectedTiers].map(tier => tier[0].toUpperCase() + tier.slice(1)).join(", ") || "no tiers selected"}`;
  }

  [...new Set(EXPERIMENTS.map(exp => exp.group))].sort().forEach(value => groupFilter.add(new Option(value, value)));
  [...new Set(EXPERIMENTS.map(verdictCategory))].sort().forEach(value => verdictFilter.add(new Option(value, value)));
  const failures = EXPERIMENTS.filter(exp => /fail|reject|aborted/i.test(exp.verdict)).length;
  const jointPasses = EXPERIMENTS.filter(exp => jointT1(exp)?.publicPass).length;
  const datedRuns = Object.keys(RUN_DETAILS).length;
  document.querySelector("#summary").textContent = `${EXPERIMENTS.length} attempts · ${datedRuns} dated remote runs · ${new Set(EXPERIMENTS.map(exp => exp.group)).size} approaches · ${jointPasses} beat both public T1 references · ${failures} failed/rejected/aborted outcomes`;

  function card(exp) {
    const article = node("article", "card");
    const heading = node("div", "card-heading");
    const titleBox = node("div");
    titleBox.append(node("h3", "", exp.title), node("code", "protocol", exp.protocol));
    heading.append(titleBox, node("span", "verdict", exp.verdict));
    article.append(heading);
    article.append(node("p", "meta", `Model state: ${exp.modelState ?? "n/a"} · Updates/examples: ${exp.updates ?? "n/a"}/${exp.examples ?? "n/a"}`));
    const joint = jointT1(exp);
    if (joint) {
      const status = joint.publicPass ? "PASS" : "FAIL";
      article.append(node("p", `joint-transfer ${joint.publicPass ? "joint-pass" : "joint-fail"}`,
        `Joint T1 transfer: ${status} · score ${joint.transferRatio.toFixed(3)} · balanced accuracy ${(100 * joint.balanced).toFixed(3)}% · seen ${joint.seen.correct}/${joint.seen.total} · OOD-N ${joint.ood.correct}/${joint.ood.total}`));
    }
    if (Object.keys(exp.metrics).length) article.append(metricsTable(exp));
    article.append(node("p", "conclusion", `Conclusion: ${exp.conclusion}`), node("p", "limitation", `Limitation: ${exp.limitation}`));
    const actions = node("div", "card-actions");
    const detailsButton = node("button", "details-button", "View run details");
    detailsButton.type = "button";
    detailsButton.addEventListener("click", () => openRunDetails(exp));
    actions.append(detailsButton);
    const links = node("div", "links");
    Object.entries(exp.links).forEach(([label, href]) => { const link = node("a", "", label === "result" ? "Result" : "Provenance"); link.href = href; links.append(link); });
    actions.append(links);
    article.append(actions);
    return article;
  }

  function render() {
    root.replaceChildren();
    const shown = EXPERIMENTS.filter(exp => (!groupFilter.value || exp.group === groupFilter.value) && (!verdictFilter.value || verdictCategory(exp) === verdictFilter.value));
    const groups = new Map();
    shown.forEach(exp => groups.set(exp.group, [...(groups.get(exp.group) || []), exp]));
    groups.forEach((items, name) => {
      const section = node("section", "approach");
      section.append(node("h2", "", name));
      const grid = node("div", "cards");
      items.forEach(exp => grid.append(card(exp)));
      section.append(grid);
      root.append(section);
    });
    document.querySelector("#empty").hidden = shown.length !== 0;
  }

  document.querySelector("#dialog-close").addEventListener("click", () => runDialog.close());
  runDialog.addEventListener("click", event => { if (event.target === runDialog) runDialog.close(); });
  groupFilter.addEventListener("change", render);
  verdictFilter.addEventListener("change", render);
  rungFilter.addEventListener("change", renderDashboard);
  tierControls.addEventListener("change", renderDashboard);
  renderDashboard();
  render();
})();
