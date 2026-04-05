function formatMs(value) {
  if (!value && value !== 0) return "0 ms";
  if (value > 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(0)} ms`;
}

function formatPct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderBars(container, items) {
  container.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No data yet";
    container.appendChild(empty);
    return;
  }
  const max = Math.max(...items.map((item) => item.value), 1);
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = item.label;

    const bar = document.createElement("span");
    bar.className = "bar-fill";
    bar.style.width = `${(item.value / max) * 100}%`;

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = item.value;

    const track = document.createElement("span");
    track.className = "bar-track";
    track.appendChild(bar);

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    container.appendChild(row);
  });
}

async function loadSummary() {
  const status = document.getElementById("status");
  const statusDot = document.getElementById("status-dot");
  const alertsEl = document.getElementById("alerts");
  const actionsEl = document.getElementById("actions");
  const categoriesEl = document.getElementById("categories");
  const updatedEl = document.getElementById("updated");

  try {
    const res = await fetch("/api/summary");
    const data = await res.json();

    status.textContent = "Live";
    statusDot.classList.add("live");

    const overview = data.overview || {};
    const confidence = overview.confidence || {};
    const detLatency = overview.detection_latency_ms || {};
    const respLatency = overview.response_latency_ms || {};
    const model = overview.model || {};

    document.getElementById("active-threats").textContent =
      overview.active_threats || 0;
    document.getElementById("active-threats-pill").textContent =
      `${overview.active_threats || 0} threats`;
    document.getElementById("alert-window").textContent =
      `Last ${Math.round((overview.alert_window_sec || 0) / 60)} min`;

    document.getElementById("confidence-avg").textContent = (
      confidence.avg || 0
    ).toFixed(2);
    document.getElementById("confidence-range").textContent =
      `min ${(confidence.min || 0).toFixed(2)} | max ${(confidence.max || 0).toFixed(2)}`;

    document.getElementById("det-latency").textContent = formatMs(
      detLatency.p95 || 0
    );
    document.getElementById("det-latency-avg").textContent =
      `avg ${formatMs(detLatency.avg || 0)}`;

    document.getElementById("resp-latency").textContent = formatMs(
      respLatency.p95 || 0
    );
    document.getElementById("resp-latency-avg").textContent =
      `avg ${formatMs(respLatency.avg || 0)}`;

    document.getElementById("auto-rate").textContent =
      `${formatPct(model.auto_action_rate || 0)} auto`;

    if (updatedEl) {
      const dt = new Date((data.updated_at || 0) * 1000);
      updatedEl.textContent = `Updated ${dt.toLocaleTimeString()}`;
    }

    alertsEl.innerHTML = "";
    (data.alerts || []).slice(0, 8).forEach((a) => {
      const item = document.createElement("li");
      const label = a.label || "unknown";
      const confidenceText = (a.confidence || 0).toFixed(2);
      const src = a.src_ip || "-";
      const dst = a.dst_ip || "-";
      item.innerHTML = `<strong>${label}</strong> <span>${confidenceText}</span> <span>${src} -> ${dst}</span>`;
      alertsEl.appendChild(item);
    });

    const categoryItems = Object.entries(overview.categories || {})
      .map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6);
    renderBars(categoriesEl, categoryItems);

    const actionItems = Object.entries(overview.actions || {})
      .map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6);
    renderBars(actionsEl, actionItems);

    const modelMetrics = document.getElementById("model-metrics");
    modelMetrics.innerHTML = "";
    const metrics = [
      { label: "Alerts processed", value: model.alerts_total || 0 },
      { label: "Actions issued", value: model.actions_total || 0 },
      {
        label: "Auto action rate",
        value: formatPct(model.auto_action_rate || 0),
      },
      { label: "Last alert age", value: formatMs(model.last_alert_age_ms || 0) },
    ];
    metrics.forEach((metric) => {
      const row = document.createElement("li");
      row.innerHTML = `<span>${metric.label}</span><strong>${metric.value}</strong>`;
      modelMetrics.appendChild(row);
    });
  } catch (err) {
    status.textContent = "Offline";
    statusDot.classList.remove("live");
  }
}

setInterval(loadSummary, 2000);
loadSummary();
