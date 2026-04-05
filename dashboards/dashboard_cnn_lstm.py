"""\nRandom Forest Model Dashboard\nPrimary Model: Random Forest Machine Learning Model\nDedicated dashboard for Random Forest intrusion detection\nRuns on port 8082\n"""
from flask import Flask, render_template_string, jsonify
import os
import sys
import time

app = Flask(__name__)

# Primary model configuration
MODEL_NAME = "Random Forest"
MODEL_DESCRIPTION = "Random Forest Machine Learning Model"

# Import shared metrics
try:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    from shared_metrics import get_model_metrics
    METRICS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import shared_metrics: {e}")
    METRICS_AVAILABLE = False

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI-NIDS Random Forest Dashboard</title>
    <link
      rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Urbanist:wght@400;600;700&display=swap"
    />
    <style>
:root {
  --bg: #08131f;
  --accent: #f5b74e;
  --accent-2: #40d3a3;
  --danger: #ff6767;
  --card: #101f33;
  --card-strong: #14263f;
  --text: #e9f0f9;
  --muted: #9fb3c8;
  --border: #203754;
  --shadow: rgba(7, 12, 20, 0.4);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: "Space Grotesk", "Urbanist", "Segoe UI", sans-serif;
  color: var(--text);
  background: radial-gradient(circle at top, #0f2944, #050b14 55%);
  min-height: 100vh;
}

.app {
  min-height: 100vh;
  padding: 2.5rem 2.75rem 4rem;
  position: relative;
}

.app::before {
  content: "";
  position: absolute;
  top: -120px;
  right: -180px;
  width: 420px;
  height: 420px;
  background: radial-gradient(circle, rgba(64, 211, 163, 0.25), transparent 65%);
  filter: blur(10px);
  z-index: 0;
}

.app::after {
  content: "";
  position: absolute;
  bottom: -200px;
  left: -120px;
  width: 520px;
  height: 520px;
  background: radial-gradient(circle, rgba(245, 183, 78, 0.2), transparent 70%);
  filter: blur(18px);
  z-index: 0;
}

.hero {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-bottom: 2rem;
  position: relative;
  z-index: 1;
}

.hero-title h1 {
  font-size: clamp(2.5rem, 4vw, 3.75rem);
  margin: 0;
  letter-spacing: 0.02em;
}

.hero-title h1 span {
  color: var(--accent-2);
}

.hero-title p {
  margin: 0.25rem 0 0;
  color: var(--muted);
}

.eyebrow {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.3em;
  color: var(--accent);
}

.subtitle {
  max-width: 640px;
}

.model-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: linear-gradient(135deg, rgba(64, 211, 163, 0.2), rgba(64, 211, 163, 0.1));
  border: 1px solid var(--accent-2);
  border-radius: 999px;
  color: var(--accent-2);
  font-weight: 600;
  font-size: 0.9rem;
}

.hero-status {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 1rem;
}

.status {
  padding: 0.5rem 1rem;
  background: var(--card);
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  border: 1px solid var(--border);
  font-weight: 600;
  box-shadow: 0 10px 20px var(--shadow);
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--danger);
  box-shadow: 0 0 0 0 rgba(255, 103, 103, 0.6);
}

.dot.live {
  background: var(--accent-2);
  animation: pulse 1.8s infinite;
}

.timestamp {
  color: var(--muted);
  font-size: 0.9rem;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.25rem;
  margin-bottom: 2rem;
  position: relative;
  z-index: 1;
}

.kpi {
  background: linear-gradient(135deg, var(--card-strong), var(--card));
  border-radius: 16px;
  padding: 1.25rem;
  border: 1px solid var(--border);
  box-shadow: 0 18px 40px var(--shadow);
}

.kpi p {
  margin: 0 0 0.5rem;
  color: var(--muted);
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.2em;
}

.kpi h2 {
  margin: 0;
  font-size: 2rem;
}

.kpi span {
  color: var(--muted);
  font-size: 0.9rem;
}

.kpi.danger h2 {
  color: var(--danger);
}

.kpi.success h2 {
  color: var(--accent-2);
}

.kpi.primary {
  border-color: var(--accent-2);
  background: linear-gradient(135deg, rgba(64, 211, 163, 0.15), var(--card-strong));
}

.kpi.primary h2 {
  color: var(--accent-2);
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  position: relative;
  z-index: 1;
}

.card {
  background: var(--card);
  border-radius: 16px;
  padding: 1.5rem;
  border: 1px solid var(--border);
  min-height: 240px;
  box-shadow: 0 24px 50px var(--shadow);
  display: flex;
  flex-direction: column;
  gap: 1rem;
  animation: floatIn 0.6s ease;
}

.card h2 {
  margin-top: 0;
  margin-bottom: 0;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.pill {
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  background: rgba(64, 211, 163, 0.15);
  color: var(--accent-2);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.pill.danger {
  background: rgba(255, 103, 103, 0.15);
  color: var(--danger);
}

.pill.primary {
  background: rgba(245, 183, 78, 0.15);
  color: var(--accent);
}

.card ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.card li {
  padding: 0.5rem 0;
  border-bottom: 1px dashed #254b72;
  color: var(--muted);
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
}

.card li strong {
  color: var(--text);
}

.card li.threat {
  color: var(--danger);
}

.card li.threat strong {
  color: var(--danger);
}

.card li.model {
  color: var(--accent-2);
}

.card li.model strong {
  color: var(--accent-2);
}

.bar-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.bar-row {
  display: grid;
  grid-template-columns: 120px 1fr 50px;
  gap: 0.75rem;
  align-items: center;
}

.bar-label {
  color: var(--text);
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.bar-track {
  position: relative;
  height: 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.bar-fill {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

.bar-fill.danger {
  background: linear-gradient(90deg, var(--danger), #ff8a8a);
}

.bar-value {
  text-align: right;
  color: var(--muted);
  font-size: 0.85rem;
}

.empty {
  color: var(--muted);
  font-size: 0.9rem;
  padding: 1rem 0;
}

.metrics li {
  display: flex;
  justify-content: space-between;
  color: var(--muted);
}

.metrics li strong {
  color: var(--text);
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(64, 211, 163, 0.6);
  }
  70% {
    box-shadow: 0 0 0 12px rgba(64, 211, 163, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(64, 211, 163, 0);
  }
}

@keyframes floatIn {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 600px) {
  .app {
    padding: 1.5rem;
  }

  .bar-row {
    grid-template-columns: 1fr;
    gap: 0.4rem;
  }

  .bar-value {
    text-align: left;
  }
}
    </style>
  </head>
  <body>
    <main class="app">
      <header class="hero">
        <div class="hero-title">
          <p class="eyebrow">AI-NIDS Monitoring</p>
          <h1>Random Forest <span>Detection Dashboard</span></h1>
          <p class="subtitle">Real-time intrusion detection using Random Forest Machine Learning Model.</p>
        </div>
        <div class="hero-status">
          <div class="model-badge">
            <span>Primary Model</span>
          </div>
          <div class="status">
            <span class="dot" id="status-dot"></span>
            <span id="status">Loading...</span>
          </div>
          <div class="timestamp" id="updated"></div>
        </div>
      </header>

      <section class="kpi-grid">
        <div class="kpi primary">
          <p>RF Predictions</p>
          <h2 id="total-predictions">0</h2>
          <span>All time</span>
        </div>
        <div class="kpi danger">
          <p>Threats Detected</p>
          <h2 id="threats-detected">0</h2>
          <span id="threat-rate">0% of total</span>
        </div>
        <div class="kpi success">
          <p>Benign Traffic</p>
          <h2 id="benign-count">0</h2>
          <span>Normal traffic</span>
        </div>
        <div class="kpi">
          <p>Avg Confidence</p>
          <h2 id="avg-confidence">0.0%</h2>
          <span>Detection certainty</span>
        </div>
      </section>

      <section class="grid">
        <div class="card">
          <div class="card-header">\n            <h2>Recent RF Predictions</h2>\n            <span class="pill primary">RF</span>
          </div>
          <ul id="alerts"></ul>
        </div>
        <div class="card">
          <div class="card-header">
            <h2>Attack Categories</h2>
            <span class="pill danger">Top Threats</span>
          </div>
          <div id="categories" class="bar-list"></div>
        </div>
        <div class="card">
          <div class="card-header">
            <h2>Detection Summary</h2>
            <span class="pill" id="summary-pill">Live</span>
          </div>
          <ul class="metrics" id="summary-metrics"></ul>
        </div>
        <div class="card">
          <div class="card-header">\n            <h2>Model Information</h2>\n            <span class="pill primary">RF</span>
          </div>
          <ul class="metrics" id="system-metrics"></ul>
        </div>
      </section>
    </main>
    <script>
function formatPct(value) {
  if (value === undefined || value === null || isNaN(value)) return "0.0%";
  return (value * 100).toFixed(1) + "%";
}

function renderBars(container, items, isThreat = true) {
  container.innerHTML = "";
  if (!items || !items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No data yet - Run tests to see metrics";
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
    label.title = item.label;

    const bar = document.createElement("span");
    bar.className = "bar-fill" + (isThreat ? " danger" : "");

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = item.value;

    const track = document.createElement("span");
    track.className = "bar-track";
    track.appendChild(bar);

    bar.style.width = ((item.value / max) * 100) + "%";

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
  const categoriesEl = document.getElementById("categories");
  const summaryMetricsEl = document.getElementById("summary-metrics");
  const systemMetricsEl = document.getElementById("system-metrics");
  const updatedEl = document.getElementById("updated");

  try {
    const res = await fetch('/api/metrics?ts=' + Date.now(), { cache: "no-store" });
    const data = await res.json();

    status.textContent = "Live";
    statusDot.classList.add("live");

    const metrics = data.metrics || {};
    const recent = metrics.recent_predictions || [];
    
    // Data now model-specific from backend for Random Forest, no filtering needed
    // Data now model-specific from backend, no filtering needed
    const cnnLstmRecent = recent;
    
    const threatTypes = metrics.threat_types || {};
    const totalPredictions = metrics.total_predictions || 0;
    const threatsDetected = metrics.threats_detected || 0;
    const benignCount = metrics.benign_count || 0;

    // Update KPIs
    document.getElementById("total-predictions").textContent = totalPredictions.toLocaleString();
    document.getElementById("threats-detected").textContent = threatsDetected.toLocaleString();
    document.getElementById("benign-count").textContent = benignCount.toLocaleString();

    const threatRate = totalPredictions > 0 ? threatsDetected / totalPredictions : 0;
    document.getElementById("threat-rate").textContent = formatPct(threatRate) + " of total";

    // Calculate average confidence from Random Forest predictions
    let avgConf = 0;
    if (cnnLstmRecent.length > 0) {
      const confSum = cnnLstmRecent.reduce((sum, p) => sum + (p.confidence || 0), 0);
      avgConf = confSum / cnnLstmRecent.length;
    } else if (recent.length > 0) {
      const confSum = recent.reduce((sum, p) => sum + (p.confidence || 0), 0);
      avgConf = confSum / recent.length;
    }
    document.getElementById("avg-confidence").textContent = formatPct(avgConf);

    // Update recent predictions list (Random Forest only)
    alertsEl.innerHTML = "";
    const displayRecent = cnnLstmRecent.slice(0, 10);
    document.getElementById("recent-count").textContent = displayRecent.length + " events";

    if (displayRecent.length === 0 && recent.length === 0) {
      const empty = document.createElement("li");
      empty.className = "empty";
      empty.textContent = "No predictions yet - Run tests to populate metrics";
      alertsEl.appendChild(empty);
    } else {
      const toDisplay = displayRecent.length > 0 ? displayRecent : recent.slice(0, 10);
      toDisplay.forEach((pred) => {
        const item = document.createElement("li");
        const label = pred.label || pred.prediction || "unknown";
        const confidence = pred.confidence || 0;
        const isThreat = pred.is_threat || pred.threat || false;
        const ts = pred.timestamp;
        const model = pred.model || "Random Forest";

        item.className = isThreat ? "threat" : "model";
        let tsText = "";
        if (ts) {
          const date = new Date(ts * 1000);
          tsText = date.toLocaleTimeString();
        }

        item.innerHTML = "<strong>" + label + "</strong> <span>" + 
          formatPct(confidence) + " " + model + " " + tsText + "</span>";
        alertsEl.appendChild(item);
      });
    }

    // Update attack categories
    const categoryItems = Object.entries(threatTypes)
      .map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6);
    renderBars(categoriesEl, categoryItems, true);

    // Update summary metrics
    summaryMetricsEl.innerHTML = "";
    const summaryData = [
      { label: "Attack Types", value: Object.keys(threatTypes).length },
      { label: "Threat/Non-Threat", value: threatsDetected + "/" + benignCount },
      { label: "Recent Events", value: cnnLstmRecent.length || recent.length }
    ];
    summaryData.forEach((metric) => {
      const row = document.createElement("li");
      row.innerHTML = "<span>" + metric.label + "</span><strong>" + metric.value + "</strong>";
      summaryMetricsEl.appendChild(row);
    });

    // Update system metrics - Model Information
    systemMetricsEl.innerHTML = "";
    const lastUpdate = metrics.last_update || "N/A";
    const systemData = [
      { label: "Model Type", value: "Hybrid CNN-LSTM" },
      { label: "Architecture", value: "CNN + LSTM" },
      { label: "Primary", value: "Yes" },
      { label: "Last Update", value: lastUpdate }
    ];
    systemData.forEach((metric) => {
      const row = document.createElement("li");
      row.innerHTML = "<span>" + metric.label + "</span><strong>" + metric.value + "</strong>";
      systemMetricsEl.appendChild(row);
    });

    if (updatedEl) {
      const rawUpdated = data.updated_at || 0;
      const updatedMs = rawUpdated > 1000000000000 ? rawUpdated : rawUpdated * 1000;
      const dt = new Date(updatedMs);
      updatedEl.textContent = "Updated " + dt.toLocaleTimeString();
    }

  } catch (err) {
    console.error("Error loading metrics:", err);
    status.textContent = "Offline";
    statusDot.classList.remove("live");
  }
}

// Expose METRICS_AVAILABLE to JavaScript
const METRICS_AVAILABLE = """ + ("true" if METRICS_AVAILABLE else "false") + """;

setInterval(loadSummary, 2000);
loadSummary();
    </script>
  </body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "dashboard": "cnn_lstm", "model": MODEL_NAME})

@app.route('/api/metrics')
def metrics():
    try:
        if METRICS_AVAILABLE:
            metrics_data = get_model_metrics("Random Forest")
            
            # Calculate average confidence from recent predictions (CNN-LSTM only)
            recent = metrics_data.get('recent_predictions', [])
            avg_confidence = 0
            if recent:
                avg_confidence = sum(p.get('confidence', 0) for p in recent) / len(recent)
            
            return jsonify({
                "model": MODEL_NAME,
                "metrics": {
                    "total_predictions": metrics_data.get('total_predictions', 0),
                    "threats_detected": metrics_data.get('threats_detected', 0),
                    "benign_count": metrics_data.get('benign_count', 0),
                    "threat_types": metrics_data.get('threat_types', {}),
                    "recent_predictions": metrics_data.get('recent_predictions', []),
                    "average_confidence": avg_confidence,
                    "last_update": metrics_data.get('last_update', '')
                },
                "updated_at": int(time.time() * 1000)
            })
        else:
            # Fallback dummy data
            return jsonify({
                "model": MODEL_NAME,
                "metrics": {
                    "total_predictions": 0,
                    "threats_detected": 0,
                    "benign_count": 0,
                    "threat_types": {},
                    "recent_predictions": [],
                    "average_confidence": 0,
                    "last_update": ""
                },
                "updated_at": int(time.time() * 1000)
            })
    except Exception as e:
        print(f"Error getting metrics: {e}")
        return jsonify({
            "error": str(e),
            "model": MODEL_NAME,
            "metrics": {
                "total_predictions": 0,
                "threats_detected": 0,
                "benign_count": 0,
                "threat_types": {},
                "recent_predictions": []
            }
        }), 500

if __name__ == "__main__":
    print("=" * 60)
    print("AI-NIDS Random Forest Dashboard")
    print("=" * 60)
    print(f"Model: {MODEL_DESCRIPTION}")
    print(f"Dashboard URL: http://localhost:8082")
    print("Displays metrics from shared_metrics.json")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8082, debug=True)

