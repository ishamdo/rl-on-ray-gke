/* RL on Ray — Playroom Frontend.
 *
 * Controls the RL Training loop by calling FastAPI endpoints (/start, /stop, /status, /metrics, /logs/stream)
 * and visualizes the convergence (Accuracy & Loss charts) and active Ray pods.
 */

const HUB_BASE = "/api/features/ray";

const els = {
  mode: document.getElementById("mode-badge"),
  dash: document.getElementById("dashboard-btn"),
  metrics: document.getElementById("metrics-btn"),
  modelName: document.getElementById("model_name"),
  framework: document.getElementById("framework"),
  batchSize: document.getElementById("batch_size"),
  groupSize: document.getElementById("group_size"),
  groupSizeOut: document.getElementById("group_size_out"),
  launch: document.getElementById("launch"),
  progress: document.getElementById("progress"),
  canvas: document.getElementById("canvas"),
  pods: document.getElementById("pods"),
  workerCount: document.getElementById("worker-count"),
  statSteps: document.getElementById("stat-tiles"),
  statPods: document.getElementById("stat-pods"),
  statTime: document.getElementById("stat-tput"),
  statElapsed: document.getElementById("stat-elapsed"),
  consoleLog: document.getElementById("console-log"),
};

const ctx = els.canvas.getContext("2d");
let cfg = { mode: "MOCK", dataBase: HUB_BASE, dashboard_url: null };
let pods = new Map(); // name -> {type, status, count, el}
let pollInterval = null;
let workersTimer = null;

let metricsHistory = { steps: [], loss: [], reward: [] };
let trainingActive = false;
let startTime = null;

/* ---- auth + bases ----------------------------------------------------- */
function jwt() {
  return localStorage.getItem("admin_jwt") || "";
}
function hubHeaders() {
  const h = { "Content-Type": "application/json" };
  const t = jwt();
  if (t) h["Authorization"] = `Bearer ${t}`;
  return h;
}
function dataHeaders() {
  return cfg.mode === "MOCK" ? hubHeaders() : { "Content-Type": "application/json" };
}

/* ---- config / bootstrap ---------------------------------------------- */
async function loadConfig() {
  const override = new URLSearchParams(location.search).get("api");
  try {
    const r = await fetch(`${HUB_BASE}/config`, { headers: hubHeaders() });
    if (r.ok) {
      const c = await r.json();
      cfg.mode = c.mode || "LIVE";
      cfg.dashboard_url = c.dashboard_url || null;
      cfg.dataBase =
        cfg.mode === "MOCK"
          ? HUB_BASE
          : override || (c.gateway_ip ? `http://${c.gateway_ip}` : HUB_BASE);
      return;
    }
  } catch (_) {
    /* fall through to standalone */
  }
  cfg.mode = "LIVE";
  cfg.dataBase = override || location.origin;
}

function applyConfigUI() {
  els.mode.textContent = cfg.mode;
  els.mode.className = "badge " + (cfg.mode === "MOCK" ? "badge-mock" : "badge-live");
}

async function refreshDashboard() {
  if (cfg.mode === "MOCK" || !els.dash.hidden) return;
  try {
    const r = await fetch(`${cfg.dataBase}/dashboard`, { headers: dataHeaders() });
    if (!r.ok) return;
    const { url } = await r.json();
    if (url) {
      els.dash.href = url;
      els.dash.hidden = false;
    }
  } catch (_) {
    /* not ready yet */
  }
}

async function refreshMetrics() {
  if (cfg.mode === "MOCK" || !els.metrics.hidden) return;
  try {
    const r = await fetch(`${cfg.dataBase}/metrics-link`, { headers: dataHeaders() });
    if (!r.ok) return;
    const { url } = await r.json();
    if (url) {
      els.metrics.href = url;
      els.metrics.hidden = false;
    }
  } catch (_) {
    /* not ready yet */
  }
}

/* ---- cluster map ------------------------------------------------------ */
function ensurePod(name, type, status) {
  let p = pods.get(name);
  if (!p) {
    const el = document.createElement("div");
    el.className = `pod ${type === "head" ? "head" : "worker"}`;
    el.innerHTML =
      `<span class="dot"></span>` +
      `<span class="pname">${name}</span>` +
      `<span class="pcount">0</span>`;
    els.pods.appendChild(el);
    p = { type, status, count: 0, el };
    pods.set(name, p);
  }
  if (status) {
    p.status = status;
    p.el.classList.toggle("pending", status !== "Running");
  }
  return p;
}

function bumpPod(name) {
  const p = ensurePod(name, "worker");
  p.count += 1;
  p.el.querySelector(".pcount").textContent = p.count;
  p.el.classList.add("flash");
  setTimeout(() => p.el.classList.remove("flash"), 220);
}

async function pollWorkers() {
  const url = cfg.mode === "MOCK" ? `${HUB_BASE}/workers` : `${cfg.dataBase}/workers`;
  try {
    const r = await fetch(url, { headers: dataHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    const seen = new Set();
    (data.pods || []).forEach((pod) => {
      seen.add(pod.pod_name);
      const type = pod.node_type === "head" ? "head" : "worker";
      ensurePod(pod.pod_name, type, pod.status);
    });
    for (const [name, p] of pods) {
      if (!seen.has(name) && p.count === 0) {
        p.el.remove();
        pods.delete(name);
      }
    }
    const running = [...pods.values()].filter((p) => p.status === "Running").length;
    els.workerCount.textContent = `· ${pods.size} pod(s)`;
    els.statPods.textContent = running || pods.size;
  } catch (_) {
    /* transient */
  }
}

/* ---- chart drawing --------------------------------------------------- */
function resetCanvas() {
  const w = els.canvas.width;
  const h = els.canvas.height;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, w, h);
  
  ctx.strokeStyle = "#222";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  ctx.fillStyle = "#888";
  ctx.font = "14px monospace";
  ctx.fillText("Model training charts empty. Start training to view graphs.", 40, h / 2 - 10);
}

function drawCharts() {
  const w = els.canvas.width;
  const h = els.canvas.height;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, w, h);
  
  // Grid lines
  ctx.strokeStyle = "#222";
  ctx.lineWidth = 1;
  ctx.beginPath();
  // horizontal split
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  // vertical grids
  for (let x = 100; x < w; x += 100) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
  }
  ctx.stroke();
  
  const len = metricsHistory.steps.length;
  if (len < 2) {
    ctx.fillStyle = "#fff";
    ctx.font = "14px monospace";
    ctx.fillText("Waiting for training steps (loading model/scaling GPU)...", 30, h / 2 - 10);
    return;
  }
  
  // Draw Accuracy (Reward) - Top half (scales 0.0 to 1.0, smoothed with EMA)
  ctx.strokeStyle = "#10b981"; // emerald green
  ctx.lineWidth = 3;
  ctx.beginPath();
  let smoothedReward = 0;
  for (let i = 0; i < len; i++) {
    const x = (i / (len - 1)) * w;
    const rawVal = metricsHistory.reward[i];
    if (i === 0) {
      smoothedReward = rawVal;
    } else {
      smoothedReward = 0.5 * rawVal + 0.5 * smoothedReward; // EMA smoothing (alpha = 0.5)
    }
    const y = (h / 2) - 15 - smoothedReward * (h / 2 - 35);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  
  // Draw Loss - Bottom half (smoothed with EMA)
  ctx.strokeStyle = "#f43f5e"; // rose/red
  ctx.lineWidth = 3;
  ctx.beginPath();
  const maxLoss = Math.max(...metricsHistory.loss, 0.5);
  let smoothedLoss = 0;
  for (let i = 0; i < len; i++) {
    const x = (i / (len - 1)) * w;
    const rawVal = metricsHistory.loss[i];
    if (i === 0) {
      smoothedLoss = rawVal;
    } else {
      smoothedLoss = 0.5 * rawVal + 0.5 * smoothedLoss; // EMA smoothing (alpha = 0.5)
    }
    const y = h - 15 - (smoothedLoss / maxLoss) * (h / 2 - 35);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  
  // Labels
  ctx.fillStyle = "#10b981";
  ctx.font = "bold 13px monospace";
  ctx.fillText(`Accuracy (Avg Reward): ${(metricsHistory.reward[len-1] * 100).toFixed(0)}%`, 15, 25);
  
  ctx.fillStyle = "#f43f5e";
  ctx.font = "bold 13px monospace";
  ctx.fillText(`Loss: ${metricsHistory.loss[len-1].toFixed(4)}`, 15, h / 2 + 25);

  ctx.fillStyle = "#888";
  ctx.font = "11px monospace";
  ctx.fillText(`Step ${metricsHistory.steps[len-1]}`, w - 70, h / 2 - 10);
}

/* ---- training loop control ------------------------------------------- */
async function startTraining() {
  els.launch.disabled = true;
  els.consoleLog.textContent = "Requesting training start...\n";

  const body = JSON.stringify({
    model_name: els.modelName.value,
    framework: els.framework.value,
    batch_size: parseInt(els.batchSize.value, 10),
    group_size: parseInt(els.groupSize.value, 10),
  });

  try {
    const url = cfg.mode === "MOCK" ? `${HUB_BASE}/start` : `${cfg.dataBase}/start`;
    const r = await fetch(url, { method: "POST", headers: dataHeaders(), body });
    if (!r.ok) throw new Error(`start failed: ${r.status}`);
    const data = await r.json();

    if (data.status === "started" || data.status === "already_running") {
      trainingActive = true;
      startTime = Date.now();
      els.launch.textContent = "Stop Training";
      els.launch.className = "btn btn-danger"; // change style to red
      els.launch.disabled = false;
      
      // Start polling status & logs
      if (pollInterval) clearInterval(pollInterval);
      pollInterval = setInterval(pollTrainingState, 1000);
    }
  } catch (e) {
    els.consoleLog.textContent += `Failed to start training: ${e.message}\n`;
    els.launch.disabled = false;
  }
}

async function stopTraining() {
  els.launch.disabled = true;
  els.consoleLog.textContent += "\nStopping training loop...\n";

  try {
    const url = cfg.mode === "MOCK" ? `${HUB_BASE}/stop` : `${cfg.dataBase}/stop`;
    const r = await fetch(url, { method: "POST", headers: dataHeaders() });
    if (!r.ok) throw new Error(`stop failed: ${r.status}`);
  } catch (e) {
    els.consoleLog.textContent += `Stop request error: ${e.message}\n`;
    els.launch.disabled = false;
  }
}

async function pollTrainingState() {
  try {
    const statusUrl = cfg.mode === "MOCK" ? `${HUB_BASE}/status` : `${cfg.dataBase}/status`;
    const r = await fetch(statusUrl, { headers: dataHeaders() });
    if (!r.ok) return;
    const s = await r.json();

    // Update status text
    if (s.status === "starting") {
      els.progress.textContent = "· starting (scaling node & loading model)...";
    } else if (s.status === "training") {
      els.progress.textContent = "· active training";
    } else if (s.status === "error") {
      els.progress.textContent = `· error: ${s.error}`;
      stopPoll();
    } else if (s.status === "idle") {
      els.progress.textContent = "· idle";
      stopPoll();
    }

    // Refresh logs & metrics
    await fetchLogs();
    await fetchMetrics();
    
    // Update elapsed time
    if (startTime) {
      const diffSecs = Math.floor((Date.now() - startTime) / 1000);
      els.statElapsed.textContent = `${diffSecs}s`;
    }
  } catch (_) {
    /* ignore transient */
  }
}

function stopPoll() {
  trainingActive = false;
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
  els.launch.textContent = "Start Training";
  els.launch.className = "btn btn-primary";
  els.launch.disabled = false;
}

async function fetchLogs() {
  try {
    const url = cfg.mode === "MOCK" ? `${HUB_BASE}/logs/stream` : `${cfg.dataBase}/logs/stream`;
    const r = await fetch(url, { headers: dataHeaders() });
    if (!r.ok) return;
    const { logs } = await r.json();
    if (logs && logs.length > 0) {
      // Find the last line that names a pod, and trigger pod animation in cluster map
      const lastLine = logs[logs.length - 1];
      const match = lastLine.match(/Worker:\s*([^\s|]+)/);
      if (match) {
        bumpPod(match[1].trim());
      }
      
      els.consoleLog.textContent = logs.join("\n");
      els.consoleLog.scrollTop = els.consoleLog.scrollHeight;
    }
  } catch (_) {}
}

async function fetchMetrics() {
  try {
    const url = cfg.mode === "MOCK" ? `${HUB_BASE}/metrics` : `${cfg.dataBase}/metrics`;
    const r = await fetch(url, { headers: dataHeaders() });
    if (!r.ok) return;
    const m = await r.json();
    metricsHistory = m;
    
    drawCharts();
    
    if (m.steps.length > 0) {
      els.statSteps.textContent = m.steps[m.steps.length - 1];
      
      // Calculate step time
      const diffSecs = (Date.now() - startTime) / 1000;
      const stepCount = m.steps[m.steps.length - 1];
      els.statTime.textContent = `${(diffSecs / stepCount).toFixed(1)}s`;
    }
  } catch (_) {}
}

/* ---- listeners -------------------------------------------------------- */
els.groupSize.addEventListener("input", () => {
  els.groupSizeOut.textContent = els.groupSize.value;
});

els.launch.addEventListener("click", () => {
  if (trainingActive) {
    stopTraining();
  } else {
    startTraining();
  }
});

/* ---- init ------------------------------------------------------------- */
(async function init() {
  await loadConfig();
  applyConfigUI();
  resetCanvas();
  
  // Initial check
  pollWorkers();
  refreshDashboard();
  refreshMetrics();
  
  // Regular polling of workers & dashboard links
  workersTimer = setInterval(() => {
    pollWorkers();
    refreshDashboard();
    refreshMetrics();
  }, 2000);
})();
