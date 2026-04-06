/* ═══════════════════════════════════════════════════════════════════════════
   ML Platform Dashboard – Main Application Logic
   ══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── Config ──────────────────────────────────────────────────────────────────
const GATEWAY = window.GATEWAY_URL || "http://localhost:8000";
const INFERENCE_SVC = window.INFERENCE_URL || "http://localhost:8002";
const POLL_INTERVAL = 15_000;  // ms

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  models: [],
  history: [],
  throughputData: Array(30).fill(0),
  selectedFile: null,
  inferFile: null,
  activeTab: "dashboard",
};

// ── DOM Helpers ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};

// ── Toast Notification ────────────────────────────────────────────────────────
function toast(msg, type = "info", duration = 4000) {
  const icons = { success: "✅", error: "❌", info: "ℹ️", warn: "⚠️" };
  const t = el("div", `toast ${type}`, `<span>${icons[type] || "ℹ️"}</span> ${msg}`);
  $("toast-container").appendChild(t);
  setTimeout(() => {
    t.style.animation = "slideOut 0.3s ease forwards";
    setTimeout(() => t.remove(), 300);
  }, duration);
}

// ── Tab Navigation ────────────────────────────────────────────────────────────
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    switchTab(tab);
  });
});

function switchTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-content").forEach(s => s.classList.toggle("active", s.id === `tab-${tab}`));

  if (tab === "models") fetchModels();
  if (tab === "monitor") refreshMonitor();
  if (tab === "inference") fetchModelsForSelect();
}

// ═══════════════════════════════════════════════════════════════════════════
//  PARTICLE CANVAS BACKGROUND
// ═══════════════════════════════════════════════════════════════════════════
(function initCanvas() {
  const canvas = $("bg-canvas");
  const ctx = canvas.getContext("2d");
  let W, H, particles;

  const COLORS = ["#7c3aed", "#2563eb", "#06b6d4", "#10b981"];

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeParticles(n) {
    return Array.from({ length: n }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      alpha: Math.random() * 0.4 + 0.1,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color + Math.round(p.alpha * 255).toString(16).padStart(2, "0");
      ctx.fill();
    });

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(124,58,237,${0.15 * (1 - d / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  resize();
  particles = makeParticles(70);
  draw();
  window.addEventListener("resize", () => { resize(); particles = makeParticles(70); });
})();

// ═══════════════════════════════════════════════════════════════════════════
//  HEALTH CHECK
// ═══════════════════════════════════════════════════════════════════════════
async function checkHealth() {
  try {
    const res = await fetch(`${GATEWAY}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();

    const dot = $("status-dot");
    const txt = $("status-text");
    if (data.status === "healthy") {
      dot.className = "status-dot ok";
      txt.textContent = "All Systems Go";
    } else {
      dot.className = "status-dot error";
      txt.textContent = "Degraded";
    }

    // Health grid
    const grid = $("health-grid");
    grid.innerHTML = "";
    for (const [svc, status] of Object.entries(data.services || {})) {
      const cls = status === "ok" ? "ok" : "err";
      const item = el("div", `health-item ${cls}`);
      item.innerHTML = `<div class="health-dot ${cls}"></div><span>${svc}</span>`;
      grid.appendChild(item);
    }

    return data;
  } catch (e) {
    $("status-dot").className = "status-dot error";
    $("status-text").textContent = "Unreachable";
    $("health-grid").innerHTML = `<div class="health-item err"><div class="health-dot err"></div>Gateway unreachable</div>`;
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  DASHBOARD KPIs
// ═══════════════════════════════════════════════════════════════════════════
async function refreshDashboard() {
  const health = await checkHealth();

  // Model count
  try {
    const r = await fetch(`${GATEWAY}/models`);
    const models = await r.json();
    $("kpi-model-count").textContent = models.length;
  } catch { $("kpi-model-count").textContent = "—"; }

  // Queue depth
  try {
    const r = await fetch(`${GATEWAY}/metrics/summary`);
    const d = await r.json();
    const total = Object.values(d.queue_depths || {}).reduce((a, b) => a + b, 0);
    $("kpi-req-count").textContent = total;
  } catch { $("kpi-req-count").textContent = "—"; }

  // GPU stats
  try {
    const r = await fetch(`${INFERENCE_SVC}/gpu/stats`);
    const d = await r.json();
    if (d.gpu_available && d.devices?.length) {
      const dev = d.devices[0];
      const pct = ((dev.allocated_memory_gb / dev.total_memory_gb) * 100).toFixed(1);
      $("kpi-gpu-mem").textContent = `${pct}%`;
    } else {
      $("kpi-gpu-mem").textContent = "CPU";
    }
  } catch { $("kpi-gpu-mem").textContent = "—"; }

  // Simulated latency
  const lat = (15 + Math.random() * 20).toFixed(1);
  $("kpi-latency-val").textContent = `${lat}ms`;
}

// ═══════════════════════════════════════════════════════════════════════════
//  MODELS TAB
// ═══════════════════════════════════════════════════════════════════════════
async function fetchModels() {
  const container = $("models-list");
  container.innerHTML = `<div class="loading-state">Loading models…</div>`;
  try {
    const r = await fetch(`${GATEWAY}/models`);
    const models = await r.json();
    state.models = models;
    renderModels(models);
  } catch (e) {
    container.innerHTML = `<div class="empty-state">Could not reach server. Is the platform running?</div>`;
  }
}

function renderModels(models) {
  const container = $("models-list");
  if (!models.length) {
    container.innerHTML = `<div class="empty-state">No models registered yet.</div>`;
    return;
  }

  const table = el("table", "model-table");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Name</th><th>ID</th><th>Version</th><th>Framework</th>
        <th>Size</th><th>Status</th><th>Actions</th>
      </tr>
    </thead>`;
  const tbody = el("tbody");

  models.forEach(m => {
    const fwBadge = { pytorch: "badge-pytorch", onnx: "badge-onnx", tensorflow: "badge-tf" }[m.framework] || "";
    const stBadge = m.status === "ready" ? "badge-ready" : "badge-error";
    const row = el("tr");
    row.innerHTML = `
      <td><strong>${esc(m.name)}</strong><br><span class="mono" style="color:var(--text-muted);font-size:0.7rem">${esc(m.description || "")}</span></td>
      <td class="mono">${m.model_id}</td>
      <td>${esc(m.version)}</td>
      <td><span class="model-badge ${fwBadge}">${esc(m.framework)}</span></td>
      <td>${m.file_size_mb.toFixed(2)} MB</td>
      <td><span class="model-badge ${stBadge}">${esc(m.status)}</span></td>
      <td>
        <button class="btn btn-danger" style="padding:4px 10px;font-size:0.75rem" onclick="deleteModel('${m.model_id}')">Delete</button>
      </td>`;
    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  container.innerHTML = "";
  container.appendChild(table);
}

window.deleteModel = async (id) => {
  if (!confirm("Delete this model?")) return;
  try {
    await fetch(`${GATEWAY}/models/${id}`, { method: "DELETE" });
    toast("Model deleted", "success");
    fetchModels();
  } catch {
    toast("Failed to delete model", "error");
  }
};

// ── File Upload ───────────────────────────────────────────────────────────────
const uploadZone = $("upload-zone");
const fileInput = $("file-input");

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => fileInput.files[0] && handleFileSelect(fileInput.files[0]));

function handleFileSelect(file) {
  state.selectedFile = file;
  uploadZone.querySelector(".upload-text").textContent = `📄 ${file.name} (${(file.size / 1e6).toFixed(2)} MB)`;
  toast(`File selected: ${file.name}`, "info");
}

$("upload-btn").addEventListener("click", uploadModel);
$("refresh-models-btn").addEventListener("click", fetchModels);

async function uploadModel() {
  if (!state.selectedFile) { toast("Select a model file first", "warn"); return; }

  const form = new FormData();
  form.append("file", state.selectedFile);
  const name = $("model-name").value || "my_model";
  const ver  = $("model-version").value || "1.0.0";
  const desc = $("model-desc").value || "";
  const fw   = $("model-framework").value;

  $("upload-progress").classList.remove("hidden");
  $("upload-btn").disabled = true;
  $("progress-fill").style.width = "10%";
  $("progress-text").textContent = "Uploading…";

  try {
    const url = `${GATEWAY}/models/upload?model_name=${encodeURIComponent(name)}&version=${encodeURIComponent(ver)}&description=${encodeURIComponent(desc)}&framework=${fw}`;
    $("progress-fill").style.width = "50%";
    const r = await fetch(url, { method: "POST", body: form });
    $("progress-fill").style.width = "90%";

    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    $("progress-fill").style.width = "100%";
    $("progress-text").textContent = `✅ Uploaded! Model ID: ${data.model_id}`;
    toast(`Model "${name}" uploaded successfully!`, "success");
    state.selectedFile = null;
    setTimeout(() => {
      $("upload-progress").classList.add("hidden");
      $("progress-fill").style.width = "0";
      uploadZone.querySelector(".upload-text").innerHTML = `Drop your model file here or <label for="file-input" class="upload-browse">browse</label>`;
    }, 3000);
    fetchModels();
  } catch (e) {
    $("progress-text").textContent = `❌ Error: ${e.message}`;
    toast(`Upload failed: ${e.message}`, "error");
  } finally {
    $("upload-btn").disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  INFERENCE TAB
// ═══════════════════════════════════════════════════════════════════════════
function fetchModelsForSelect() {
  fetch(`${GATEWAY}/models`)
    .then(r => r.json())
    .then(models => {
      const sel = $("infer-model-select");
      const existing = Array.from(sel.options).map(o => o.value);
      models.forEach(m => {
        if (!existing.includes(m.model_id)) {
          const opt = document.createElement("option");
          opt.value = m.model_id;
          opt.textContent = `${m.name} v${m.version}`;
          sel.appendChild(opt);
        }
      });
    })
    .catch(() => {});
}

// Image selection
const inferZone = $("infer-upload-zone");
const inferInput = $("infer-file-input");
inferZone.addEventListener("click", () => inferInput.click());
inferInput.addEventListener("change", () => inferInput.files[0] && handleInferImage(inferInput.files[0]));

inferZone.addEventListener("dragover", e => { e.preventDefault(); inferZone.style.borderColor = "var(--accent-cyan)"; });
inferZone.addEventListener("dragleave", () => inferZone.style.borderColor = "");
inferZone.addEventListener("drop", e => {
  e.preventDefault();
  inferZone.style.borderColor = "";
  if (e.dataTransfer.files[0]) handleInferImage(e.dataTransfer.files[0]);
});

function handleInferImage(file) {
  state.inferFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    $("preview-img").src = e.target.result;
    $("preview-container").classList.remove("hidden");
    inferZone.classList.add("hidden");
  };
  reader.readAsDataURL(file);
}

$("remove-img-btn").addEventListener("click", () => {
  state.inferFile = null;
  $("preview-container").classList.add("hidden");
  inferZone.classList.remove("hidden");
  inferInput.value = "";
});

// Priority slider
$("infer-priority").addEventListener("input", e => $("priority-val").textContent = e.target.value);

$("run-infer-btn").addEventListener("click", runInference);

async function runInference() {
  const modelId = $("infer-model-select").value;
  const mode = document.querySelector('input[name="infer-mode"]:checked').value;
  const priority = parseInt($("infer-priority").value);

  let imageB64 = null;
  if (state.inferFile) {
    imageB64 = await fileToBase64(state.inferFile);
  } else {
    // Generate random image for demo
    imageB64 = generateDemoImage();
    logEntry("warn", "No image selected — using synthetic random image (demo mode)");
  }

  $("run-infer-btn").disabled = true;
  $("run-infer-btn").textContent = "⏳ Running…";
  $("infer-results").innerHTML = `<div class="loading-state">Running inference on ${modelId === "demo" ? "ResNet50 (demo)" : modelId}…</div>`;

  const payload = { model_id: modelId, input_data: { image_b64: imageB64 }, priority };

  try {
    if (mode === "sync") {
      const r = await fetch(`${GATEWAY}/infer/sync`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      renderResults(data, modelId, "sync");
      addHistory(data.request_id || generateId(), modelId, "completed", data.latency_ms);
    } else {
      const r = await fetch(`${GATEWAY}/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(await r.text());
      const queued = await r.json();
      addHistory(queued.request_id, modelId, "queued", null);
      renderResults(null, modelId, "async", queued.request_id);
      pollInferenceResult(queued.request_id, modelId);
    }
  } catch (e) {
    $("infer-results").innerHTML = `<div class="empty-state" style="color:var(--accent-red)">❌ ${esc(e.message)}</div>`;
    toast(`Inference failed: ${e.message}`, "error");
  } finally {
    $("run-infer-btn").disabled = false;
    $("run-infer-btn").textContent = "▶ Run Inference";
  }
}

async function pollInferenceResult(requestId, modelId) {
  $("infer-results").innerHTML = `<div class="loading-state">⏳ Queued (ID: ${requestId}). Polling…</div>`;
  for (let i = 0; i < 60; i++) {
    await sleep(2000);
    try {
      const r = await fetch(`${GATEWAY}/infer/status/${requestId}`);
      const d = await r.json();
      if (d.status === "completed") {
        renderResults(d.result, modelId, "async");
        updateHistory(requestId, "completed", d.result?.latency_ms);
        return;
      }
      if (d.status === "failed") {
        $("infer-results").innerHTML = `<div class="empty-state" style="color:var(--accent-red)">❌ Task failed</div>`;
        updateHistory(requestId, "failed");
        return;
      }
    } catch {}
  }
  $("infer-results").innerHTML = `<div class="empty-state">⏱️ Timeout waiting for result</div>`;
}

function renderResults(data, modelId, mode, requestId) {
  if (!data && mode === "async") {
    // Queued, waiting
    return;
  }

  const preds = data?.predictions || [];
  let html = `<div class="infer-meta">
    <span>🤖 ${modelId === "demo" ? "ResNet50 (demo)" : modelId}</span>
    <span>⚡ ${data?.device || "—"}</span>
    ${data?.latency_ms ? `<span>⏱ ${data.latency_ms}ms</span>` : ""}
    <span>📡 ${mode}</span>
  </div>`;

  if (!preds.length) {
    html += `<div class="empty-state">No predictions returned (check model output format).</div>`;
  } else {
    html += `<div class="prediction-list">`;
    preds.forEach((p, i) => {
      const pct = (p.confidence * 100).toFixed(1);
      html += `
        <div class="prediction-item">
          <div class="pred-rank ${i === 0 ? "top" : ""}">${p.rank}</div>
          <div class="pred-bar-wrap">
            <div class="pred-label">${esc(p.label)}</div>
            <div class="pred-bar-bg"><div class="pred-bar-fill" style="width:${pct}%"></div></div>
          </div>
          <div class="pred-confidence">${pct}%</div>
        </div>`;
    });
    html += `</div>`;
    logEntry("ok", `Inference complete: top=${preds[0].label} (${(preds[0].confidence * 100).toFixed(1)}%)`);
  }

  $("infer-results").innerHTML = html;
  toast("Inference complete!", "success");
}

// ── History ───────────────────────────────────────────────────────────────────
function addHistory(id, model, status, latency) {
  state.history.unshift({ id, model: model === "demo" ? "demo" : model.slice(0, 6), status, latency, time: new Date().toLocaleTimeString() });
  if (state.history.length > 20) state.history.pop();
  renderHistory();
}
function updateHistory(id, status, latency) {
  const item = state.history.find(h => h.id === id);
  if (item) { item.status = status; if (latency) item.latency = latency; }
  renderHistory();
}
function renderHistory() {
  const c = $("infer-history");
  if (!state.history.length) { c.innerHTML = `<div class="empty-state">No requests yet.</div>`; return; }
  c.innerHTML = `<div class="history-list">${state.history.map(h => `
    <div class="history-item">
      <span class="mono" style="color:var(--text-muted)">${h.time}</span>
      <span>${h.model}</span>
      <span class="history-status status-${h.status}">${h.status}</span>
      <span style="color:var(--text-muted);font-size:0.75rem">${h.latency ? h.latency + "ms" : "—"}</span>
    </div>`).join("")}</div>`;
}

// ═══════════════════════════════════════════════════════════════════════════
//  MONITOR TAB
// ═══════════════════════════════════════════════════════════════════════════
async function refreshMonitor() {
  await refreshGpuStats();
  await refreshQueueStats();
  updateThroughputChart();
}

async function refreshGpuStats() {
  const c = $("gpu-stats");
  try {
    const r = await fetch(`${INFERENCE_SVC}/gpu/stats`);
    const d = await r.json();
    if (!d.gpu_available) {
      c.innerHTML = `<div class="gpu-device-card">
        <div class="gpu-device-name">CPU Mode (No GPU detected)</div>
        <p style="color:var(--text-muted);font-size:0.8rem">CUDA not available – running on CPU. For GPU acceleration, deploy with NVIDIA GPU node.</p>
      </div>`;
      return;
    }
    c.innerHTML = d.devices.map(dev => {
      const memPct = ((dev.allocated_memory_gb / dev.total_memory_gb) * 100).toFixed(1);
      const resvPct = ((dev.reserved_memory_gb / dev.total_memory_gb) * 100).toFixed(1);
      return `<div class="gpu-device-card">
        <div class="gpu-device-name">🎮 GPU ${dev.id}: ${esc(dev.name)}</div>
        <div class="gpu-bar-group">
          <div class="gpu-bar-label"><span>Memory Allocated</span><span>${dev.allocated_memory_gb.toFixed(2)} / ${dev.total_memory_gb} GB (${memPct}%)</span></div>
          <div class="gpu-bar-bg"><div class="gpu-bar-fill bar-green" style="width:${memPct}%"></div></div>
        </div>
        <div class="gpu-bar-group">
          <div class="gpu-bar-label"><span>Memory Reserved</span><span>${dev.reserved_memory_gb.toFixed(2)} GB (${resvPct}%)</span></div>
          <div class="gpu-bar-bg"><div class="gpu-bar-fill bar-blue" style="width:${resvPct}%"></div></div>
        </div>
        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:6px">SM Count: ${dev.multi_processor_count}</div>
      </div>`;
    }).join("");
  } catch {
    c.innerHTML = `<div class="loading-state">Inference service unreachable</div>`;
  }
}

async function refreshQueueStats() {
  const c = $("queue-stats");
  try {
    const r = await fetch(`${GATEWAY}/metrics/summary`);
    const d = await r.json();
    const colors = ["#6b7280", "#f59e0b", "#06b6d4", "#7c3aed", "#ef4444"];
    c.innerHTML = Object.entries(d.queue_depths || {}).map(([k, v], i) => `
      <div class="queue-item">
        <div class="queue-priority">
          <div class="queue-pip" style="background:${colors[i % colors.length]}"></div>
          <span>${k.replace("_", " ")} priority</span>
        </div>
        <div class="queue-count" style="color:${colors[i % colors.length]}">${v}</div>
      </div>`).join("") || `<div class="loading-state">No queue data</div>`;
  } catch {
    c.innerHTML = `<div class="loading-state">Gateway unreachable</div>`;
  }
}

// ── Throughput Chart ──────────────────────────────────────────────────────────
let chartCtx = null;
let animFrame = null;

function updateThroughputChart() {
  const canvas = $("throughput-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || 600;
  const H = 120;
  canvas.width = W;
  canvas.height = H;

  // Push simulated data point
  state.throughputData.push(Math.random() * 40 + 10);
  if (state.throughputData.length > 30) state.throughputData.shift();

  ctx.clearRect(0, 0, W, H);
  const max = Math.max(...state.throughputData);
  const step = W / (state.throughputData.length - 1);

  // Fill
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, "rgba(124,58,237,0.3)");
  grad.addColorStop(1, "rgba(124,58,237,0)");

  ctx.beginPath();
  ctx.moveTo(0, H - (state.throughputData[0] / max) * (H - 20));
  state.throughputData.forEach((v, i) => {
    ctx.lineTo(i * step, H - (v / max) * (H - 20));
  });
  ctx.lineTo(W, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(0, H - (state.throughputData[0] / max) * (H - 20));
  state.throughputData.forEach((v, i) => ctx.lineTo(i * step, H - (v / max) * (H - 20)));
  ctx.strokeStyle = "#7c3aed";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Labels
  ctx.fillStyle = "rgba(148,163,184,0.8)";
  ctx.font = "11px Inter";
  ctx.fillText(`${max.toFixed(0)} img/s`, 8, 16);
  ctx.fillText("0", 8, H - 4);
}

// ── Log Stream ────────────────────────────────────────────────────────────────
function logEntry(level, msg) {
  const c = $("log-stream");
  const e = el("div", `log-entry ${level}`, `[${new Date().toLocaleTimeString()}] ${msg}`);
  c.appendChild(e);
  c.scrollTop = c.scrollHeight;
  if (c.children.length > 100) c.removeChild(c.firstChild);
}
$("clear-log-btn").addEventListener("click", () => $("log-stream").innerHTML = "");

// ═══════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════
function esc(str) {
  if (!str) return "";
  return String(str).replace(/[&<>"']/g, c => ({ "&": "&amp;","<": "&lt;",">": "&gt;",'"': "&quot;","'": "&#39;" }[c]));
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function generateId() { return Math.random().toString(36).slice(2, 10); }

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = e => {
      // Strip data URL prefix: "data:image/...;base64,"
      const b64 = e.target.result.split(",")[1];
      resolve(b64);
    };
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function generateDemoImage() {
  // Create a tiny canvas, draw random pixels, export as base64 JPEG
  const c = document.createElement("canvas");
  c.width = 224; c.height = 224;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(224, 224);
  for (let i = 0; i < img.data.length; i++) img.data[i] = Math.random() * 255 | 0;
  ctx.putImageData(img, 0, 0);
  return c.toDataURL("image/jpeg", 0.8).split(",")[1];
}

// ═══════════════════════════════════════════════════════════════════════════
//  POLLING LOOP
// ═══════════════════════════════════════════════════════════════════════════
async function tick() {
  if (state.activeTab === "dashboard") await refreshDashboard();
  if (state.activeTab === "monitor") await refreshMonitor();
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  await refreshDashboard();
  logEntry("ok", "ML Platform dashboard ready");
  setInterval(tick, POLL_INTERVAL);
})();
