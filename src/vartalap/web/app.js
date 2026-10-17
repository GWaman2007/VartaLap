/**
 * Vartalap Real-Time Conversational AI Frontend
 * Interactive Canvas Voice Orb, Real-time WebSocket sync, and Telemetry
 */

// State tracking
let isRunning = false;
let currentState = 'IDLE';
let currentRms = 0.0;
let ws = null;
let animationFrameId = null;

// DOM Elements
const stateIndicator = document.getElementById('stateIndicator');
const stateText = document.getElementById('stateText');
const orbSubtitle = document.getElementById('orbSubtitle');
const orbCanvas = document.getElementById('orbCanvas');
const orbPulse = document.getElementById('orbPulse');
const rmsValue = document.getElementById('rmsValue');
const vuFill = document.getElementById('vuFill');
const toggleBtn = document.getElementById('toggleBtn');
const btnIcon = document.getElementById('btnIcon');
const btnText = document.getElementById('btnText');
const transcriptFeed = document.getElementById('transcriptFeed');
const partialTranscriptText = document.getElementById('partialTranscriptText');
const micSelect = document.getElementById('micSelect');
const outSelect = document.getElementById('outSelect');
const wsStatus = document.getElementById('wsStatus');
const clearChatBtn = document.getElementById('clearChatBtn');

// Pipeline Steps
const pipelineSteps = {
  IDLE: document.getElementById('step-idle'),
  MONITORING: document.getElementById('step-monitoring'),
  TALKING: document.getElementById('step-talking'),
  ARBITRATION: document.getElementById('step-arbitration')
};

// Telemetry Elements
const metricStt = document.getElementById('metricStt');
const metricLlm = document.getElementById('metricLlm');
const latValStt = document.getElementById('latValStt');
const latBarStt = document.getElementById('latBarStt');
const latValLlm = document.getElementById('latValLlm');
const latBarLlm = document.getElementById('latBarLlm');
const latValTts = document.getElementById('latValTts');
const totalTurnLatency = document.getElementById('totalTurnLatency');

// ============================================================================
// 1. PROCEDURAL CANVAS VOICE ORB VISUALIZER
// ============================================================================
const ctx = orbCanvas.getContext('2d');
let orbTime = 0;

function drawOrb() {
  const w = orbCanvas.width;
  const h = orbCanvas.height;
  const cx = w / 2;
  const cy = h / 2;

  ctx.clearRect(0, 0, w, h);

  orbTime += 0.04;

  // Reactivity to RMS & State
  const energy = Math.min(1.0, currentRms * 12);
  let baseRadius = 65 + energy * 35;
  let primaryColor = '56, 189, 248'; // Cyan default

  if (currentState === 'TALKING') {
    primaryColor = '167, 139, 250'; // Violet
    baseRadius += 15;
  } else if (currentState === 'MONITORING') {
    primaryColor = '56, 189, 248'; // Bright Cyan
  } else if (currentState === 'ARBITRATION') {
    primaryColor = '52, 211, 153'; // Emerald
  } else {
    primaryColor = '148, 163, 184'; // Slate
  }

  // Draw 3 dynamic undulating energy rings
  for (let ring = 0; ring < 3; ring++) {
    ctx.beginPath();
    const ringOffset = ring * 1.5;
    const ringRadius = baseRadius + ring * 12;
    const points = 36;

    for (let i = 0; i <= points; i++) {
      const angle = (i / points) * Math.PI * 2;
      const wave = Math.sin(angle * 4 + orbTime + ringOffset) * (6 + energy * 18);
      const r = ringRadius + wave;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.closePath();
    ctx.strokeStyle = `rgba(${primaryColor}, ${0.5 - ring * 0.15})`;
    ctx.lineWidth = 2.5 - ring * 0.5;
    ctx.shadowColor = `rgba(${primaryColor}, 0.8)`;
    ctx.shadowBlur = 15;
    ctx.stroke();
  }

  // Draw glowing center core
  const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, baseRadius * 0.85);
  grad.addColorStop(0, `rgba(${primaryColor}, 0.85)`);
  grad.addColorStop(0.5, `rgba(${primaryColor}, 0.35)`);
  grad.addColorStop(1, 'rgba(0,0,0,0)');

  ctx.beginPath();
  ctx.arc(cx, cy, baseRadius * 0.85, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.shadowBlur = 25;
  ctx.fill();

  // Scale DOM pulse element
  if (orbPulse) {
    const scale = 1.0 + energy * 0.6;
    orbPulse.style.transform = `scale(${scale})`;
    orbPulse.style.opacity = `${0.3 + energy * 0.7}`;
  }

  animationFrameId = requestAnimationFrame(drawOrb);
}

drawOrb();

// ============================================================================
// 2. WEBSOCKET REAL-TIME SYNC
// ============================================================================
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[WS] Connected to Vartalap Backend');
    wsStatus.querySelector('.conn-dot').classList.remove('disconnected');
  };

  ws.onclose = () => {
    console.warn('[WS] Disconnected from Vartalap Backend. Reconnecting in 2s...');
    wsStatus.querySelector('.conn-dot').classList.add('disconnected');
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error('[WS Error]', err);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleWsMessage(data);
    } catch (e) {
      console.error('[WS Parse Error]', e, event.data);
    }
  };
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'state_change':
      updateState(msg.state);
      break;

    case 'energy':
      updateEnergy(msg.rms);
      break;

    case 'partial_transcript':
      updatePartialTranscript(msg.text);
      break;

    case 'user_spoke':
      addUserMessage(msg.text, msg.latency_ms);
      break;

    case 'agent_talk':
      addAgentMessage(msg.text, msg.latency_ms, msg.ttft_ms, msg.is_danger);
      break;

    case 'agent_done':
      // Turn yielded
      updateState('IDLE');
      break;

    case 'devices':
      populateDevices(msg.devices, msg.default_in, msg.default_out);
      break;

    case 'status':
      isRunning = msg.running;
      updateControlBtn();
      break;
  }
}

// ============================================================================
// 3. UI STATE & TELEMETRY UPDATES
// ============================================================================
function updateState(newState) {
  currentState = newState;

  // Update State Pill
  stateIndicator.className = `state-pill state-${newState.toLowerCase()}`;
  stateText.innerText = newState;

  // Update Orb Subtitle
  const subtitles = {
    IDLE: 'Awaiting Voice Input',
    MONITORING: 'Listening & Evaluating Speech...',
    TALKING: 'Agent Speaking Response...',
    ARBITRATION: 'Arbitrating Turn Transition...'
  };
  orbSubtitle.innerText = subtitles[newState] || newState;

  // Update 4-State Pipeline Step highlighting
  Object.keys(pipelineSteps).forEach((key) => {
    const el = pipelineSteps[key];
    if (el) {
      if (key === newState) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    }
  });
}

function updateEnergy(rms) {
  currentRms = rms;
  rmsValue.innerText = `${rms.toFixed(4)} RMS`;
  const pct = Math.min(100, Math.round(rms * 250));
  vuFill.style.width = `${pct}%`;
}

function updatePartialTranscript(text) {
  if (partialTranscriptText) {
    partialTranscriptText.innerText = text || '...';
  }
}

// ============================================================================
// 4. TRANSCRIPT FEED CARDS
// ============================================================================
function clearWelcomeBanner() {
  const welcome = transcriptFeed.querySelector('.welcome-banner');
  if (welcome) {
    welcome.remove();
  }
}

function addUserMessage(text, latencyMs = 220) {
  clearWelcomeBanner();

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const card = document.createElement('div');
  card.className = 'msg-card msg-user';

  card.innerHTML = `
    <div class="msg-meta">
      <span class="sender-tag">🎙️ YOU</span>
      <span class="latency-badge text-cyan">⚡ ${Math.round(latencyMs)}ms • Faster-Whisper</span>
      <span class="msg-time">${timeStr}</span>
    </div>
    <div class="msg-content">${escapeHtml(text)}</div>
  `;

  transcriptFeed.appendChild(card);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;

  // Clear live partial bar
  updatePartialTranscript('...');

  // Update STT telemetry
  metricStt.innerText = `${Math.round(latencyMs)}ms`;
  latValStt.innerText = `${Math.round(latencyMs)}ms`;
}

function addAgentMessage(text, latencyMs = 1100, ttftMs = 1100, isDanger = false) {
  clearWelcomeBanner();

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const card = document.createElement('div');
  card.className = `msg-card msg-agent ${isDanger ? 'msg-danger' : ''}`;

  const dangerTag = isDanger ? '<span class="latency-badge" style="background:rgba(244,63,94,0.2);color:#fb7185">⚠ SAFETY INTERRUPT</span>' : '';

  card.innerHTML = `
    <div class="msg-meta">
      <span class="sender-tag">🤖 VARTALAP</span>
      ${dangerTag}
      <span class="latency-badge text-violet">⚡ ${(latencyMs / 1000).toFixed(2)}s • Gemma NPU</span>
      <span class="msg-time">${timeStr}</span>
    </div>
    <div class="msg-content">${escapeHtml(text)}</div>
  `;

  transcriptFeed.appendChild(card);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;

  // Update LLM Telemetry
  metricLlm.innerText = `${(latencyMs / 1000).toFixed(1)}s`;
  latValLlm.innerText = `${(latencyMs / 1000).toFixed(1)}s`;
  const total = ((latencyMs + 220 + 60) / 1000).toFixed(1);
  totalTurnLatency.innerText = `~${total}s`;
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (m) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[m]);
}

// ============================================================================
// 5. AUDIO DEVICES POPULATION
// ============================================================================
function populateDevices(devices, defaultIn, defaultOut) {
  micSelect.innerHTML = '';
  outSelect.innerHTML = '';

  devices.forEach((d) => {
    if (d.max_input_channels > 0) {
      const opt = document.createElement('option');
      opt.value = d.index;
      opt.text = `[${d.index}] ${d.name}`;
      if (d.index === defaultIn) opt.selected = true;
      micSelect.appendChild(opt);
    }

    if (d.max_output_channels > 0) {
      const opt = document.createElement('option');
      opt.value = d.index;
      opt.text = `[${d.index}] ${d.name}`;
      if (d.index === defaultOut) opt.selected = true;
      outSelect.appendChild(opt);
    }
  });
}

// ============================================================================
// 6. CONTROLS & EVENT LISTENERS
// ============================================================================
function updateControlBtn() {
  if (isRunning) {
    toggleBtn.classList.add('btn-active');
    btnIcon.innerText = '■';
    btnText.innerText = 'STOP VARTALAP';
  } else {
    toggleBtn.classList.remove('btn-active');
    btnIcon.innerText = '▶';
    btnText.innerText = 'START VARTALAP';
  }
}

toggleBtn.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    if (isRunning) {
      ws.send(JSON.stringify({ action: 'stop' }));
    } else {
      const mic = parseInt(micSelect.value, 10);
      const out = parseInt(outSelect.value, 10);
      ws.send(JSON.stringify({ action: 'start', mic, out }));
    }
  }
});

clearChatBtn.addEventListener('click', () => {
  transcriptFeed.innerHTML = `
    <div class="welcome-banner">
      <div class="welcome-icon">⚡</div>
      <h4>Vartalap is Ready</h4>
      <p>Speak into your microphone to begin. As you speak, Faster-Whisper transcribes in 220ms, and Gemma evaluates on the NPU every tick.</p>
    </div>
  `;
});

// Quick Test Chips
document.querySelectorAll('.chip-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const text = btn.getAttribute('data-text');
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'simulate_user', text }));
    }
  });
});

// Device Selection Change
micSelect.addEventListener('change', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'set_mic', mic: parseInt(micSelect.value, 10) }));
  }
});

outSelect.addEventListener('change', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'set_out', out: parseInt(outSelect.value, 10) }));
  }
});

// Initialize on Load
window.addEventListener('DOMContentLoaded', () => {
  connectWebSocket();
});
