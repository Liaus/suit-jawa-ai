"use strict";

const gestureImages = {
  Gajah: "/images/gajah",
  Orang: "/images/orang",
  Semut: "/images/semut",
};

const elements = {
  cameraFeed: document.getElementById("camera-feed"),
  cameraFallback: document.getElementById("camera-fallback"),
  cameraStatus: document.getElementById("camera-status"),
  cameraStatusText: document.getElementById("camera-status-text"),
  connectionMessage: document.getElementById("connection-message"),
  roundCurrent: document.getElementById("round-current"),
  roundTotal: document.getElementById("round-total"),
  roundProgress: document.getElementById("round-progress"),
  playerScore: document.getElementById("player-score"),
  aiScore: document.getElementById("ai-score"),
  accuracyValue: document.getElementById("accuracy-value"),
  accuracyFraction: document.getElementById("accuracy-fraction"),
  playerGesture: document.getElementById("player-gesture"),
  playerConfidence: document.getElementById("player-confidence"),
  phaseLabel: document.getElementById("phase-label"),
  phaseTitle: document.getElementById("phase-title"),
  phaseMessage: document.getElementById("phase-message"),
  aiPrediction: document.getElementById("ai-prediction"),
  aiStrategy: document.getElementById("ai-strategy"),
  aiCounter: document.getElementById("ai-counter"),
  aiCounterImage: document.getElementById("ai-counter-image"),
  aiCounterPlaceholder: document.getElementById("ai-counter-placeholder"),
  lastResult: document.getElementById("last-result"),
  startButton: document.getElementById("start-button"),
  startButtonLabel: document.getElementById("start-button-label"),
  fullscreenButton: document.getElementById("fullscreen-button"),
  recap: document.getElementById("recap"),
  recapClose: document.getElementById("recap-close"),
  recapPlayerScore: document.getElementById("recap-player-score"),
  recapAiScore: document.getElementById("recap-ai-score"),
  recapAccuracy: document.getElementById("recap-accuracy"),
  recapCorrect: document.getElementById("recap-correct"),
  roundHistory: document.getElementById("round-history-list"),
  replayButton: document.getElementById("replay-button"),
};

let streamActive = false;
let recapWasVisible = false;
let recapDismissed = false;

function percentage(value) {
  return value === null || value === undefined
    ? "—"
    : `${Math.round(value * 100)}%`;
}

function updateCamera(camera) {
  const ready = Boolean(camera && camera.ready);
  elements.cameraStatus.classList.toggle("is-ready", ready);
  elements.cameraStatus.classList.toggle("is-error", !ready && Boolean(camera && camera.error));
  elements.cameraStatusText.textContent = ready ? "KAMERA AKTIF" : "KAMERA OFFLINE";
  elements.cameraFallback.querySelector("small").textContent =
    (camera && camera.error) || "Menunggu koneksi webcam…";

  if (ready) {
    elements.cameraFallback.hidden = true;
    elements.cameraFeed.hidden = false;
    if (!streamActive) {
      streamActive = true;
      elements.cameraFeed.src = `/video_feed?t=${Date.now()}`;
    }
  } else {
    streamActive = false;
    elements.cameraFeed.hidden = true;
    elements.cameraFeed.removeAttribute("src");
    elements.cameraFallback.hidden = false;
  }
}

function updateGesture(gesture) {
  if (!gesture || !gesture.hand_detected) {
    elements.playerGesture.textContent = "TANGAN BELUM TERDETEKSI";
    elements.playerConfidence.textContent = "—";
    return;
  }

  elements.playerGesture.textContent = gesture.is_valid ? gesture.label.toUpperCase() : "UNKNOWN";
  elements.playerConfidence.textContent = percentage(gesture.confidence);
}

function updateForecast(forecast, aiMove) {
  elements.aiPrediction.textContent = forecast
    ? forecast.predicted_gesture.toUpperCase()
    : "MENUNGGU RONDE";
  elements.aiStrategy.textContent = forecast
    ? `Strategi ${forecast.strategy.replaceAll("+", " + ")} · keyakinan ${percentage(forecast.confidence)}`
    : "Prediksi baru tersedia saat ronde dimulai.";

  elements.aiCounter.textContent = aiMove ? aiMove.toUpperCase() : "MENUNGGU";
  const image = gestureImages[aiMove];
  if (image) {
    elements.aiCounterImage.src = image;
    elements.aiCounterImage.alt = `Ilustrasi gestur counter ${aiMove}`;
    elements.aiCounterImage.hidden = false;
    elements.aiCounterPlaceholder.hidden = true;
  } else {
    elements.aiCounterImage.removeAttribute("src");
    elements.aiCounterImage.hidden = true;
    elements.aiCounterPlaceholder.hidden = false;
  }
}

function updatePhase(state) {
  const phaseCopy = {
    ready: ["ARENA SIAP", "Siap bertanding?"],
    countdown: ["BERSIAP", `Ronde ${state.current_round} dimulai`],
    counter_reveal: ["AI MEMBACA POLA", "Counter AI disiapkan"],
    collecting_gesture: ["TUNJUKKAN GESTUR", "SUIT!"],
    round_result: ["RONDE SELESAI", state.last_result ? state.last_result.message : "Hasil ronde"],
    finished: ["SESI SELESAI", "Permainan selesai"],
  };
  const copy = phaseCopy[state.state] || ["ARENA AKTIF", state.message];
  elements.phaseLabel.textContent = copy[0];
  elements.phaseTitle.textContent = copy[1];
  elements.phaseMessage.textContent = state.message || "Tunjukkan salah satu gestur di depan kamera.";
}

function updateHistory(history) {
  elements.roundHistory.replaceChildren();
  for (const result of history) {
    const row = document.createElement("li");
    row.className = "history-row";
    const values = [
      String(result.round_number).padStart(2, "0"),
      result.player_gesture || "Tidak stabil",
      result.ai_gesture || "—",
      result.message,
    ];
    for (const value of values) {
      const cell = document.createElement("span");
      cell.textContent = value;
      row.append(cell);
    }
    elements.roundHistory.append(row);
  }
}

function updateRecap(state) {
  const visible = state.state === "finished";
  if (!visible) {
    recapDismissed = false;
    recapWasVisible = false;
    elements.recap.hidden = true;
    return;
  }

  if (recapDismissed) {
    elements.recap.hidden = true;
    return;
  }
  elements.recap.hidden = false;
  if (recapWasVisible) {
    return;
  }

  const statistics = state.prediction_statistics;
  elements.recapPlayerScore.textContent = String(state.scores.player);
  elements.recapAiScore.textContent = String(state.scores.ai);
  elements.recapAccuracy.textContent = percentage(statistics.accuracy);
  elements.recapCorrect.textContent = `${statistics.correct} / ${statistics.total}`;
  updateHistory(state.history);
  recapWasVisible = true;
  elements.replayButton.focus();
}

function closeRecap() {
  if (elements.recap.hidden) {
    return;
  }
  recapDismissed = true;
  elements.recap.hidden = true;
  elements.startButton.focus();
}

function updateStartButton(state) {
  const ready = Boolean(state.camera && state.camera.ready);
  const canStart = ready && (state.state === "ready" || state.state === "finished");
  elements.startButton.disabled = !canStart;
  if (!ready) {
    elements.startButtonLabel.textContent = "KAMERA BELUM SIAP";
  } else if (state.state === "finished") {
    elements.startButtonLabel.textContent = "MAIN LAGI";
  } else if (state.state === "ready") {
    elements.startButtonLabel.textContent = "MULAI BERMAIN";
  } else {
    elements.startButtonLabel.textContent = "RONDE BERJALAN";
  }
}

function renderState(state) {
  updateCamera(state.camera);
  updateGesture(state.gesture);
  updateForecast(state.forecast, state.ai_move);
  updatePhase(state);

  elements.roundCurrent.textContent = String(state.current_round).padStart(2, "0");
  elements.roundTotal.textContent = String(state.total_rounds).padStart(2, "0");
  const progress = state.total_rounds
    ? (state.current_round / state.total_rounds) * 100
    : 0;
  elements.roundProgress.style.width = `${Math.min(100, progress)}%`;
  elements.playerScore.textContent = String(state.scores.player);
  elements.aiScore.textContent = String(state.scores.ai);

  const statistics = state.prediction_statistics;
  elements.accuracyValue.textContent = percentage(statistics.accuracy);
  elements.accuracyFraction.textContent = `${statistics.total} tebakan dinilai`;

  const lastResult = state.last_result;
  elements.lastResult.textContent = lastResult
    ? `${lastResult.message} · P ${lastResult.player_gesture || "—"} / AI ${lastResult.ai_gesture || "—"}`
    : "BELUM ADA RONDE";

  const error = state.game_error || (state.camera && state.camera.error);
  elements.connectionMessage.textContent = error || state.message;
  updateStartButton(state);
  updateRecap(state);
}

async function refreshState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Server kiosk merespons ${response.status}.`);
    }
    renderState(await response.json());
  } catch (error) {
    elements.connectionMessage.textContent = `Koneksi ke kiosk terputus: ${error.message}`;
    elements.startButton.disabled = true;
    elements.startButtonLabel.textContent = "SERVER TIDAK TERHUBUNG";
  }
}

async function startGame() {
  elements.startButton.disabled = true;
  try {
    const response = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `Server kiosk merespons ${response.status}.`);
    }
    renderState(result);
  } catch (error) {
    elements.connectionMessage.textContent = error.message;
    await refreshState();
  }
}

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await document.documentElement.requestFullscreen();
    }
  } catch (error) {
    elements.connectionMessage.textContent = `Mode layar penuh tidak tersedia: ${error.message}`;
  }
}

elements.cameraFeed.addEventListener("error", () => {
  streamActive = false;
  elements.cameraFeed.hidden = true;
  elements.cameraFallback.hidden = false;
  elements.cameraFallback.querySelector("small").textContent = "Stream kamera terputus.";
});
elements.startButton.addEventListener("click", startGame);
elements.replayButton.addEventListener("click", startGame);
elements.recapClose.addEventListener("click", closeRecap);
elements.fullscreenButton.addEventListener("click", toggleFullscreen);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeRecap();
  }
});

refreshState();
window.setInterval(refreshState, 180);
