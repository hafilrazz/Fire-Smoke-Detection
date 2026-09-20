/**
 * Fire and Smoke Detection - Enterprise Surveillance Command Center Controller
 * Manages real-time WebRTC streams, PyTorch neural telemetry, Web Audio sirens,
 * clipboard ingestion, sample filtering, and interactive API playgrounds.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ========================================================================
  // 1. Live Surveillance Clock
  // ========================================================================
  const liveClockDisplay = document.getElementById("liveClockDisplay");
  const hudTimestamp = document.getElementById("hudTimestamp");

  function updateClock() {
    const now = new Date();
    const utcString = now.toTimeString().split(" ")[0] + " LOC";
    if (liveClockDisplay) {
      liveClockDisplay.textContent = utcString;
    }
    if (hudTimestamp) {
      const datePart = now.toISOString().split("T")[0];
      hudTimestamp.textContent = `${datePart} ${now.toTimeString().split(" ")[0]}`;
    }
  }
  setInterval(updateClock, 1000);
  updateClock();

  // ========================================================================
  // 2. Audio Alert & Multi-Tone Siren (Web Audio API)
  // ========================================================================
  let audioCtx = null;
  let isMuted = false;
  let isSirenActive = false;
  let sirenTimer = null;

  function initAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        audioCtx = new AudioContextClass();
      }
    }
  }

  function playTone(freqStart, freqEnd, duration, type = "sawtooth", gainLevel = 0.2) {
    if (isMuted) return;
    initAudioContext();
    if (!audioCtx) return;

    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }

    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freqStart, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(freqEnd, audioCtx.currentTime + duration);

      gain.gain.setValueAtTime(gainLevel, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("Audio synthesizer error:", e);
    }
  }

  function triggerHazardSiren(hazardType = "Fire") {
    if (isMuted || isSirenActive) return;
    isSirenActive = true;

    if (hazardType === "Fire") {
      // Urgent high-frequency dual sweep
      playTone(950, 1400, 0.35, "sawtooth", 0.25);
      setTimeout(() => {
        playTone(1400, 950, 0.35, "sawtooth", 0.25);
      }, 350);
    } else {
      // Pulsed warning warble for Smoke
      playTone(650, 850, 0.4, "sine", 0.2);
    }

    sirenTimer = setTimeout(() => {
      isSirenActive = false;
    }, 1200);
  }

  const audioToggleBtn = document.getElementById("audioToggleBtn");
  const audioBtnText = document.getElementById("audioBtnText");
  if (audioToggleBtn) {
    audioToggleBtn.addEventListener("click", () => {
      initAudioContext();
      isMuted = !isMuted;
      audioToggleBtn.classList.toggle("active", !isMuted);

      if (isMuted) {
        if (audioBtnText) audioBtnText.textContent = "Siren: Muted";
        showToast("Audio alarm sirens muted");
      } else {
        if (audioBtnText) audioBtnText.textContent = "Siren: Armed";
        playTone(700, 1000, 0.15, "sine", 0.15);
        showToast("Audio alarm sirens armed");
      }
    });
  }

  // ========================================================================
  // 3. Alert Sensitivity Threshold Control
  // ========================================================================
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdValDisplay = document.getElementById("thresholdValDisplay");
  let detectionThreshold = 70;

  if (thresholdSlider && thresholdValDisplay) {
    thresholdSlider.addEventListener("input", (e) => {
      detectionThreshold = parseInt(e.target.value, 10);
      thresholdValDisplay.textContent = `${detectionThreshold}%`;
      playTone(850, 920, 0.02, "triangle", 0.02);
    });
  }

  // ========================================================================
  // 4. Toast Notifications
  // ========================================================================
  const toastNotice = document.getElementById("toastNotice");
  const toastMessage = document.getElementById("toastMessage");
  let toastTimeout = null;

  function showToast(message) {
    if (!toastNotice || !toastMessage) return;
    toastMessage.textContent = message;
    toastNotice.classList.add("show");

    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
      toastNotice.classList.remove("show");
    }, 2800);
  }

  // ========================================================================
  // 5. Segmented Tab Navigation
  // ========================================================================
  const navTabItems = document.querySelectorAll(".nav-tab-item");
  const tabPanels = document.querySelectorAll(".tab-content-panel");

  navTabItems.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetPanelId = tab.getAttribute("data-tab");

      navTabItems.forEach((t) => t.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const activePanel = document.getElementById(targetPanelId);
      if (activePanel) {
        activePanel.classList.add("active");
      }

      playTone(1400, 1900, 0.035, "sine", 0.03);

      // If switching away from webcam tab, disengage camera stream
      if (targetPanelId !== "webcamTabPanel" && isCctvLive) {
        stopCctvStream();
      }
    });
  });

  // ========================================================================
  // 5b. DEFCON Readiness & Atmospheric Telemetry
  // ========================================================================
  function updateGlobalDefconState(pred, confidence = 0, latency = null) {
    const defconBadge = document.getElementById("hudDefconBadge");

    if (pred === "Fire") {
      document.body.className = "threat-state-fire";
      if (defconBadge) {
        defconBadge.textContent = "DEFCON 1 // CRITICAL FIRE";
        defconBadge.style.color = "var(--hazard-fire)";
        defconBadge.style.textShadow = "0 0 10px var(--hazard-fire-glow)";
      }
    } else if (pred === "Smoke") {
      document.body.className = "threat-state-smoke";
      if (defconBadge) {
        defconBadge.textContent = "DEFCON 2 // SMOKE ADVISORY";
        defconBadge.style.color = "var(--hazard-smoke)";
        defconBadge.style.textShadow = "0 0 10px var(--hazard-smoke-glow)";
      }
    } else {
      document.body.className = "threat-state-safe";
      if (defconBadge) {
        defconBadge.textContent = "DEFCON 4 // SECURE";
        defconBadge.style.color = "var(--hazard-safe)";
        defconBadge.style.textShadow = "0 0 8px var(--hazard-safe-glow)";
      }
    }
  }

  // ========================================================================
  // 6. Telemetry & Health Probe
  // ========================================================================
  async function probeSystemHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        const modelPill = document.getElementById("modelPill");
        const devicePill = document.getElementById("devicePill");
        if (modelPill) modelPill.textContent = `${data.model}`;
        if (devicePill) devicePill.textContent = `DEV: ${data.device.toUpperCase()}`;
      }
    } catch (e) {
      console.warn("Telemetry probe failed:", e);
    }
  }
  probeSystemHealth();

  // ========================================================================
  // 7. Categorized Sample Gallery
  // ========================================================================
  const samplesGrid = document.getElementById("samplesGrid");
  const filterChips = document.querySelectorAll(".filter-chip");
  let cachedSamples = [];

  async function loadSampleLibrary() {
    if (!samplesGrid) return;
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) return;
      cachedSamples = await res.json();
      renderFilteredSamples("all");
    } catch (e) {
      console.error("Failed to load sample library:", e);
    }
  }
  loadSampleLibrary();

  filterChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      filterChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const filter = chip.getAttribute("data-filter");
      renderFilteredSamples(filter);
    });
  });

  function renderFilteredSamples(filter) {
    if (!samplesGrid) return;
    samplesGrid.innerHTML = "";

    const filtered = cachedSamples.filter((sample) => {
      if (filter === "all") return true;
      return sample.category === filter;
    });

    filtered.forEach((sample) => {
      const item = document.createElement("div");
      item.className = "gallery-item";
      item.title = `Inspect ${sample.filename} (${sample.hint})`;
      item.innerHTML = `
        <img src="${sample.url}" alt="${sample.filename}" loading="lazy" />
        <div class="gallery-tag">${sample.hint.replace(" Sample", "")}</div>
      `;
      item.addEventListener("click", () => {
        executeSampleInspection(sample);
      });
      samplesGrid.appendChild(item);
    });
  }

  async function executeSampleInspection(sample) {
    stagePreview(sample.url);
    setAssessmentStaging();

    try {
      const res = await fetch("/api/predict/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_filename: sample.filename }),
      });
      const data = await res.json();
      applyThreatTelemetry(data);
    } catch (e) {
      applyAssessmentError("Failed to execute inference on sample.");
    }
  }

  // ========================================================================
  // 8. Image Viewport, Zoom & Clipboard Ingestion
  // ========================================================================
  const imageDropzone = document.getElementById("imageDropzone");
  const imageFileInput = document.getElementById("imageFileInput");
  const imagePreview = document.getElementById("imagePreview");
  const imagePlaceholder = document.getElementById("imagePlaceholder");
  const viewportToolbar = document.getElementById("viewportToolbar");

  let currentZoom = 1.0;

  if (imageDropzone && imageFileInput) {
    imageDropzone.addEventListener("click", () => imageFileInput.click());

    imageDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      imageDropzone.classList.add("dragover");
    });
    imageDropzone.addEventListener("dragleave", () => {
      imageDropzone.classList.remove("dragover");
    });
    imageDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      imageDropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        processUploadedImage(e.dataTransfer.files[0]);
      }
    });
    imageFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        processUploadedImage(e.target.files[0]);
      }
    });
  }

  // Paste from clipboard support
  window.addEventListener("paste", (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image") !== -1) {
        const blob = items[i].getAsFile();
        showToast("Image pasted from clipboard");
        processUploadedImage(blob);
        break;
      }
    }
  });

  // Viewport Zoom Tools
  const btnZoomIn = document.getElementById("btnZoomIn");
  const btnZoomOut = document.getElementById("btnZoomOut");
  const btnZoomReset = document.getElementById("btnZoomReset");

  if (btnZoomIn && imagePreview) {
    btnZoomIn.addEventListener("click", () => {
      currentZoom = Math.min(3.0, currentZoom + 0.25);
      imagePreview.style.transform = `scale(${currentZoom})`;
    });
  }
  if (btnZoomOut && imagePreview) {
    btnZoomOut.addEventListener("click", () => {
      currentZoom = Math.max(0.5, currentZoom - 0.25);
      imagePreview.style.transform = `scale(${currentZoom})`;
    });
  }
  if (btnZoomReset && imagePreview) {
    btnZoomReset.addEventListener("click", () => {
      currentZoom = 1.0;
      imagePreview.style.transform = "scale(1.0)";
    });
  }

  function stagePreview(src) {
    if (imagePlaceholder) imagePlaceholder.style.display = "none";
    if (viewportToolbar) viewportToolbar.style.display = "flex";
    if (imagePreview) {
      imagePreview.src = src;
      imagePreview.style.display = "block";
      currentZoom = 1.0;
      imagePreview.style.transform = "scale(1.0)";
    }
  }

  function processUploadedImage(file) {
    if (!file.type.startsWith("image/")) {
      showToast("Invalid file format. Provide an image.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (evt) => {
      stagePreview(evt.target.result);
    };
    reader.readAsDataURL(file);

    executeUploadInference(file);
  }

  async function executeUploadInference(file) {
    setAssessmentStaging();

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/predict/image", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.status === "error") {
        applyAssessmentError(data.message);
      } else {
        applyThreatTelemetry(data);
      }
    } catch (err) {
      applyAssessmentError("Network transmission error during inference.");
    }
  }

  function setAssessmentStaging() {
    const banner = document.getElementById("imageAssessmentBanner");
    const heading = document.getElementById("threatHeading");
    const desc = document.getElementById("threatDescription");
    const pct = document.getElementById("threatConfidenceDisplay");
    const iconBox = document.getElementById("threatIconBox");

    if (banner) banner.className = "assessment-banner";
    if (heading) heading.textContent = "INSPECTING TENSOR...";
    if (desc) desc.textContent = "Running forward pass through ResNet-50 feature extractor";
    if (pct) pct.textContent = "--%";
    if (iconBox) {
      iconBox.innerHTML = `<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>`;
    }
  }

  function applyThreatTelemetry(data) {
    const banner = document.getElementById("imageAssessmentBanner");
    const heading = document.getElementById("threatHeading");
    const desc = document.getElementById("threatDescription");
    const pct = document.getElementById("threatConfidenceDisplay");
    const iconBox = document.getElementById("threatIconBox");

    const globalPill = document.getElementById("globalIncidentPill");

    const pred = data.prediction || "Neutral";
    const confidence = data.confidence || 0.0;
    const isHazard = data.is_hazard;

    // Sync DEFCON state, atmospheric lighting, and tactical diagnostic canvases
    updateGlobalDefconState(pred, confidence, data.latency_ms);

    // Trigger siren if confidence crosses user-configured threshold
    if (isHazard && confidence >= detectionThreshold) {
      triggerHazardSiren(pred);
    }

    // Update Banner Appearance
    if (banner) {
      banner.className = "assessment-banner";
      if (pred === "Fire") {
        banner.classList.add("threat-fire");
      } else if (pred === "Smoke") {
        banner.classList.add("threat-smoke");
      } else {
        banner.classList.add("threat-safe");
      }
    }

    // Update Text & Icons
    if (heading) {
      heading.textContent =
        pred === "Fire"
          ? "CRITICAL HAZARD - FIRE DETECTED"
          : pred === "Smoke"
          ? "HIGH ADVISORY - SMOKE DETECTED"
          : "STATUS SECURE - AMBIENT NORMAL";
    }

    if (desc) {
      desc.textContent = isHazard
        ? `Thermal anomaly detected. Severity: ${data.hazard_level || "ALERT"}. Recommended immediate dispatch.`
        : "No combustion or smoke dispersal identified. Scene parameters verify clear.";
    }

    if (pct) {
      pct.textContent = `${confidence.toFixed(1)}%`;
      pct.style.color = data.color_hex || "#fff";
    }

    if (iconBox) {
      if (pred === "Fire") {
        iconBox.innerHTML = `<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="var(--hazard-fire)"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"></path></svg>`;
      } else if (pred === "Smoke") {
        iconBox.innerHTML = `<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="var(--hazard-smoke)"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"></path></svg>`;
      } else {
        iconBox.innerHTML = `<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="var(--hazard-safe)"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>`;
      }
    }

    // Global Command Status Pill
    if (globalPill) {
      if (pred === "Fire") {
        globalPill.textContent = "CRITICAL: FIRE CONFIRMED";
        globalPill.style.color = "var(--hazard-fire)";
        globalPill.style.borderColor = "rgba(239, 68, 68, 0.4)";
      } else if (pred === "Smoke") {
        globalPill.textContent = "WARNING: SMOKE IDENTIFIED";
        globalPill.style.color = "var(--hazard-smoke)";
        globalPill.style.borderColor = "rgba(245, 158, 11, 0.4)";
      } else {
        globalPill.textContent = "SECURE - NO THREATS";
        globalPill.style.color = "var(--hazard-safe)";
        globalPill.style.borderColor = "rgba(16, 185, 129, 0.3)";
      }
    }

    // Update Distribution Bars
    const probs = data.probabilities || {};
    syncMeter("valFire", "barFire", probs["Fire"] || 0.0);
    syncMeter("valSmoke", "barSmoke", probs["Smoke"] || 0.0);
    syncMeter("valNeutral", "barNeutral", probs["Neutral"] || 0.0);

    // Update Telemetry Grid
    const metaLatency = document.getElementById("metaLatency");
    if (metaLatency) metaLatency.textContent = `${data.latency_ms || "--"} ms`;

    const metaResolution = document.getElementById("metaResolution");
    if (metaResolution) metaResolution.textContent = data.resolution || "-- x --";

    const metaSource = document.getElementById("metaSource");
    if (metaSource) metaSource.textContent = data.filename || data.source || "Optical Ingestion";

    const metaThreatLevel = document.getElementById("metaThreatLevel");
    if (metaThreatLevel) {
      metaThreatLevel.textContent =
        pred === "Fire"
          ? "Level 3: Critical"
          : pred === "Smoke"
          ? "Level 2: Warning"
          : "Level 1: Secure";
    }

    // Update JSON Inspector
    const jsonDisplay = document.getElementById("jsonPayloadDisplay");
    if (jsonDisplay) {
      jsonDisplay.textContent = JSON.stringify(data, null, 2);
    }
  }

  function syncMeter(valId, barId, pct) {
    const valElem = document.getElementById(valId);
    const barElem = document.getElementById(barId);
    if (valElem) valElem.textContent = `${pct.toFixed(1)}%`;
    if (barElem) barElem.style.width = `${Math.min(100, Math.max(0, pct))}%`;
  }

  function applyAssessmentError(errText) {
    const heading = document.getElementById("threatHeading");
    const desc = document.getElementById("threatDescription");
    if (heading) heading.textContent = "INSPECTION ERROR";
    if (desc) desc.textContent = errText;
  }

  // Copy JSON button
  const btnCopyJson = document.getElementById("btnCopyJson");
  if (btnCopyJson) {
    btnCopyJson.addEventListener("click", (e) => {
      e.stopPropagation();
      const jsonDisplay = document.getElementById("jsonPayloadDisplay");
      if (jsonDisplay && navigator.clipboard) {
        navigator.clipboard.writeText(jsonDisplay.textContent).then(() => {
          showToast("JSON payload copied to clipboard");
        });
      }
    });
  }

  // ========================================================================
  // 9. Video Feed Audit Pipeline
  // ========================================================================
  const videoDropzone = document.getElementById("videoDropzone");
  const videoFileInput = document.getElementById("videoFileInput");
  const videoProcessingState = document.getElementById("videoProcessingState");
  const videoResultsPanel = document.getElementById("videoResultsPanel");
  const videoEmptyPrompt = document.getElementById("videoEmptyPrompt");

  if (videoDropzone && videoFileInput) {
    videoDropzone.addEventListener("click", () => videoFileInput.click());

    videoDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      videoDropzone.classList.add("dragover");
    });
    videoDropzone.addEventListener("dragleave", () => {
      videoDropzone.classList.remove("dragover");
    });
    videoDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      videoDropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        executeVideoAudit(e.dataTransfer.files[0]);
      }
    });
    videoFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        executeVideoAudit(e.target.files[0]);
      }
    });
  }

  function executeVideoAudit(file) {
    if (!file.type.startsWith("video/")) {
      showToast("Invalid format. Please supply an MP4/AVI/MOV video.");
      return;
    }

    if (videoProcessingState) videoProcessingState.style.display = "block";
    if (videoResultsPanel) videoResultsPanel.style.display = "none";
    if (videoEmptyPrompt) videoEmptyPrompt.style.display = "none";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("sample_interval", "0.5");

    fetch("/api/predict/video", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        if (videoProcessingState) videoProcessingState.style.display = "none";
        if (data.status === "error") {
          showToast(`Audit failed: ${data.message}`);
          if (videoEmptyPrompt) videoEmptyPrompt.style.display = "block";
        } else {
          renderVideoAuditResults(data);
        }
      })
      .catch((err) => {
        if (videoProcessingState) videoProcessingState.style.display = "none";
        if (videoEmptyPrompt) videoEmptyPrompt.style.display = "block";
        showToast("Error processing video file.");
      });
  }

  function renderVideoAuditResults(data) {
    if (!videoResultsPanel) return;
    videoResultsPanel.style.display = "block";

    const kpiOverall = document.getElementById("kpiOverall");
    if (kpiOverall) {
      kpiOverall.textContent = data.overall_prediction.toUpperCase();
      kpiOverall.style.color =
        data.overall_prediction === "Fire"
          ? "var(--hazard-fire)"
          : data.overall_prediction === "Smoke"
          ? "var(--hazard-smoke)"
          : "var(--hazard-safe)";
    }

    const kpiFire = document.getElementById("kpiFire");
    if (kpiFire) kpiFire.textContent = `${data.stats?.fire_percent || 0}%`;

    const kpiSmoke = document.getElementById("kpiSmoke");
    if (kpiSmoke) kpiSmoke.textContent = `${data.stats?.smoke_percent || 0}%`;

    const kpiFrames = document.getElementById("kpiFrames");
    if (kpiFrames) kpiFrames.textContent = data.sampled_frames || 0;

    // Build timeline feed
    const timelineFeed = document.getElementById("videoTimelineFeed");
    if (timelineFeed && data.timeline) {
      timelineFeed.innerHTML = "";
      data.timeline.forEach((item) => {
        const row = document.createElement("div");
        row.className = "incident-row";
        const tagClass =
          item.prediction === "Fire"
            ? "tag-fire"
            : item.prediction === "Smoke"
            ? "tag-smoke"
            : "tag-safe";

        row.innerHTML = `
          <span>TIMESTAMP: ${item.timestamp.toFixed(1)}s (Frame #${item.frame})</span>
          <span class="pill-threat-tag ${tagClass}">${item.prediction} ${item.confidence.toFixed(1)}%</span>
        `;
        timelineFeed.appendChild(row);
      });
    }

    if (data.is_hazard) {
      triggerHazardSiren(data.overall_prediction);
    }
  }

  // ========================================================================
  // 10. Live CCTV Surveillance Engine
  // ========================================================================
  const btnStartCamera = document.getElementById("btnStartCamera");
  const btnStopCamera = document.getElementById("btnStopCamera");
  const btnCaptureSnapshot = document.getElementById("btnCaptureSnapshot");
  const cameraSelect = document.getElementById("cameraSelect");

  const webcamVideo = document.getElementById("webcamVideo");
  const webcamCanvas = document.getElementById("webcamCanvas");
  const cctvAssessmentBanner = document.getElementById("cctvAssessmentBanner");
  const cctvThreatHeading = document.getElementById("cctvThreatHeading");
  const cctvThreatSub = document.getElementById("cctvThreatSub");
  const cctvThreatPct = document.getElementById("cctvThreatPct");
  const cctvThreatIcon = document.getElementById("cctvThreatIcon");

  const hudFpsCounter = document.getElementById("hudFpsCounter");
  const hudActiveThreatTag = document.getElementById("hudActiveThreatTag");
  const liveIncidentFeed = document.getElementById("liveIncidentFeed");
  const btnClearLog = document.getElementById("btnClearLog");

  let cctvStream = null;
  let cctvInterval = null;
  let isCctvLive = false;
  let isFrameInFlight = false;
  let fpsSampleCount = 0;
  let fpsTimerStart = Date.now();
  let latestCctvData = null;

  // Enumerate Connected Camera Devices
  async function populateCameraDevices() {
    if (!cameraSelect || !navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((d) => d.kind === "videoinput");
      if (videoDevices.length > 0) {
        cameraSelect.innerHTML = "";
        videoDevices.forEach((dev, idx) => {
          const opt = document.createElement("option");
          opt.value = dev.deviceId;
          opt.textContent = dev.label || `Optical Sensor Channel 0${idx + 1}`;
          cameraSelect.appendChild(opt);
        });
      }
    } catch (e) {
      console.warn("Could not enumerate camera devices:", e);
    }
  }
  populateCameraDevices();

  if (btnStartCamera) {
    btnStartCamera.addEventListener("click", startCctvStream);
  }
  if (btnStopCamera) {
    btnStopCamera.addEventListener("click", stopCctvStream);
  }

  async function startCctvStream() {
    initAudioContext();
    const deviceId = cameraSelect?.value;
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
      audio: false,
    };

    try {
      cctvStream = await navigator.mediaDevices.getUserMedia(constraints);
      webcamVideo.srcObject = cctvStream;
      await webcamVideo.play();

      isCctvLive = true;
      btnStartCamera.style.display = "none";
      btnStopCamera.style.display = "inline-flex";
      if (btnCaptureSnapshot) btnCaptureSnapshot.disabled = false;

      if (cctvThreatHeading) cctvThreatHeading.textContent = "SURVEILLANCE ENGAGED";
      if (cctvThreatSub) cctvThreatSub.textContent = "Active perimeter scan in progress...";

      // Re-populate devices with authorized device labels
      populateCameraDevices();

      // Poll frames every 350ms for low-latency near real-time detection
      cctvInterval = setInterval(pollWebcamFrame, 350);
      showToast("CCTV surveillance stream engaged");
    } catch (err) {
      alert("Unable to access optical sensor: " + err.message);
    }
  }

  function stopCctvStream() {
    if (cctvInterval) {
      clearInterval(cctvInterval);
      cctvInterval = null;
    }
    if (cctvStream) {
      cctvStream.getTracks().forEach((t) => t.stop());
      cctvStream = null;
    }
    if (webcamVideo) {
      webcamVideo.srcObject = null;
    }

    isCctvLive = false;
    latestCctvData = null;
    btnStartCamera.style.display = "inline-flex";
    btnStopCamera.style.display = "none";
    if (btnCaptureSnapshot) btnCaptureSnapshot.disabled = true;

    if (cctvAssessmentBanner) cctvAssessmentBanner.className = "assessment-banner";
    if (cctvThreatHeading) cctvThreatHeading.textContent = "SENSOR OFFLINE";
    if (cctvThreatSub) cctvThreatSub.textContent = "Activate video ingestion to initiate real-time AI perimeter surveillance";
    if (cctvThreatPct) cctvThreatPct.textContent = "--%";
    if (hudActiveThreatTag) hudActiveThreatTag.textContent = "STANDBY";
    if (hudFpsCounter) hudFpsCounter.textContent = "INFERENCE: 0.0 FPS";
    updateGlobalDefconState("Neutral", 0, null);
  }

  async function pollWebcamFrame() {
    if (!isCctvLive || isFrameInFlight || !webcamVideo || !webcamCanvas) return;

    const ctx = webcamCanvas.getContext("2d");
    webcamCanvas.width = 320;
    webcamCanvas.height = 240;

    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
    const frameBase64 = webcamCanvas.toDataURL("image/jpeg", 0.7);

    isFrameInFlight = true;
    try {
      const res = await fetch("/api/predict/frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: frameBase64 }),
      });
      const data = await res.json();
      latestCctvData = data;
      renderCctvTelemetry(data);

      fpsSampleCount++;
      const now = Date.now();
      if (now - fpsTimerStart >= 1000) {
        const measuredFps = ((fpsSampleCount * 1000) / (now - fpsTimerStart)).toFixed(1);
        if (hudFpsCounter) hudFpsCounter.textContent = `INFERENCE: ${measuredFps} FPS`;
        fpsSampleCount = 0;
        fpsTimerStart = now;
      }
    } catch (e) {
      console.warn("CCTV frame infer error:", e);
    } finally {
      isFrameInFlight = false;
    }
  }

  function renderCctvTelemetry(data) {
    if (!data || !cctvAssessmentBanner) return;

    const pred = data.prediction || "Neutral";
    const confidence = data.confidence || 0.0;
    const isHazard = data.is_hazard;

    // Sync DEFCON state, atmospheric lighting, and tactical diagnostic canvases
    updateGlobalDefconState(pred, confidence, data.latency_ms);

    cctvAssessmentBanner.className = "assessment-banner";
    if (pred === "Fire") {
      cctvAssessmentBanner.classList.add("threat-fire");
    } else if (pred === "Smoke") {
      cctvAssessmentBanner.classList.add("threat-smoke");
    } else {
      cctvAssessmentBanner.classList.add("threat-safe");
    }

    if (cctvThreatHeading) {
      cctvThreatHeading.textContent =
        pred === "Fire"
          ? "CRITICAL HAZARD - FIRE DETECTED"
          : pred === "Smoke"
          ? "HIGH ADVISORY - SMOKE DETECTED"
          : "STATUS SECURE - PERIMETER CLEAR";
    }

    if (cctvThreatSub) {
      cctvThreatSub.textContent = isHazard
        ? `Optical trigger: ${data.hazard_level || "ALERT"} state detected by neural classifier.`
        : "Perimeter clear. No thermal combustion signatures detected.";
    }

    if (cctvThreatPct) {
      cctvThreatPct.textContent = `${confidence.toFixed(1)}%`;
      cctvThreatPct.style.color = data.color_hex || (isHazard ? "#ef4444" : "#10b981");
    }

    if (hudActiveThreatTag) {
      if (isHazard) {
        hudActiveThreatTag.textContent = `THREAT: ${pred.toUpperCase()} (${confidence.toFixed(0)}%)`;
        hudActiveThreatTag.style.color = data.color_hex || (pred === "Fire" ? "var(--hazard-fire)" : "var(--hazard-smoke)");
      } else {
        hudActiveThreatTag.textContent = "SECURE - CLEAR";
        hudActiveThreatTag.style.color = "var(--hazard-safe)";
      }
    }

    // Trigger audible alarm if verified hazard reaches sensitivity threshold
    if (isHazard && confidence >= detectionThreshold) {
      triggerHazardSiren(pred);
      logIncidentFeed(pred, confidence);
    }
  }

  // Incident Feed Logger
  let lastLoggedIncidentTime = 0;
  function logIncidentFeed(predClass, conf) {
    const now = Date.now();
    // Throttle log entries to once per 2.5 seconds to prevent spamming
    if (now - lastLoggedIncidentTime < 2500 || !liveIncidentFeed) return;
    lastLoggedIncidentTime = now;

    const timeStr = new Date().toTimeString().split(" ")[0];
    const row = document.createElement("div");
    row.className = "incident-row";
    const tagClass = predClass === "Fire" ? "tag-fire" : "tag-smoke";

    row.innerHTML = `
      <span>[${timeStr}] CAM-01 ANOMALY</span>
      <span class="pill-threat-tag ${tagClass}">${predClass} ${conf.toFixed(1)}%</span>
    `;

    // Remove standby message if first log
    if (liveIncidentFeed.children.length === 1 && liveIncidentFeed.children[0].textContent.includes("Waiting")) {
      liveIncidentFeed.innerHTML = "";
    }

    liveIncidentFeed.prepend(row);
  }

  if (btnClearLog && liveIncidentFeed) {
    btnClearLog.addEventListener("click", () => {
      liveIncidentFeed.innerHTML = `
        <div class="incident-row" style="color: var(--text-tertiary);">
          <span>Incident log register cleared. Surveillance active.</span>
        </div>
      `;
      showToast("Incident stream cleared");
    });
  }

  // Snapshot Capture & Download with Bounding Box Overlay
  if (btnCaptureSnapshot && webcamVideo) {
    btnCaptureSnapshot.addEventListener("click", () => {
      if (!isCctvLive) return;

      const snapCanvas = document.createElement("canvas");
      snapCanvas.width = webcamVideo.videoWidth || 640;
      snapCanvas.height = webcamVideo.videoHeight || 480;
      const snapCtx = snapCanvas.getContext("2d");

      // Draw mirrored video frame
      snapCtx.translate(snapCanvas.width, 0);
      snapCtx.scale(-1, 1);
      snapCtx.drawImage(webcamVideo, 0, 0, snapCanvas.width, snapCanvas.height);
      snapCtx.setTransform(1, 0, 0, 1, 0, 0);

      // Overlay watermark stamp
      snapCtx.fillStyle = "rgba(13, 18, 28, 0.85)";
      snapCtx.fillRect(10, 10, 260, 36);
      snapCtx.fillStyle = latestCctvData?.is_hazard ? "#ef4444" : "#10b981";
      snapCtx.font = "bold 13px 'JetBrains Mono', monospace";
      const statusStamp = latestCctvData?.is_hazard ? `HAZARD: ${latestCctvData.prediction.toUpperCase()}` : "STATUS: SECURE";
      snapCtx.fillText(`FIRE-SMOKE AI | ${statusStamp}`, 20, 33);

      const dataUrl = snapCanvas.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `CCTV_Snapshot_${Date.now()}.png`;
      a.click();
      showToast("High-resolution surveillance snapshot downloaded");
    });
  }

  // ========================================================================
  // 11. API Copy Code Buttons
  // ========================================================================
  const copyButtons = document.querySelectorAll(".btn-copy-code[data-copy]");
  copyButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const codeToCopy = btn.getAttribute("data-copy");
      if (codeToCopy && navigator.clipboard) {
        navigator.clipboard.writeText(codeToCopy).then(() => {
          showToast("cURL command copied to clipboard");
        });
      }
    });
  });
});
