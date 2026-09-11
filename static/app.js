(() => {
  // ---------------- Element refs ----------------
  const sampleBtn = document.getElementById("sampleBtn");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const resetBtn = document.getElementById("resetBtn");
  const crossCheckBtn = document.getElementById("crossCheckBtn");
  const fullReportBtn = document.getElementById("fullReportBtn");
  const printBtn = document.getElementById("printBtn");
  const blendSlider = document.getElementById("blendSlider");

  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const loadingSteps = document.getElementById("loadingSteps");
  const errorState = document.getElementById("errorState");
  const resultState = document.getElementById("resultState");

  const historyStrip = document.getElementById("historyStrip");
  const historyEmpty = document.getElementById("historyEmpty");

  // ---------------- Session state ----------------
  let seq = 0;
  const history = [];          // [{id, data, context}]
  let activeId = null;
  let currentContext = null;   // {mode: "upload"|"sample", file, filename}

  const STEP_LABELS = {
    quick: ["Reading the image", "Running the classifier", "Generating the Grad-CAM explanation"],
    cross_check: ["Reading the image", "Running the classifier", "Generating the Grad-CAM explanation", "Cross-checking with Score-CAM"],
    full: [
      "Reading the image",
      "Running the classifier",
      "Generating the Grad-CAM explanation",
      "Cross-checking with Score-CAM",
      "Running LIME (1,500 samples — the slow part)",
      "Comparing all three methods",
    ],
  };
  // Rough timing (ms) for when each step should appear to start — a UI
  // approximation to keep a long wait from looking frozen, not telemetry
  // from the server. "full" mode's LIME step is intentionally the longest.
  const STEP_TIMING = {
    quick: [0, 300, 900],
    cross_check: [0, 300, 900, 3500],
    full: [0, 300, 900, 3500, 8000, 30000],
  };

  // ================================================================
  // Input handling
  // ================================================================
  sampleBtn.addEventListener("click", () => {
    runAnalysis({ mode: "sample", filename: null }, "quick");
  });

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dropzone--drag");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dropzone--drag");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) runAnalysis({ mode: "upload", file }, "quick");
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) runAnalysis({ mode: "upload", file: fileInput.files[0] }, "quick");
    fileInput.value = ""; // allow re-selecting the same file later
  });

  crossCheckBtn.addEventListener("click", () => {
    if (!currentContext) return;
    runAnalysis(currentContext, "cross_check", { updateExisting: true });
  });
  fullReportBtn.addEventListener("click", () => {
    if (!currentContext) return;
    runAnalysis(currentContext, "full", { updateExisting: true });
  });

  resetBtn.addEventListener("click", resetToEmpty);
  printBtn.addEventListener("click", () => window.print());

  blendSlider.addEventListener("input", () => {
    document.getElementById("imgGradcam").style.opacity = blendSlider.value / 100;
  });

  // ================================================================
  // Core request flow
  // ================================================================
  async function runAnalysis(context, mode, opts = {}) {
    currentContext = context;
    showLoading(mode);

    const form = new FormData();
    form.append("mode", mode);
    let url = "/predict/sample";
    if (context.mode === "upload") {
      url = "/predict/upload";
      form.append("image", context.file);
    } else if (context.filename) {
      form.append("filename", context.filename);
    }

    try {
      const res = await fetch(url, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        showError(data.error || "Something went wrong.");
        return;
      }
      finishLoadingSteps();
      // Remember the exact sample filename so a follow-up tier re-analyses
      // this same image rather than drawing a new random one.
      if (context.mode === "sample") currentContext = { mode: "sample", filename: data.filename };

      if (opts.updateExisting && activeId !== null) {
        const entry = history.find((h) => h.id === activeId);
        if (entry) entry.data = data;
      } else {
        addHistoryEntry(data, currentContext);
      }
      showResult(data);
    } catch (err) {
      showError("Could not reach the server: " + err.message);
    }
  }

  // ================================================================
  // Loading state (staged, approximate progress)
  // ================================================================
  let stepTimers = [];

  function showLoading(mode) {
    emptyState.hidden = true;
    errorState.hidden = true;
    resultState.hidden = true;
    loadingState.hidden = false;
    resetBtn.hidden = false;

    const labels = STEP_LABELS[mode] || STEP_LABELS.quick;
    const timing = STEP_TIMING[mode] || STEP_TIMING.quick;
    loadingSteps.innerHTML = labels.map((l) => `<li>${l}</li>`).join("");
    const items = loadingSteps.querySelectorAll("li");

    clearStepTimers();
    timing.forEach((delay, i) => {
      stepTimers.push(
        setTimeout(() => {
          items.forEach((el, j) => {
            el.classList.toggle("step--active", j === i);
            el.classList.toggle("step--done", j < i);
          });
        }, delay)
      );
    });
  }

  function finishLoadingSteps() {
    clearStepTimers();
    loadingSteps.querySelectorAll("li").forEach((el) => {
      el.classList.remove("step--active");
      el.classList.add("step--done");
    });
  }

  function clearStepTimers() {
    stepTimers.forEach(clearTimeout);
    stepTimers = [];
  }

  function showError(msg) {
    clearStepTimers();
    loadingState.hidden = true;
    resultState.hidden = true;
    emptyState.hidden = true;
    errorState.hidden = false;
    errorState.textContent = msg;
  }

  function resetToEmpty() {
    clearStepTimers();
    loadingState.hidden = true;
    errorState.hidden = true;
    resultState.hidden = true;
    emptyState.hidden = false;
    resetBtn.hidden = true;
    activeId = null;
    currentContext = null;
    setHistoryActive(null);
  }

  // ================================================================
  // Result rendering
  // ================================================================
  function tierOf(data) {
    if (data.overlap) return "full";
    if (data.scorecam_image) return "cross_check";
    return "quick";
  }

  function showResult(data) {
    loadingState.hidden = true;
    errorState.hidden = true;
    emptyState.hidden = true;
    resultState.hidden = false;
    resetBtn.hidden = false;

    document.getElementById("predLabel").textContent = data.prediction;

    const badge = document.getElementById("confBadge");
    badge.classList.remove("tier-high", "tier-moderate", "tier-low");
    badge.classList.add(`tier-${data.confidence_tier}`);
    const tierWord = { high: "High confidence", moderate: "Moderate confidence", low: "Low confidence" }[data.confidence_tier] || "Confidence";
    document.getElementById("confText").textContent = `${tierWord} · ${data.confidence}%`;
    document.getElementById("lowConfNote").hidden = data.confidence_tier !== "low";

    document.getElementById("noFaceBanner").hidden = data.face_detected !== false;

    document.getElementById("probNon").textContent = `${data.prob_non_autistic}%`;
    document.getElementById("probAsd").textContent = `${data.prob_autistic}%`;
    document.getElementById("probNonFill").style.width = `${data.prob_non_autistic}%`;
    document.getElementById("probAsdFill").style.width = `${data.prob_autistic}%`;

    const sampleMeta = document.getElementById("sampleMeta");
    if (data.source === "sample") {
      sampleMeta.hidden = false;
      document.getElementById("sampleFilename").textContent = data.filename;
      document.getElementById("sampleTrue").textContent = data.true_label
        ? `· labelled ${data.true_label} in the test set`
        : "";
    } else {
      sampleMeta.hidden = true;
    }

    document.getElementById("imgOriginal").src = data.original_image;
    const gradImg = document.getElementById("imgGradcam");
    gradImg.src = data.gradcam_image;
    gradImg.style.opacity = blendSlider.value / 100;

    // Score-CAM
    const scBlock = document.getElementById("scorecamBlock");
    if (data.scorecam_image) {
      scBlock.hidden = false;
      document.getElementById("imgScorecam").src = data.scorecam_image;
    } else {
      scBlock.hidden = true;
    }

    // LIME
    const limeBlock = document.getElementById("limeBlock");
    if (data.lime_image) {
      limeBlock.hidden = false;
      document.getElementById("imgLime").src = data.lime_image;
    } else {
      limeBlock.hidden = true;
    }

    // 4-panel comparison figure
    const panel4Block = document.getElementById("panel4Block");
    if (data.panel_4_image) {
      panel4Block.hidden = false;
      document.getElementById("imgPanel4").src = data.panel_4_image;
    } else {
      panel4Block.hidden = true;
    }

    // Cross-method overlap / agreement grid
    const overlapBlock = document.getElementById("overlapBlock");
    if (data.overlap) {
      overlapBlock.hidden = false;
      const o = data.overlap;
      document.getElementById("imgOverlapGS").src = o.grad_vs_score.image;
      document.getElementById("metricGS").textContent = `IoU ${o.grad_vs_score.iou} · Spearman ${o.grad_vs_score.spearman}`;
      document.getElementById("imgOverlapGL").src = o.grad_vs_lime.image;
      document.getElementById("metricGL").textContent = `IoU ${o.grad_vs_lime.iou} · Spearman ${o.grad_vs_lime.spearman}`;
      document.getElementById("imgOverlapSL").src = o.score_vs_lime.image;
      document.getElementById("metricSL").textContent = `IoU ${o.score_vs_lime.iou} · Spearman ${o.score_vs_lime.spearman}`;
      document.getElementById("imgOverlap3").src = o.three_way.image;
      document.getElementById("metric3").textContent =
        `IoU(G,S) ${o.three_way.iou_grad_score} · IoU(G,L) ${o.three_way.iou_grad_lime} · IoU(S,L) ${o.three_way.iou_score_lime}`;
    } else {
      overlapBlock.hidden = true;
    }

    // Tier action buttons: only offer tiers beyond the current one
    const tier = tierOf(data);
    crossCheckBtn.hidden = tier !== "quick";
    fullReportBtn.hidden = tier === "full";

    document.getElementById("regionSummary").textContent = data.region_summary;
  }

  // ================================================================
  // Session history strip
  // ================================================================
  function addHistoryEntry(data, context) {
    seq += 1;
    const id = seq;
    history.unshift({ id, data, context });
    if (history.length > 8) history.pop();
    activeId = id;
    renderHistory();
  }

  function renderHistory() {
    historyEmpty.hidden = history.length > 0;
    historyStrip.innerHTML = "";
    history.forEach((entry) => {
      const btn = document.createElement("button");
      btn.className = "history__item" + (entry.id === activeId ? " history__item--active" : "");
      btn.title = `${entry.data.prediction} · ${entry.data.confidence}%`;
      btn.innerHTML = `<img src="${entry.data.gradcam_image}" alt="">`;
      btn.addEventListener("click", () => {
        activeId = entry.id;
        currentContext = entry.context;
        showResult(entry.data);
        renderHistory();
      });
      historyStrip.appendChild(btn);
    });
  }

  function setHistoryActive(id) {
    activeId = id;
    renderHistory();
  }
})();