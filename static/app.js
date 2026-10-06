(() => {
  // ---------------- Element refs ----------------
  const sampleBtn = document.getElementById("sampleBtn");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const blurToggle = document.getElementById("blurEyesToggle");
  const resetBtn = document.getElementById("resetBtn");
  const crossCheckBtn = document.getElementById("crossCheckBtn");
  const fullReportBtn = document.getElementById("fullReportBtn");
  const printBtn = document.getElementById("printBtn");
  const blendSlider = document.getElementById("blendSlider");
  const tryAnotherBtn = document.getElementById("tryAnotherBtn");

  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const loadingSteps = document.getElementById("loadingSteps");
  const errorState = document.getElementById("errorState");
  const rejectedState = document.getElementById("rejectedState");
  const rejectedMessage = document.getElementById("rejectedMessage");
  const rejectedSampleNote = document.getElementById("rejectedSampleNote");
  const rejectedFilename = document.getElementById("rejectedFilename");
  const resultState = document.getElementById("resultState");

  const historyStrip = document.getElementById("historyStrip");
  const historyEmpty = document.getElementById("historyEmpty");

  // ---------------- Session state ----------------
  let seq = 0;
  const history = [];          // [{id, data, context}]
  let activeId = null;
  let currentContext = null;   // {mode: "upload"|"sample", file, filename, blurEyes}

  const STEP_LABELS = {
    quick: ["Reading the image", "Checking the face", "Running the classifier", "Generating the Grad-CAM explanation"],
    cross_check: [
      "Reading the image",
      "Checking the face",
      "Running the classifier",
      "Generating the Grad-CAM explanation",
      "Cross-checking with Score-CAM",
    ],
    full: [
      "Reading the image",
      "Checking the face",
      "Running the classifier",
      "Generating the Grad-CAM explanation",
      "Cross-checking with Score-CAM",
      "Running LIME (1,500 samples — the slow part)",
      "Comparing all three methods",
    ],
  };
  const STEP_TIMING = {
    quick: [0, 250, 600, 1100],
    cross_check: [0, 250, 600, 1100, 3500],
    full: [0, 250, 600, 1100, 3500, 8000, 30000],
  };

  // ================================================================
  // Input handling
  // ================================================================
  sampleBtn.addEventListener("click", () => {
    runAnalysis({ mode: "sample", filename: null }, "quick");
  });

  tryAnotherBtn.addEventListener("click", () => {
    runAnalysis({ mode: "sample", filename: null }, "quick");
  });

  function startUpload(file) {
    // The eye-blur choice is remembered with the upload, so follow-up
    // analyses (cross-check, full report) of the same photo keep it.
    runAnalysis({ mode: "upload", file, blurEyes: blurToggle.checked }, "quick");
  }

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
    if (file) startUpload(file);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) startUpload(fileInput.files[0]);
    fileInput.value = "";
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
      if (context.blurEyes) form.append("blur_eyes", "true");
    } else if (context.filename) {
      form.append("filename", context.filename);
    }

    try {
      const res = await fetch(url, { method: "POST", body: form });
      const data = await res.json();

      // 422 = the face check refused the photo before any analysis.
      if (res.status === 422 && data.rejected) {
        showRejected(data);
        return;
      }
      if (!res.ok) {
        showError(data.message || data.error || "Something went wrong.");
        return;
      }

      finishLoadingSteps();
      if (context.mode === "sample") {
        currentContext = { mode: "sample", filename: data.filename };
      }

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
  // View states
  // ================================================================
  let stepTimers = [];

  function hideAllStates() {
    emptyState.hidden = true;
    loadingState.hidden = true;
    errorState.hidden = true;
    rejectedState.hidden = true;
    resultState.hidden = true;
  }

  function showLoading(mode) {
    hideAllStates();
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
    hideAllStates();
    errorState.hidden = false;
    errorState.textContent = msg;
  }

  // A photo the face check refused. Not an error: it is the app working as
  // designed, so it gets its own calm panel with advice instead of red text.
  function showRejected(data) {
    clearStepTimers();
    hideAllStates();

    const isSample = data.source === "sample";
    rejectedMessage.textContent = data.message || "This photo could not be analysed.";
    rejectedSampleNote.hidden = !isSample;
    rejectedFilename.textContent = isSample ? data.filename || "" : "";
    tryAnotherBtn.hidden = !isSample;

    rejectedState.hidden = false;
    resetBtn.hidden = false;
    currentContext = null;
    activeId = null;
    renderHistory();
  }

  function resetToEmpty() {
    clearStepTimers();
    hideAllStates();
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

  // Show `text` in element `el`, or hide the element when there is none.
  // Always textContent: server text is never interpreted as HTML.
  function setNotice(el, text) {
    if (text) {
      el.textContent = text;
      el.hidden = false;
    } else {
      el.textContent = "";
      el.hidden = true;
    }
  }

  function fmt3(v) {
    return typeof v === "number" ? v.toFixed(3) : "—";
  }

  function showResult(data) {
    hideAllStates();
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

    // Warnings and notes from the face check and the region analysis
    setNotice(document.getElementById("offFaceWarning"), data.off_face_warning);
    setNotice(document.getElementById("tiltNotice"), data.tilt_warning ? data.tilt_message : null);
    document.getElementById("cropNotice").hidden = data.face_cropped !== true;
    document.getElementById("fallbackNotice").hidden = data.face_detector_used_fallback !== true;

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

    // Privacy: shown under every image whose eyes were pixelated
    setNotice(document.getElementById("privacyCaption"), data.eyes_blurred ? data.privacy_caption : null);

    const scBlock = document.getElementById("scorecamBlock");
    if (data.scorecam_image) {
      scBlock.hidden = false;
      document.getElementById("imgScorecam").src = data.scorecam_image;
    } else {
      scBlock.hidden = true;
    }

    const limeBlock = document.getElementById("limeBlock");
    if (data.lime_image) {
      limeBlock.hidden = false;
      document.getElementById("imgLime").src = data.lime_image;
    } else {
      limeBlock.hidden = true;
    }

    const panel4Block = document.getElementById("panel4Block");
    if (data.panel_4_image) {
      panel4Block.hidden = false;
      document.getElementById("imgPanel4").src = data.panel_4_image;
    } else {
      panel4Block.hidden = true;
    }

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

    // Region analysis: outlines + bar chart, then the plain-language summary
    const regionBlock = document.getElementById("regionBlock");
    if (data.region_figure_image) {
      regionBlock.hidden = false;
      document.getElementById("imgRegions").src = data.region_figure_image;
    } else {
      regionBlock.hidden = true;
    }
    document.getElementById("regionSummary").textContent = data.region_summary;

    // Faithfulness (full report only)
    const faithBlock = document.getElementById("faithfulnessBlock");
    const f = data.faithfulness;
    if (f) {
      faithBlock.hidden = false;
      document.getElementById("faithGradDel").textContent = fmt3(f.GradCAM_DeletionAUC);
      document.getElementById("faithGradIns").textContent = fmt3(f.GradCAM_InsertionAUC);
      document.getElementById("faithScoreDel").textContent = fmt3(f.ScoreCAM_DeletionAUC);
      document.getElementById("faithScoreIns").textContent = fmt3(f.ScoreCAM_InsertionAUC);
    } else {
      faithBlock.hidden = true;
    }

    const tier = tierOf(data);
    crossCheckBtn.hidden = tier !== "quick";
    fullReportBtn.hidden = tier === "full";
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
      const thumb = document.createElement("img");
      thumb.src = entry.data.gradcam_image;
      thumb.alt = "";
      btn.appendChild(thumb);
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