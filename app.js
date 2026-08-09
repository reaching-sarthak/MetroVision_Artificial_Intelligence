const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const dropzoneInner = document.getElementById("dropzoneInner");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("preview");
const replaceBtn = document.getElementById("replaceBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMsg = document.getElementById("errorMsg");
const statusEl = document.getElementById("status");

const resultPanel = document.getElementById("resultPanel");
const verdictBar = document.getElementById("verdictBar");
const verdictClass = document.getElementById("verdictClass");
const verdictConfidence = document.getElementById("verdictConfidence");
const breakdown = document.getElementById("breakdown");
const metaEl = document.getElementById("meta");

const tabAssess = document.getElementById("tabAssess");
const tabAbout = document.getElementById("tabAbout");
const assessView = document.getElementById("assessView");
const aboutView = document.getElementById("aboutView");

let selectedFile = null;

const CLASS_KEY = {
  Pothole: "pothole",
  Crack: "crack",
  Both: "both",
  Normal: "normal",
};

// ---------- tabs ----------
function showTab(name) {
  const assessActive = name === "assess";
  assessView.hidden = !assessActive;
  aboutView.hidden = assessActive;
  tabAssess.classList.toggle("active", assessActive);
  tabAbout.classList.toggle("active", !assessActive);
  tabAssess.setAttribute("aria-selected", String(assessActive));
  tabAbout.setAttribute("aria-selected", String(!assessActive));
}
tabAssess.addEventListener("click", () => showTab("assess"));
tabAbout.addEventListener("click", () => showTab("about"));

// ---------- model status ----------
fetch("/health")
  .then((r) => r.json())
  .then(() => {
    statusEl.textContent = "model ready";
    statusEl.classList.add("ok");
  })
  .catch(() => {
    statusEl.textContent = "model unavailable";
    statusEl.classList.add("error");
  });

// ---------- file selection ----------
// dropzone is a <label for="fileInput">, so clicking anywhere inside it
// (including the replace pill) already opens the file picker natively.
// This listener just reinforces it for the pill specifically.
replaceBtn?.addEventListener("click", (e) => {
  e.preventDefault();
  fileInput.click();
});
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("That's not an image file.");
    return;
  }
  hideError();
  selectedFile = file;

  const url = URL.createObjectURL(file);
  preview.src = url;
  dropzoneInner.hidden = true;
  previewWrap.hidden = false;

  analyzeBtn.disabled = false;
  resultPanel.hidden = true;
}

// ---------- analyze ----------
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  analyzeBtn.disabled = true;
  analyzeBtn.classList.add("loading");
  analyzeBtn.textContent = "Analyzing…";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong.");
      return;
    }
    renderResult(data);
  } catch (err) {
    showError("Couldn't reach the server. Is it still running?");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove("loading");
    analyzeBtn.textContent = "Analyze photo";
  }
});

function renderResult(data) {
  const key = CLASS_KEY[data.predicted_class] || "normal";

  verdictBar.className = "verdict-bar " + key;
  verdictClass.className = "verdict-class " + key;
  verdictClass.textContent = data.predicted_class;
  verdictConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  breakdown.innerHTML = "";
  const order = ["Pothole", "Crack", "Both", "Normal"];
  for (const className of order) {
    const p = data.probabilities[className] ?? 0;
    const rowKey = CLASS_KEY[className];

    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-label">${className}</span>
      <span class="bar-track"><span class="bar-fill ${rowKey}" style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span class="bar-value">${(p * 100).toFixed(1)}%</span>
    `;
    breakdown.appendChild(row);
  }

  metaEl.textContent = `checkpoint: ${data.checkpoint} · trained steps: ${data.trained_steps}`;
  resultPanel.hidden = false;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}
function hideError() {
  errorMsg.hidden = true;
}
