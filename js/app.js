const ALGORITHMS = [
  "Lasso (L1 Ranking)",
  "Forward SFS",
  "RFE",
  "SelectKBest",
  "Permutation Importance",
];
const COMPARE = "Compare all";

const state = {
  presets: [],
  problem: null,
  algo: "Lasso (L1 Ranking)",
  k: 2,
  scheme: null,
  customText: null,
  charts: {},
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function money(value, unit) {
  if (!Number.isFinite(value)) return "—";
  const digits = Math.abs(value) >= 100 ? 2 : 4;
  const formatted = value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  if (unit === "USD") return `$${formatted}`;
  if (unit) return `${formatted} ${unit}`;
  return formatted;
}

function formatP(value) {
  if (!Number.isFinite(value)) return "—";
  if (value < 0.001) return "< 0.001";
  return value.toFixed(4);
}

function fixed(value, digits) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function formatFeature(features) {
  return `[${features.join(", ")}]`;
}

function currentModels() {
  if (state.algo === COMPARE) return null;
  return state.problem.algorithms[state.algo];
}

function formatMse(value) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 100) {
    return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return value.toFixed(4);
}

function currentModel() {
  if (state.scheme != null && state.problem.bestSchemes) {
    return state.problem.bestSchemes[state.scheme];
  }
  const models = state.algo === COMPARE
    ? flattenModels().filter((item) => item.k === state.k)
    : currentModels();
  if (!models || !models.length) return null;
  if (state.algo === COMPARE) {
    return models.slice().sort((a, b) => a.rmse - b.rmse)[0];
  }
  return models.find((model) => model.k === state.k) || models[models.length - 1];
}

function flattenModels() {
  return ALGORITHMS.flatMap((name) =>
    (state.problem.algorithms[name] || []).map((model) => ({ ...model, method: name }))
  );
}

function setStatus(message, isError) {
  const node = $("status");
  node.textContent = message;
  node.classList.toggle("error", Boolean(isError));
}

function renderProblemOptions() {
  const select = $("problem");
  select.innerHTML = state.presets
    .map((problem) => `<option value="${problem.id}">${problem.title}</option>`)
    .join("");
  if (state.problem?.id === "custom") {
    select.insertAdjacentHTML("beforeend", `<option value="custom">自訂資料集</option>`);
  }
  select.value = state.problem.id;
}

function renderPhases() {
  const steps = [
    ["1 商業理解", "決定要預測什麼"],
    ["2 資料理解", `${state.problem.n} 筆`],
    ["3 資料準備", state.problem.split],
    ["4 建模", "sklearn OLS + 5 種選變數"],
    ["5 評估", `最佳 RMSE ${money(state.problem.best.rmse, "")}`],
    ["6 部署", "這個頁面與預測框"],
  ];
  $("phases").innerHTML = steps
    .map(([name, text]) => `<div class="phase"><b>${name}</b><span>${text}</span></div>`)
    .join("");
}

function renderAlgorithms() {
  const names = [...ALGORITHMS, COMPARE];
  $("algorithms").innerHTML = names
    .map((name) => `<button type="button" data-algo="${name}" class="${state.scheme == null && name === state.algo ? "active" : ""}">${name}</button>`)
    .join("");
  $("algorithms").querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.scheme = null;
      state.algo = button.dataset.algo;
      const maxK = state.problem.encodedFeatures.length;
      if (state.k > maxK) state.k = maxK;
      render();
    });
  });
}

function renderK() {
  const maxK = state.problem.encodedFeatures.length;
  const range = $("k-range");
  range.min = 1;
  range.max = maxK;
  range.value = state.k;
  $("k-label").textContent = String(state.k);
  const model = currentModel();
  $("active-features").textContent = model
    ? `目前使用 ${model.k} 個特徵：${model.features.join("、")}`
    : "";
}

function renderMetrics() {
  const model = currentModel();
  const full = state.problem.fullModel;
  const overfit = full.rmse > model.rmse * 1.02;
  const mse = model.mse == null ? model.rmse ** 2 : model.mse;
  const cards = [
    ["Test MSE", formatMse(mse), "越低越好"],
    ["Test RMSE", money(model.rmse, state.problem.targetUnit === "USD" ? "USD" : ""), "測試集"],
    ["R-squared", model.r2.toFixed(4), "測試集"],
    ["Adjusted R²", model.adjR2 == null ? "—" : model.adjR2.toFixed(4), "懲罰多餘變數"],
  ];
  $("metrics").innerHTML = cards
    .map(([label, value, hint], index) => {
      const warn = index === 0 && overfit && state.k === full.k;
      return `<article class="metric ${warn ? "warn" : ""}"><div class="label">${label}</div><strong>${value}</strong><small>${hint}</small></article>`;
    })
    .join("");
}

function chartDatasets(key) {
  const names = state.algo === COMPARE ? ALGORITHMS : [state.algo];
  const colors = ["#0e6b67", "#8a4b08", "#8d342c", "#2c5aa0", "#5c4d7a"];
  return names.map((name, index) => ({
    label: name,
    data: state.problem.algorithms[name].map((model) => model[key]),
    borderColor: colors[ALGORITHMS.indexOf(name)],
    backgroundColor: colors[ALGORITHMS.indexOf(name)],
    tension: 0.15,
    pointRadius: state.problem.algorithms[name].map((model) => (model.k === state.k ? 6 : 3)),
  }));
}

function renderCharts() {
  const labels = state.problem.encodedFeatures.map((_, index) => String(index + 1));
  draw("rmse-chart", "rmse", labels, "RMSE");
  draw("r2-chart", "r2", labels, "R-squared");
}

function draw(canvasId, key, labels, yLabel) {
  if (state.charts[canvasId]) state.charts[canvasId].destroy();
  state.charts[canvasId] = new Chart($(canvasId), {
    type: "line",
    data: { labels, datasets: chartDatasets(key) },
    options: {
      responsive: true,
      plugins: { legend: { display: state.algo === COMPARE } },
      scales: {
        x: { title: { display: true, text: "Number of Features" } },
        y: { title: { display: true, text: yLabel } },
      },
    },
  });
}

function renderPath() {
  const rows = state.algo === COMPARE
    ? flattenModels().filter((model) => model.k === state.k)
    : currentModels().map((model) => ({ ...model, method: state.algo }));
  $("path-body").innerHTML = rows
    .map((model) => {
      return `<tr class="${model.k === state.k ? "active-row" : ""}">
        <td>${model.k}</td>
        <td>${model.method}</td>
        <td>${formatFeature(model.features)}</td>
        <td class="num">${model.rmse.toFixed(4)}</td>
        <td class="num">${model.r2.toFixed(6)}</td>
      </tr>`;
    })
    .join("");
}

function renderEquation() {
  const model = currentModel();
  const parts = model.coefficients.map((coef) => {
    if (coef.name === "Intercept") return money(coef.coef, "");
    const sign = coef.coef < 0 ? "−" : "+";
    const mag = Math.abs(coef.coef) >= 100 ? Math.abs(coef.coef).toFixed(2) : Math.abs(coef.coef).toFixed(4);
    return `${sign} ${mag} × (${coef.name})`;
  });
  $("equation").textContent = `${state.problem.target} = ${parts.join(" ")}`;
  $("coef-body").innerHTML = model.coefficients
    .map((coef) => {
      const pClass = coef.p < 0.05 ? "good" : "";
      const vif = coef.vif == null ? "—" : coef.vif === Infinity ? "∞" : coef.vif.toFixed(3);
      const vifClass = coef.vif > 10 ? "bad" : "";
      return `<tr>
        <td>${coef.name}</td>
        <td class="num">${fixed(coef.coef, 6)}</td>
        <td class="num">${fixed(coef.se, 6)}</td>
        <td class="num">${fixed(coef.t, 3)}</td>
        <td class="num ${pClass}">${formatP(coef.p)}</td>
        <td>[${fixed(coef.ciLow, 4)}, ${fixed(coef.ciHigh, 4)}]</td>
        <td class="num ${vifClass}">${vif}</td>
      </tr>`;
    })
    .join("");
}

function predictorValue(key) {
  const existing = document.querySelector(`[data-input="${CSS.escape(key)}"]`);
  if (existing) return existing.value;
  const example = state.problem.deployed?.exampleInputs || {};
  if (example[key] != null) return example[key];
  const field = state.problem.predictors.find((item) => item.key === key);
  return field?.kind === "number" ? field.default : field?.reference;
}

function renderPredictor() {
  const model = currentModel();
  const used = new Set(model.features);
  const fields = state.problem.predictors.filter((field) => {
    if (field.kind === "number") return used.has(field.key);
    return Object.values(field.dummies).some((dummy) => used.has(dummy));
  });
  $("predictor").innerHTML = fields
    .map((field) => {
      if (field.kind === "number") {
        const value = predictorValue(field.key);
        return `<label class="field">${esc(field.label)}
          <input data-input="${esc(field.key)}" type="number" step="any" value="${esc(value)}">
        </label>`;
      }
      const value = predictorValue(field.key);
      const options = field.options
        .map((option) => `<option ${option === value ? "selected" : ""}>${option}</option>`)
        .join("");
      return `<label class="field">${esc(field.label)}
        <select data-input="${esc(field.key)}">${options}</select>
      </label>`;
    })
    .join("");
  $("predictor").querySelectorAll("input, select").forEach((node) => {
    node.addEventListener("input", updatePrediction);
  });
  $("prediction-label").textContent = `預測 ${state.problem.target}`;
  updatePrediction();
}

function updatePrediction() {
  const model = currentModel();
  const encoded = {};
  state.problem.predictors.forEach((field) => {
    const node = document.querySelector(`[data-input="${CSS.escape(field.key)}"]`);
    if (!node) return;
    if (field.kind === "number") encoded[field.key] = Number(node.value);
    else {
      Object.entries(field.dummies).forEach(([level, dummy]) => {
        encoded[dummy] = node.value === level ? 1 : 0;
      });
    }
  });
  let total = 0;
  model.coefficients.forEach((coef) => {
    if (coef.name === "Intercept") total += coef.coef;
    else total += coef.coef * (encoded[coef.name] || 0);
  });
  $("prediction").textContent = money(total, state.problem.targetUnit);
}

function renderResiduals() {
  const model = currentModel();
  $("resid-body").innerHTML = model.residuals
    .map((row) => `<tr>
      <td>${row.rank}</td>
      <td>${row.id}</td>
      <td class="num">${row.actual.toLocaleString("en-US", { maximumFractionDigits: 2 })}</td>
      <td class="num">${row.predicted.toLocaleString("en-US", { maximumFractionDigits: 2 })}</td>
      <td class="num">${row.residual.toLocaleString("en-US", { maximumFractionDigits: 2 })}</td>
      <td class="num">${row.pctError == null ? "—" : row.pctError.toFixed(2) + "%"}</td>
    </tr>`)
    .join("");
}

function renderSchemes() {
  const panel = $("schemes-panel");
  const schemes = state.problem.bestSchemes;
  if (!schemes) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("schemes").innerHTML = schemes
    .map((scheme, index) => {
      const label = `#${scheme.rank} k=${scheme.k} · ${scheme.features.join(", ")}`;
      return `<button type="button" data-scheme="${index}" class="${state.scheme === index ? "active" : ""}">${esc(label)}</button>`;
    })
    .join("");
  $("schemes").querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.scheme = Number(button.dataset.scheme);
      const scheme = state.problem.bestSchemes[state.scheme];
      state.k = scheme.k;
      render();
    });
  });
}

function renderTop20() {
  const panel = $("top20-panel");
  const rows = state.problem.top20;
  if (!rows) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("top20-body").innerHTML = rows
    .map((row) => `<tr class="${state.scheme != null && state.problem.bestSchemes[state.scheme].rank === row.rank ? "active-row" : ""}">
      <td>${row.rank}</td>
      <td>${row.k}</td>
      <td>${esc(formatFeature(row.features))}</td>
      <td class="num">${formatMse(row.mse)}</td>
      <td class="num">${row.rmse.toFixed(4)}</td>
      <td class="num">${row.r2.toFixed(6)}</td>
    </tr>`)
    .join("");
}

function render() {
  const problem = state.problem;
  $("problem-title").textContent = problem.title;
  document.title = `DIC2 多元線性迴歸 · ${problem.title}`;
  $("problem-note").textContent = `${problem.subtitle}。${problem.businessGoal} ${problem.notes} 訓練 ${problem.nTrain} / 測試 ${problem.nTest}。最低測試 RMSE：${problem.best.algorithm}，k=${problem.best.k}，${problem.best.rmse.toFixed(4)}。`;
  renderProblemOptions();
  renderPhases();
  renderAlgorithms();
  renderSchemes();
  renderK();
  renderMetrics();
  renderCharts();
  renderTop20();
  renderPath();
  renderEquation();
  renderPredictor();
  renderResiduals();
}

function selectProblem(problem, preferredK) {
  state.problem = problem;
  state.scheme = problem.bestSchemes ? 0 : null;
  state.algo = problem.algorithms?.["Lasso (L1 Ranking)"]
    ? "Lasso (L1 Ranking)"
    : problem.deployed?.algorithm || "Lasso (L1 Ranking)";
  state.k = problem.bestSchemes
    ? problem.bestSchemes[0].k
    : preferredK || problem.deployed?.k || problem.best.k || 2;
  render();
}

function knownPreset(url) {
  const value = url.toLowerCase();
  if (value.includes("kc_house_data.csv")) return state.presets.find((item) => item.id === "kingcounty");
  if (value.includes("50_startups.csv")) return state.presets.find((item) => item.id === "startups");
  if (value.includes("bostonhousing.csv")) return state.presets.find((item) => item.id === "boston");
  return null;
}

async function loadUrl(url) {
  const preset = knownPreset(url);
  if (preset) {
    setStatus(`這是課程資料集，改顯示 sklearn 的 ${preset.title} 結果。`);
    selectProblem(preset);
    return;
  }
  setStatus("正在下載 CSV…");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`下載失敗 (${response.status})。請改用 raw CSV 網址。`);
  const text = await response.text();
  if (text.trim().startsWith("<")) throw new Error("網址回傳的是網頁，不是 CSV。");
  prepareCustom(text, `已下載 ${url}`);
}

function prepareCustom(text, message) {
  state.customText = text;
  const headers = window.parseCsvHeaders(text);
  const target = $("target-col");
  target.hidden = false;
  $("analyze-custom").hidden = false;
  target.innerHTML = headers.map((header) => `<option>${header}</option>`).join("");
  const guess = headers.find((header) => /profit|medv|price|sales|target/i.test(header)) || headers.at(-1);
  target.value = guess;
  setStatus(`${message}。目標欄位先用 ${guess}，可以改選後再分析。`);
  runCustom();
}

function runCustom() {
  if (!state.customText) return;
  setStatus("分析中…");
  window.setTimeout(() => {
    try {
      const result = window.analyzeCsv(state.customText, $("target-col").value);
      setStatus(`自訂資料集分析完成：${result.n} 筆，${result.encodedFeatures.length} 個編碼後特徵。`);
      selectProblem(result, result.best.k);
    } catch (error) {
      setStatus(error.message, true);
    }
  }, 30);
}

async function init() {
  const response = await fetch("data/presets.json");
  state.presets = await response.json();
  selectProblem(state.presets[0], 2);
  $("problem").addEventListener("change", () => {
    const found = state.presets.find((item) => item.id === $("problem").value);
    if (found) selectProblem(found, found.id === "startups" ? 2 : found.best.k);
  });
  $("k-range").addEventListener("input", () => {
    state.scheme = null;
    state.k = Number($("k-range").value);
    render();
  });
  $("dataset-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await loadUrl($("dataset-url").value.trim());
    } catch (error) {
      setStatus(error.message, true);
    }
  });
  $("analyze-custom").addEventListener("click", runCustom);
  $("dataset-file").addEventListener("change", async () => {
    const file = $("dataset-file").files[0];
    if (!file) return;
    prepareCustom(await file.text(), `已讀取 ${file.name}`);
  });
}

init().catch((error) => setStatus(error.message, true));
