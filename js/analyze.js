/* Browser multiple linear regression for a user-supplied CSV.
   Preset problems stay on the sklearn results. This path is only used when
   the pasted dataset is not one of the two course files. */
(function () {
  function parseCsv(text) {
    const rows = [];
    let row = [];
    let cell = "";
    let quoted = false;
    const src = text.replace(/^\uFEFF/, "");
    for (let i = 0; i < src.length; i++) {
      const ch = src[i];
      if (quoted) {
        if (ch === '"') {
          if (src[i + 1] === '"') {
            cell += '"';
            i++;
          } else quoted = false;
        } else cell += ch;
      } else if (ch === '"') quoted = true;
      else if (ch === ",") {
        row.push(cell.trim());
        cell = "";
      } else if (ch === "\n") {
        row.push(cell.trim());
        if (row.some((value) => value !== "")) rows.push(row);
        row = [];
        cell = "";
      } else if (ch !== "\r") cell += ch;
    }
    row.push(cell.trim());
    if (row.some((value) => value !== "")) rows.push(row);
    if (rows.length < 3) throw new Error("CSV 至少需要標題列和兩筆資料");
    const headers = rows[0];
    const records = rows.slice(1).map((values) => {
      const record = {};
      headers.forEach((header, index) => {
        record[header] = values[index] ?? "";
      });
      return record;
    });
    return { headers, records };
  }

  function isNumeric(values) {
    const parsed = values.map((value) => Number(value)).filter((value) => value !== "" && !Number.isNaN(Number(value)));
    return parsed.length / values.length >= 0.9;
  }

  function encode(records, target) {
    const predictors = Object.keys(records[0]).filter((name) => name !== target);
    const columns = [];
    const schema = [];
    const matrix = records.map(() => []);
    const y = [];

    predictors.forEach((name) => {
      const values = records.map((row) => row[name]);
      if (isNumeric(values)) {
        columns.push(name);
        schema.push({
          key: name,
          label: name,
          kind: "number",
          min: Math.min(...values.map(Number).filter((value) => !Number.isNaN(value))),
          max: Math.max(...values.map(Number).filter((value) => !Number.isNaN(value))),
          default: median(values.map(Number).filter((value) => !Number.isNaN(value))),
        });
        records.forEach((row, index) => matrix[index].push(Number(row[name])));
      } else {
        const levels = [...new Set(values.filter(Boolean))].sort();
        const dummies = {};
        levels.slice(1).forEach((level) => {
          const dummy = `${name}_${level}`;
          columns.push(dummy);
          dummies[level] = dummy;
          records.forEach((row, index) => matrix[index].push(row[name] === level ? 1 : 0));
        });
        schema.push({
          key: name,
          label: name,
          kind: "category",
          options: levels,
          reference: levels[0],
          dummies,
        });
      }
    });

    records.forEach((row, index) => y.push(Number(row[target])));
    const keep = y.map((value, index) => Number.isFinite(value) && matrix[index].every(Number.isFinite));
    return {
      columns,
      schema,
      X: matrix.filter((_, index) => keep[index]),
      y: y.filter((_, index) => keep[index]),
    };
  }

  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function split(X, y, seed = 0) {
    const index = X.map((_, i) => i);
    let state = seed + 1;
    const random = () => {
      state = (1664525 * state + 1013904223) % 4294967296;
      return state / 4294967296;
    };
    for (let i = index.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [index[i], index[j]] = [index[j], index[i]];
    }
    const nTest = Math.max(1, Math.round(X.length * 0.2));
    const test = index.slice(0, nTest);
    const train = index.slice(nTest);
    const take = (rows, idx) => idx.map((i) => rows[i]);
    return { XTrain: take(X, train), yTrain: take(y, train), XTest: take(X, test), yTest: take(y, test), testIndex: test };
  }

  function solve(a, b) {
    const n = b.length;
    const m = a.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
      }
      [m[col], m[pivot]] = [m[pivot], m[col]];
      const div = m[col][col];
      if (Math.abs(div) < 1e-10) return null;
      for (let row = col + 1; row < n; row++) {
        const factor = m[row][col] / div;
        for (let c = col; c <= n; c++) m[row][c] -= factor * m[col][c];
      }
    }
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = m[i][n];
      for (let c = i + 1; c < n; c++) sum -= m[i][c] * x[c];
      x[i] = sum / m[i][i];
    }
    return x;
  }

  function takeColumns(rows, indexes) {
    return rows.map((row) => indexes.map((index) => row[index]));
  }

  function withIntercept(rows) {
    return rows.map((row) => [1, ...row]);
  }

  function predict(beta, rows) {
    const design = withIntercept(rows);
    return design.map((row) => row.reduce((sum, value, index) => sum + value * beta[index], 0));
  }

  function fit(XTrain, yTrain, indexes) {
    const beta = solve(xtx(withIntercept(takeColumns(XTrain, indexes))), xty(withIntercept(takeColumns(XTrain, indexes)), yTrain));
    return beta;
  }

  function xtx(rows) {
    const p = rows[0].length;
    const out = Array.from({ length: p }, () => Array(p).fill(0));
    rows.forEach((row) => {
      for (let i = 0; i < p; i++) {
        for (let j = i; j < p; j++) out[i][j] += row[i] * row[j];
      }
    });
    for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) out[i][j] = out[j][i];
    return out;
  }

  function xty(rows, y) {
    const p = rows[0].length;
    const out = Array(p).fill(0);
    rows.forEach((row, index) => {
      for (let i = 0; i < p; i++) out[i] += row[i] * y[index];
    });
    return out;
  }

  function metrics(yTrue, yPred, p) {
    const n = yTrue.length;
    let sse = 0;
    let sae = 0;
    let mean = 0;
    yTrue.forEach((value) => (mean += value));
    mean /= n;
    let sst = 0;
    yTrue.forEach((value, index) => {
      const err = value - yPred[index];
      sse += err * err;
      sae += Math.abs(err);
      sst += (value - mean) ** 2;
    });
    const r2 = sst === 0 ? 0 : 1 - sse / sst;
    return {
      rmse: Math.sqrt(sse / n),
      mae: sae / n,
      r2,
      adjR2: n > p + 1 ? 1 - (1 - r2) * (n - 1) / (n - p - 1) : null,
    };
  }

  function inference(XTrain, yTrain, indexes, names) {
    const subset = takeColumns(XTrain, indexes);
    const design = withIntercept(subset);
    const beta = solve(xtx(design), xty(design, yTrain));
    if (!beta) return null;
    const fitted = predict(beta, subset);
    const n = yTrain.length;
    const k = beta.length;
    let sse = 0;
    let mean = yTrain.reduce((sum, value) => sum + value, 0) / n;
    let sst = 0;
    yTrain.forEach((value, index) => {
      sse += (value - fitted[index]) ** 2;
      sst += (value - mean) ** 2;
    });
    const df = n - k;
    const sigma2 = sse / df;
    const cov = invert(xtx(design));
    if (!cov) return null;
    const coefficients = beta.map((coef, index) => {
      const se = Math.sqrt(Math.max(cov[index][index], 0) * sigma2);
      const t = se === 0 ? 0 : coef / se;
      const p = df > 0 ? 2 * (1 - studentTCdf(Math.abs(t), df)) : 1;
      const crit = studentTPpf(0.975, df);
      return {
        name: index === 0 ? "Intercept" : names[index - 1],
        coef,
        se,
        t,
        p,
        ciLow: coef - crit * se,
        ciHigh: coef + crit * se,
        vif: null,
      };
    });
    names.forEach((name, featureIndex) => {
      const others = indexes.filter((_, i) => i !== featureIndex);
      if (!others.length) {
        coefficients[featureIndex + 1].vif = 1;
        return;
      }
      const aux = fit(XTrain, subset.map((row) => row[featureIndex]), others);
      if (!aux) {
        coefficients[featureIndex + 1].vif = null;
        return;
      }
      const auxPred = predict(aux, takeColumns(XTrain, others));
      const target = subset.map((row) => row[featureIndex]);
      const r2 = metrics(target, auxPred, others.length).r2;
      coefficients[featureIndex + 1].vif = r2 >= 0.999999 ? Infinity : 1 / (1 - r2);
    });
    const fStat = ((sst - sse) / (k - 1)) / (sse / df);
    return { beta, coefficients, fStat, fP: fSf(fStat, k - 1, df) };
  }

  function invert(matrix) {
    const n = matrix.length;
    const a = matrix.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let row = col + 1; row < n; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
      [a[col], a[pivot]] = [a[pivot], a[col]];
      const div = a[col][col];
      if (Math.abs(div) < 1e-10) return null;
      for (let c = col; c < a[col].length; c++) a[col][c] /= div;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = a[row][col];
        for (let c = col; c < a[row].length; c++) a[row][c] -= factor * a[col][c];
      }
    }
    return a.map((row) => row.slice(n));
  }

  function logGamma(z) {
    const p = [
      676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406,
      12.507343278686905, -0.13857109526572012, 9.9843696540787e-6, 1.5056327351493116e-7,
    ];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
    const x = z - 1;
    let a = 0.9999999999998099;
    for (let i = 0; i < p.length; i++) a += p[i] / (x + i + 1);
    const t = x + p.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  function betacf(a, b, x) {
    const max = 200;
    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;
    let c = 1;
    let d = 1 - (qab * x) / qap;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= max; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < 1e-30) d = 1e-30;
      c = 1 + aa / c;
      if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d;
      h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < 1e-30) d = 1e-30;
      c = 1 + aa / c;
      if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-7) break;
    }
    return h;
  }

  function regularizedBeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return (bt * betacf(a, b, x)) / a;
    return 1 - (bt * betacf(b, a, 1 - x)) / b;
  }

  function studentTCdf(t, df) {
    const x = df / (df + t * t);
    const ib = regularizedBeta(x, df / 2, 0.5);
    return t >= 0 ? 1 - 0.5 * ib : 0.5 * ib;
  }

  function studentTPpf(prob, df) {
    let lo = 0;
    let hi = 1;
    while (studentTCdf(hi, df) < prob) hi *= 2;
    for (let i = 0; i < 60; i++) {
      const mid = (lo + hi) / 2;
      if (studentTCdf(mid, df) < prob) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  }

  function fSf(f, d1, d2) {
    if (!(f > 0)) return 1;
    const x = (d1 * f) / (d1 * f + d2);
    return 1 - regularizedBeta(x, d1 / 2, d2 / 2);
  }

  function standardize(X) {
    const p = X[0].length;
    const mean = Array(p).fill(0);
    const sd = Array(p).fill(0);
    X.forEach((row) => row.forEach((value, j) => (mean[j] += value)));
    mean.forEach((_, j) => (mean[j] /= X.length));
    X.forEach((row) => row.forEach((value, j) => (sd[j] += (value - mean[j]) ** 2)));
    sd.forEach((_, j) => (sd[j] = Math.sqrt(sd[j] / X.length) || 1));
    return X.map((row) => row.map((value, j) => (value - mean[j]) / sd[j]));
  }

  function soft(value, amount) {
    if (value > amount) return value - amount;
    if (value < -amount) return value + amount;
    return 0;
  }

  function coordinateDescent(X, y, alpha, l1Ratio) {
    const n = X.length;
    const p = X[0].length;
    const beta = Array(p).fill(0);
    const yMean = y.reduce((sum, value) => sum + value, 0) / n;
    const yc = y.map((value) => value - yMean);
    for (let iter = 0; iter < 80; iter++) {
      for (let j = 0; j < p; j++) {
        let rho = 0;
        for (let i = 0; i < n; i++) {
          let pred = 0;
          for (let k = 0; k < p; k++) if (k !== j) pred += X[i][k] * beta[k];
          rho += X[i][j] * (yc[i] - pred);
        }
        rho /= n;
        const l1 = alpha * l1Ratio;
        const l2 = alpha * (1 - l1Ratio);
        beta[j] = soft(rho, l1) / (1 + l2);
      }
    }
    return beta;
  }

  function rankByPenalty(XTrain, yTrain, l1Ratio) {
    const Xs = standardize(XTrain);
    let best = null;
    for (let step = 0; step < 12; step++) {
      const alpha = 10 ** (-4 + step * 0.35);
      const folds = cvScore(Xs, yTrain, (xFit, yFit) => coordinateDescent(xFit, yFit, alpha, l1Ratio));
      if (!best || folds < best.score) best = { score: folds, beta: coordinateDescent(Xs, yTrain, alpha, l1Ratio) };
    }
    return best.beta
      .map((coef, index) => ({ index, coef: Math.abs(coef) }))
      .sort((a, b) => b.coef - a.coef)
      .map((item) => item.index);
  }

  function cvRmse(X, y, indexes) {
    const folds = 5;
    let sse = 0;
    let count = 0;
    for (let fold = 0; fold < folds; fold++) {
      const test = [];
      const train = [];
      for (let i = 0; i < X.length; i++) (i % folds === fold ? test : train).push(i);
      if (train.length <= indexes.length + 1 || !test.length) continue;
      const beta = fit(train.map((i) => X[i]), train.map((i) => y[i]), indexes);
      if (!beta) return Infinity;
      const pred = predict(beta, takeColumns(test.map((i) => X[i]), indexes));
      test.forEach((rowIndex, predIndex) => {
        sse += (y[rowIndex] - pred[predIndex]) ** 2;
        count++;
      });
    }
    return count ? sse / count : Infinity;
  }

  function cvScore(X, y, fitter) {
    const folds = 5;
    const n = X.length;
    let total = 0;
    let count = 0;
    for (let fold = 0; fold < folds; fold++) {
      const test = [];
      const train = [];
      for (let i = 0; i < n; i++) (i % folds === fold ? test : train).push(i);
      if (!test.length || train.length <= X[0].length) continue;
      const beta = fitter(train.map((i) => X[i]), train.map((i) => y[i]));
      if (!beta) continue;
      test.forEach((i) => {
        let pred = beta.length === X[0].length + 1 ? beta[0] : 0;
        const offset = beta.length === X[0].length + 1 ? 1 : 0;
        if (offset === 0) {
          const mean = train.reduce((sum, idx) => sum + y[idx], 0) / train.length;
          pred = mean;
          for (let j = 0; j < X[0].length; j++) pred += X[i][j] * beta[j];
        } else {
          for (let j = 0; j < X[0].length; j++) pred += X[i][j] * beta[j + 1];
        }
        total += (y[i] - pred) ** 2;
        count++;
      });
    }
    return count ? total / count : Infinity;
  }

  function evaluate(indexes, names, pack) {
    const info = inference(pack.XTrain, pack.yTrain, indexes, names);
    if (!info) return null;
    const pred = predict(info.beta, takeColumns(pack.XTest, indexes));
    const score = metrics(pack.yTest, pred, indexes.length);
    const residuals = pack.yTest
      .map((actual, index) => {
        const residual = actual - pred[index];
        return {
          id: pack.testIndex[index],
          actual,
          predicted: pred[index],
          residual,
          absError: Math.abs(residual),
          pctError: actual === 0 ? null : (Math.abs(residual) / Math.abs(actual)) * 100,
        };
      })
      .sort((a, b) => b.absError - a.absError)
      .slice(0, 10)
      .map((row, rank) => ({ rank: rank + 1, ...row }));
    return {
      k: indexes.length,
      features: names,
      ...score,
      fStat: info.fStat,
      fP: info.fP,
      coefficients: info.coefficients,
      residuals,
    };
  }

  function combinations(items, k) {
    const out = [];
    const walk = (start, path) => {
      if (path.length === k) {
        out.push([...path]);
        return;
      }
      for (let i = start; i < items.length; i++) {
        path.push(items[i]);
        walk(i + 1, path);
        path.pop();
      }
    };
    walk(0, []);
    return out;
  }

  function analyzeCsv(text, target) {
    const table = parseCsv(text);
    if (!table.headers.includes(target)) throw new Error(`找不到目標欄位 ${target}`);
    const encoded = encode(table.records, target);
    if (encoded.y.length < 8) throw new Error("有效資料少於 8 筆，無法切訓練集與測試集");
    let columns = encoded.columns;
    let X = encoded.X;
    if (columns.length > 12) {
      const corr = columns.map((name, index) => {
        const xs = X.map((row) => row[index]);
        const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length;
        const meanY = encoded.y.reduce((sum, value) => sum + value, 0) / encoded.y.length;
        let num = 0;
        let dx = 0;
        let dy = 0;
        xs.forEach((value, i) => {
          num += (value - meanX) * (encoded.y[i] - meanY);
          dx += (value - meanX) ** 2;
          dy += (encoded.y[i] - meanY) ** 2;
        });
        return { index, score: Math.abs(num / Math.sqrt(dx * dy || 1)) };
      });
      const keep = corr.sort((a, b) => b.score - a.score).slice(0, 12).map((item) => item.index).sort((a, b) => a - b);
      columns = keep.map((index) => columns[index]);
      X = X.map((row) => keep.map((index) => row[index]));
    }
    const pack = split(X, encoded.y);
    const p = columns.length;
    const all = columns.map((_, index) => index);
    const lassoOrder = rankByPenalty(pack.XTrain, pack.yTrain, 1);
    const elasticOrder = rankByPenalty(pack.XTrain, pack.yTrain, 0.5);

    const forward = [];
    let selected = [];
    for (let k = 1; k <= p; k++) {
      let best = null;
      all.filter((index) => !selected.includes(index)).forEach((index) => {
        const trial = [...selected, index];
        const score = cvRmse(pack.XTrain, pack.yTrain, trial);
        if (!best || score < best.score) best = { score, index };
      });
      selected = [...selected, best.index].sort((a, b) => a - b);
      forward.push([...selected]);
    }

    const rfe = [];
    let remaining = [...all];
    const removal = [];
    while (remaining.length > 1) {
      const beta = fit(standardize(takeColumns(pack.XTrain, remaining)), pack.yTrain, remaining.map((_, i) => i));
      if (!beta) break;
      let dropAt = 0;
      let bestAbs = Infinity;
      beta.slice(1).forEach((coef, index) => {
        if (Math.abs(coef) < bestAbs) {
          bestAbs = Math.abs(coef);
          dropAt = index;
        }
      });
      removal.push(remaining[dropAt]);
      remaining = remaining.filter((_, index) => index !== dropAt);
    }
    removal.push(remaining[0]);
    const kept = removal.slice().reverse();
    for (let k = 1; k <= p; k++) rfe.push(kept.slice(0, k).sort((a, b) => a - b));

    const exhaustive = [];
    for (let k = 1; k <= p; k++) {
      const combos = combinations(all, k);
      if (combos.length > 3000) {
        exhaustive.push(forward[k - 1]);
        continue;
      }
      let best = null;
      combos.forEach((combo) => {
        const beta = fit(pack.XTrain, pack.yTrain, combo);
        if (!beta) return;
        const pred = predict(beta, takeColumns(pack.XTest, combo));
        const rmse = metrics(pack.yTest, pred, combo.length).rmse;
        if (!best || rmse < best.rmse) best = { rmse, combo };
      });
      exhaustive.push(best.combo);
    }

    const paths = {
      "Lasso (L1 Ranking)": lassoOrder.map((_, k) => lassoOrder.slice(0, k + 1)),
      "Forward SFS": forward,
      RFE: rfe,
      ElasticNet: elasticOrder.map((_, k) => elasticOrder.slice(0, k + 1)),
      "Exhaustive Search": exhaustive,
    };
    const algorithms = {};
    let bestModel = null;
    Object.entries(paths).forEach(([name, subsets]) => {
      algorithms[name] = subsets.map((indexes) =>
        evaluate(indexes, indexes.map((index) => columns[index]), pack)
      ).filter(Boolean);
      algorithms[name].forEach((model) => {
        if (!bestModel || model.rmse < bestModel.rmse) bestModel = { algorithm: name, ...model };
      });
    });
    const full = algorithms["Lasso (L1 Ranking)"].at(-1);
    return {
      id: "custom",
      title: "自訂資料集",
      subtitle: `目標欄位 ${target}`,
      target,
      targetUnit: "",
      source: "使用者輸入的 CSV",
      notes: "這份結果在瀏覽器即時計算，分割方式與 sklearn 的 random_state=0 不同。課程兩題請用下拉選單，數字才是 sklearn 的結果。",
      businessGoal: `用上傳的 CSV 預測 ${target}，並比較不同特徵數的測試誤差。`,
      n: encoded.y.length,
      nTrain: pack.yTrain.length,
      nTest: pack.yTest.length,
      split: "test_size=0.2, browser seeded shuffle",
      reference: {},
      encodedFeatures: columns,
      predictors: encoded.schema,
      best: { algorithm: bestModel.algorithm, k: bestModel.k, features: bestModel.features, rmse: bestModel.rmse, r2: bestModel.r2 },
      fullModel: { k: full.k, rmse: full.rmse, r2: full.r2, features: full.features },
      deployed: { algorithm: bestModel.algorithm, k: bestModel.k, features: bestModel.features, rmse: bestModel.rmse, r2: bestModel.r2, prediction: null, exampleInputs: {} },
      algorithms,
    };
  }

  window.analyzeCsv = analyzeCsv;
  window.parseCsvHeaders = function (text) {
    return parseCsv(text).headers;
  };
})();
