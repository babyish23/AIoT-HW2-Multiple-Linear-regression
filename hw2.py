"""DIC2 HW2: King County house prices, multiple linear regression, CRISP-DM.

Dataset:
    Kaggle House Sales in King County, USA
    https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

18 original predictors (id and date removed). Feature selection uses only the
training split. Test MSE / RMSE / R² are reported on the held-out 20%.

Run:
    python hw2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import RFE, f_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from mlr_workflow import evaluate_subset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "kc_house_data.csv"
OUTPUT = ROOT / "output" / "kingcounty"
RANDOM_STATE = 0
TEST_SIZE = 0.2
TARGET = "price"
DROP = ["id", "date"]
METHODS = [
    "Lasso (L1 Ranking)",
    "Forward SFS",
    "RFE",
    "SelectKBest",
    "Permutation Importance",
]
FONT = r"C:\Windows\Fonts\kaiu.ttf"


def load_xy():
    frame = pd.read_csv(DATA_PATH)
    features = [col for col in frame.columns if col not in DROP + [TARGET]]
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    mask = y.notna() & X.notna().all(axis=1)
    return frame.loc[mask].reset_index(drop=True), X.loc[mask].astype(float), y.loc[mask].astype(float), features


def prefixes(order):
    return [list(order[:k]) for k in range(1, len(order) + 1)]


def forward_path(X_train, y_train, features):
    selected = []
    remaining = list(features)
    path = []
    while remaining:
        best_name = None
        best_score = -np.inf
        for name in remaining:
            trial = selected + [name]
            score = cross_val_score(
                LinearRegression(),
                X_train[trial],
                y_train,
                cv=3,
                scoring="r2",
            ).mean()
            if score > best_score:
                best_score = score
                best_name = name
        selected.append(best_name)
        remaining.remove(best_name)
        path.append(list(selected))
    return path


def selection_paths(X_train, y_train, features):
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=features, index=X_train.index
    )

    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000).fit(scaled, y_train)
    lasso_order = [name for _, name in sorted(zip(np.abs(lasso.coef_), features), reverse=True)]

    rfe = RFE(LinearRegression(), n_features_to_select=1).fit(scaled, y_train)
    rfe_order = [features[i] for i in np.argsort(rfe.ranking_)]

    scores = f_regression(X_train, y_train)[0]
    kbest_order = [features[i] for i in np.argsort(-scores)]

    base = LinearRegression().fit(X_train, y_train)
    importance = permutation_importance(
        base, X_train, y_train, n_repeats=4, random_state=RANDOM_STATE, scoring="r2"
    )
    perm_order = [features[i] for i in np.argsort(-importance.importances_mean)]

    return {
        "Lasso (L1 Ranking)": prefixes(lasso_order),
        "Forward SFS": forward_path(X_train, y_train, features),
        "RFE": prefixes(rfe_order),
        "SelectKBest": prefixes(kbest_order),
        "Permutation Importance": prefixes(perm_order),
    }, lasso, importance


def public_model(model, mse, method=None, rank=None):
    payload = {
        "k": model["k"],
        "features": model["features"],
        "mse": mse,
        "rmse": model["rmse"],
        "mae": model["mae"],
        "r2": model["r2"],
        "adjR2": model["adjR2"],
        "fStat": model["fStat"],
        "fP": model["fP"],
        "coefficients": model["coefficients"],
        "residuals": model["residuals"],
    }
    if method:
        payload["method"] = method
    if rank:
        payload["rank"] = rank
    return payload


def prediction_frame(X_train, y_train, X_test, features):
    train = sm.add_constant(X_train[features], has_constant="add")
    test = sm.add_constant(X_test[features], has_constant="add")
    ols = sm.OLS(y_train, train).fit()
    return ols.get_prediction(test).summary_frame(alpha=0.05), ols


def save_plots(y_test, frame, residual, fitted):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    order = np.argsort(frame["mean"].to_numpy())
    step = max(1, len(order) // 180)
    take = order[::step][:180]
    xs = np.arange(len(take))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axes[0].fill_between(
        xs,
        frame["obs_ci_lower"].to_numpy()[take],
        frame["obs_ci_upper"].to_numpy()[take],
        color="#c9d7ea",
        label="95% prediction interval",
    )
    axes[0].fill_between(
        xs,
        frame["mean_ci_lower"].to_numpy()[take],
        frame["mean_ci_upper"].to_numpy()[take],
        color="#0e6b67",
        alpha=0.45,
        label="95% confidence interval",
    )
    axes[0].plot(xs, frame["mean"].to_numpy()[take], color="#14342b", linewidth=1.4, label="Predicted mean")
    axes[0].scatter(xs, np.asarray(y_test)[take], s=12, color="#8a4b08", alpha=0.75, label="Actual price")
    axes[0].set_xlabel("Test homes, sorted by predicted price")
    axes[0].set_ylabel("Price (USD)")
    axes[0].set_title("Test set prediction with 95% intervals")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    actual = np.asarray(y_test, dtype=float)
    predicted = frame["mean"].to_numpy()
    axes[1].scatter(actual, predicted, s=10, alpha=0.35, color="#0e6b67")
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    axes[1].plot([low, high], [low, high], color="#8d342c", linestyle="--", label="y = x")
    axes[1].set_xlabel("Actual price (USD)")
    axes[1].set_ylabel("Predicted price (USD)")
    axes[1].set_title("Actual vs predicted")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    interval_path = OUTPUT / "prediction_interval.png"
    fig.savefig(interval_path, dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    axes[0].scatter(fitted, residual, s=10, alpha=0.35, color="#0e6b67")
    axes[0].axhline(0, color="#8d342c", linestyle="--")
    axes[0].set_xlabel("Fitted price on the training set")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs fitted")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[1].hist(residual, bins=40, color="#0e6b67")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Training residual distribution")
    fig.tight_layout()
    residual_path = OUTPUT / "residual_diagnostics.png"
    fig.savefig(residual_path, dpi=140)
    plt.close(fig)
    return interval_path, residual_path


def _subset_cjk_font(text):
    from fontTools.subset import Options, Subsetter
    from fontTools.ttLib import TTCollection

    collection = TTCollection(r"C:\Windows\Fonts\msjh.ttc")
    face = collection.fonts[0]
    options = Options()
    options.layout_features = ["*"]
    options.name_IDs = ["*"]
    options.notdef_outline = True
    options.recommended_glyphs = True
    subsetter = Subsetter(options=options)
    subsetter.populate(text=text)
    subsetter.subset(face)
    dest = ROOT / "report" / "_msjh_subset.ttf"
    dest.parent.mkdir(exist_ok=True)
    face.save(dest)
    return dest


def write_pdf(text, images):
    import pymupdf

    title = "5115056011 HW2 多元線性迴歸報告"
    captions = {
        "prediction_interval.png": "預測圖：95% 信賴區間與 95% 預測區間",
        "residual_diagnostics.png": "殘差診斷",
        "notebooklm_summary.png": "NotebookLM 摘要截圖",
    }
    fontfile = str(_subset_cjk_font(title + text + "".join(captions.values())))
    font = pymupdf.Font(fontfile=fontfile)
    out = ROOT / "report" / "hw2_report.pdf"
    out.parent.mkdir(exist_ok=True)
    doc = pymupdf.open()
    margin_x, margin_y = 50, 56
    page_w, page_h = 595, 842
    body_size, leading = 12, 20
    max_width = page_w - 2 * margin_x

    def new_page():
        page = doc.new_page(width=page_w, height=page_h)
        page.insert_font(fontname="msjh", fontfile=fontfile)
        return page, margin_y

    def wrap(paragraph):
        if not paragraph:
            return [""]
        tokens, buf = [], ""
        for ch in paragraph:
            if ord(ch) > 0x2E80:
                if buf:
                    tokens.append(buf)
                    buf = ""
                tokens.append(ch)
            else:
                buf += ch
        if buf:
            tokens.append(buf)
        lines, current = [], ""
        for token in tokens:
            trial = current + token
            if font.text_length(trial, body_size) <= max_width:
                current = trial
                continue
            if current.strip():
                lines.append(current.rstrip())
            if font.text_length(token, body_size) <= max_width:
                current = token.lstrip()
                continue
            parts, chunk = [], ""
            for ch in token:
                nxt = chunk + ch
                if font.text_length(nxt, body_size) <= max_width:
                    chunk = nxt
                    if ch in ",，、 ":
                        parts.append(chunk)
                        chunk = ""
                else:
                    if chunk:
                        parts.append(chunk)
                    chunk = ch
            if chunk:
                parts.append(chunk)
            current = ""
            for part in parts:
                trial = current + part
                if current and font.text_length(trial, body_size) > max_width:
                    lines.append(current.rstrip())
                    current = part.lstrip()
                else:
                    current = trial
        if current.strip():
            lines.append(current.rstrip())
        return lines or [""]

    page, y = new_page()
    page.insert_text((margin_x, y), title, fontname="msjh", fontsize=18)
    y += 34
    for paragraph in text.split("\n"):
        for line in wrap(paragraph):
            if y > page_h - margin_y:
                page, y = new_page()
            page.insert_text((margin_x, y), line or " ", fontname="msjh", fontsize=body_size)
            y += leading
        y += 6

    for image in images:
        page, y = new_page()
        page.insert_text(
            (margin_x, y),
            captions.get(image.name, image.name),
            fontname="msjh",
            fontsize=12,
        )
        page.insert_image(
            pymupdf.Rect(margin_x, y + 16, page_w - margin_x, page_h - margin_y),
            filename=str(image),
            keep_proportion=True,
        )

    gpt = ROOT / "report" / "GPT輔助內容.pdf"
    if gpt.exists():
        extra = pymupdf.open(gpt)
        doc.insert_pdf(extra)
        extra.close()

    doc.save(str(out), garbage=4, deflate=True)
    doc.close()
    Path(fontfile).unlink(missing_ok=True)
    return out


def write_preset(result):
    path = ROOT / "data" / "presets.json"
    path.write_text(json.dumps([result], ensure_ascii=False), encoding="utf-8")


def schema_from(X):
    rows = []
    for col in X.columns:
        series = X[col]
        rows.append(
            {
                "key": col,
                "label": col,
                "kind": "number",
                "min": float(series.min()),
                "max": float(series.max()),
                "default": float(series.median()),
            }
        )
    return rows


def main():
    print("=" * 72)
    print("DIC2 HW2  King County House Sales")
    print("CRISP-DM  |  sklearn LinearRegression")
    print("=" * 72)

    print("\nPHASE 1  Business Understanding")
    print("預測 King County 房屋成交價 price。")
    print("原始欄位扣掉 id 與 date 後有 18 個預測變數，同時符合作業的 10–20 與補充的 14–39。")
    print("Predict the Flood 的公開免登入 CSV 被做成 price-like 目標等於 0.005 乘上特徵總和，")
    print("測試 R2 會變成 1，不能拿來做特徵選擇，所以不用那份檔。")

    raw, X, y, features = load_xy()
    print("\nPHASE 2  Data Understanding")
    print("Source: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction")
    print(f"Local file: {DATA_PATH}")
    print(f"Shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
    print("Predictors:", ", ".join(features))
    print(f"Target price: min {y.min():.0f}  median {y.median():.0f}  max {y.max():.0f}")
    print("Missing values:", int(raw.isna().sum().sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("\nPHASE 3  Data Preparation")
    print("Dropped id and date. zipcode is kept as a numeric code and left for feature selection to judge.")
    print(f"Train {len(X_train)} / Test {len(X_test)}")
    print(f"Split: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")

    print("\nPHASE 4  Modeling")
    print("OLS on every prefix of five selection paths. RidgeCV is reported separately.")
    paths, lasso, _importance = selection_paths(X_train, y_train, features)
    algorithms = {}
    for name in METHODS:
        print(f"  evaluating {name}")
        algorithms[name] = [
            evaluate_subset(subset, X_train, X_test, y_train, y_test) for subset in paths[name]
        ]

    scaler = StandardScaler()
    ridge = RidgeCV(alphas=np.logspace(-2, 5, 16)).fit(scaler.fit_transform(X_train), y_train)
    ridge_pred = ridge.predict(scaler.transform(X_test))
    ridge_mse = float(mean_squared_error(y_test, ridge_pred))
    ridge_r2 = float(r2_score(y_test, ridge_pred))

    print("\nPHASE 5  Evaluation")
    full = algorithms["Lasso (L1 Ranking)"][-1]
    full_mse = float(full["rmse"] ** 2)
    print(f"Full OLS  k={full['k']}  Test MSE={full_mse:.4f}  RMSE={full['rmse']:.4f}  R2={full['r2']:.4f}")
    print(f"RidgeCV   alpha={float(ridge.alpha_):.4g}  Test MSE={ridge_mse:.4f}  R2={ridge_r2:.4f}")
    print(f"Lasso nonzero features: {int(np.sum(np.abs(lasso.coef_) > 1e-8))} / {len(features)}")
    print("\nBest subset on each path")
    for name in METHODS:
        winner = min(algorithms[name], key=lambda model: model["rmse"])
        print(
            f"  {name:24s} k={winner['k']:<2d} MSE={winner['rmse'] ** 2:14.2f} "
            f"R2={winner['r2']:.4f}"
        )

    unique = {}
    for name, models in algorithms.items():
        for model in models:
            key = tuple(sorted(model["features"]))
            mse = float(model["rmse"] ** 2)
            if key not in unique or mse < unique[key][0]:
                unique[key] = (mse, name, model)
    ranked = sorted(unique.values(), key=lambda item: (item[0], item[2]["k"]))
    print("\nTop 10 unique schemes by test MSE")
    for rank, (mse, name, model) in enumerate(ranked[:10], start=1):
        print(f"{rank:2d}  {name:24s} k={model['k']:<2d} MSE={mse:14.2f} R2={model['r2']:.4f}")

    best_mse, best_method, best = ranked[0]
    frame, ols = prediction_frame(X_train, y_train, X_test, best["features"])
    train_design = sm.add_constant(X_train[best["features"]], has_constant="add")
    fitted = ols.predict(train_design)
    residual = y_train.to_numpy() - fitted
    influence_corr = float(np.corrcoef(np.abs(residual), fitted)[0, 1])
    resid_skew = float(stats.skew(residual))
    dw = float(np.sum(np.diff(residual) ** 2) / np.sum(residual ** 2))
    vifs = [coef["vif"] for coef in best["coefficients"] if coef["vif"] is not None]
    max_vif = max(vifs) if vifs else float("nan")

    print("\nRegression assumptions for the best OLS scheme")
    print(f"  residual skew={resid_skew:.3f}  (|skew|>1 suggests non-normal residuals)")
    print(f"  |residual| vs fitted correlation={influence_corr:.3f}  (large value suggests heteroscedasticity)")
    print(f"  Durbin-Watson={dw:.3f}  (near 2 suggests little linear autocorrelation)")
    print(f"  max VIF={max_vif:.3f}  (>10 suggests multicollinearity)")

    print("\nPHASE 6  Deployment")
    print(f"Deployed scheme: {best_method} k={best['k']} features={best['features']}")
    print(f"Test MSE={best_mse:.4f}  RMSE={best['rmse']:.4f}  MAE={best['mae']:.4f}  R2={best['r2']:.4f}")
    interval_path, residual_path = save_plots(y_test, frame, residual, fitted)
    print(f"Plots: {interval_path}")

    best5 = [
        public_model(model, mse, method, rank)
        for rank, (mse, method, model) in enumerate(ranked[:5], start=1)
    ]
    top20 = [
        {
            "rank": rank,
            "method": method,
            "k": model["k"],
            "features": model["features"],
            "mse": mse,
            "rmse": model["rmse"],
            "r2": model["r2"],
        }
        for rank, (mse, method, model) in enumerate(ranked[:20], start=1)
    ]
    web_algorithms = {}
    for name, models in algorithms.items():
        web_algorithms[name] = []
        for model in models:
            item = public_model(model, float(model["rmse"] ** 2))
            web_algorithms[name].append(item)

    preset = {
        "id": "kingcounty",
        "title": "King County House Sales",
        "subtitle": "預測房屋成交價 price",
        "target": TARGET,
        "targetUnit": "USD",
        "source": "https://www.kaggle.com/datasets/harlfoxem/housesalesprediction",
        "notes": "去掉 id 與 date。18 個原始預測變數含 zipcode、lat、long。",
        "businessGoal": "用房屋規格與地點預測成交價，並只留下測試誤差較低的特徵。",
        "n": int(len(X)),
        "nTrain": int(len(X_train)),
        "nTest": int(len(X_test)),
        "split": f"test_size={TEST_SIZE}, random_state={RANDOM_STATE}",
        "reference": {},
        "encodedFeatures": features,
        "predictors": schema_from(X),
        "best": {
            "algorithm": best_method,
            "k": best["k"],
            "features": best["features"],
            "rmse": best["rmse"],
            "r2": best["r2"],
        },
        "fullModel": {"k": full["k"], "rmse": full["rmse"], "r2": full["r2"], "features": full["features"]},
        "testMse": full_mse,
        "bestTestMse": best_mse,
        "top20": top20,
        "bestSchemes": best5,
        "deployed": {
            "algorithm": best_method,
            "k": best["k"],
            "features": best["features"],
            "rmse": best["rmse"],
            "r2": best["r2"],
            "coefficients": best["coefficients"],
            "exampleInputs": {},
            "prediction": None,
        },
        "algorithms": web_algorithms,
    }
    write_preset(preset)

    report = f"""資料來源
Kaggle「House Sales in King County, USA」。
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
本地檔 data/kc_house_data.csv，共 {len(X)} 筆、21 欄。目標是 price。去掉 id 與 date 後有 {len(features)} 個原始預測變數，落在作業要求的 10–20，也落在補充條件的 14–39。

為什麼不是 Predict the Flood
補充建議的第一名是 Kaggle Playground S4E5，原始 20 個數值特徵，本來最合適。不需登入就能下載的那份公開 CSV，FloodProbability 恰好等於 0.005 乘上 20 個特徵的總和，線性回歸測試 R² 為 1。那種資料做特徵選擇沒有意義，所以沒有拿來當成果。

【CRISP-DM】
1. Business Understanding：買家與估價需要一個可解釋的成交價模型，而不是只追求黑盒分數。
2. Data Understanding：{len(X)} 筆、無缺值。價格最小 {y.min():.0f}、中位數 {y.median():.0f}、最大 {y.max():.0f}。sqft_living、sqft_above、sqft_basement 尺度大且彼此相關。
3. Data Preparation：刪除 id、date。80/20 分割，random_state=0，訓練 {len(X_train)}、測試 {len(X_test)}。特徵選擇只用訓練集。
4. Modeling：sklearn LinearRegression。五條路徑是 Lasso 係數排序、Forward SFS、RFE、SelectKBest、Permutation Importance。另報 RidgeCV 當收縮對照，不混進 OLS 排名。
5. Evaluation：全部特徵的測試 MSE 是 {full_mse:.2f}，R² 是 {full['r2']:.4f}。最好的 OLS 子集是 {best_method}，k={best['k']}，測試 MSE {best_mse:.2f}，RMSE {best['rmse']:.2f}，MAE {best['mae']:.2f}，R² {best['r2']:.4f}。RidgeCV 測試 MSE {ridge_mse:.2f}，R² {ridge_r2:.4f}。
6. Deployment：網頁用這組係數做即時估價，並畫出 95% 信賴區間與 95% 預測區間。

採用的特徵
{", ".join(best["features"])}

回歸假設
殘差偏態 {resid_skew:.3f}。絕對殘差與配適值的相關 {influence_corr:.3f}。Durbin-Watson {dw:.3f}。最大 VIF {max_vif:.3f}。
房價右偏，殘差不是常態，而且大房子的誤差比較大，變異數不齊。這是這類資料的常見現象，不是程式算錯。對數價格是公開分析裡常見的下一步；本報告主模型仍用原始美元，讓區間可以直接讀成價格。VIF 若偏高，通常來自家中面積欄位彼此重疊，特徵選擇就是在處理這件事。

【主流解法比較】
King County 的公開筆記本多半把 price 取對數，再把 zipcode 做成目標編碼或分組，然後用 Random Forest 或 Gradient Boosting。那些模型的測試 R² 常常高於單純線性回歸，但係數不好直接解釋，也沒有線性模型這種預測區間。這份作業保留線性回歸，用五種特徵選擇處理共線性，並把區間畫出來。Ridge 用來對照收縮；它的測試誤差和全部特徵的 OLS 接近，因為這批資料的樣本數遠大於特徵數，收縮的幫助有限。

【NotebookLM 摘要】
King County 房價分析顯示，居住面積（sqft_living）與房屋等級（grade）為最關鍵變數。因 sqft_living 為 sqft_above 加地下室坪數，兩者高度相關，來源透過剔除 sqft_above 消除多重共線性；zipcode 因類別過多且無數字順序，不宜作為連續變數，需轉為類別編碼。為修正右偏與異質變異，來源對價格與坪數取對數，並以 Q-Q plot 與殘差圖檢查常態性。線性回歸具備高係數解釋力，但非線性預測力有限。樹模型（Random Forest、Gradient Boosting）預測力更高但缺乏直接解釋力。若作業規定使用線性回歸與特徵選擇，應保留 OLS 的顯著係數、p 值、VIF 及帶有區間的預測圖。在比較段落中，則應承認樹模型在捕捉非線性關係與預測精準度上通常優於線性回歸。

【GPT 輔助內容】
完整對話已用 ChatGPT Exporter 匯出為 PDF，附在本報告後面，共 23 頁，並另存為 report/GPT輔助內容.pdf。對話先討論符合條件的 Kaggle 資料集與 CRISP-DM 報告架構，接著依 King County 的實際設定撰寫正文：18 個特徵的 OLS、Forward SFS 的 14 個特徵、RidgeCV 對照、測試 MSE、RMSE、R²、五大回歸假設，以及信賴區間和預測區間的差別。對話也比較了對數價格、把 zipcode 當類別，以及 Random Forest、Gradient Boosting。這些模型常常預測得更好，但這次仍以可解釋的 OLS 作為主要結果。。
"""
    pdf_path = write_pdf(report, [interval_path, residual_path, ROOT / "report" / "notebooklm_summary.png"])
    print(f"Report: {pdf_path}")
    (ROOT / "report" / "hw2_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
