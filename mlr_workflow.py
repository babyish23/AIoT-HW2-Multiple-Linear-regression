"""Shared CRISP-DM workflow for multiple linear regression.

50 Startups and Boston Housing both call run_crispdm(). The modeling steps
use scikit-learn only: LinearRegression, LassoCV, ElasticNetCV, RFE, and
SequentialFeatureSelector.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0
TEST_SIZE = 0.20
ALGORITHMS = [
    "Lasso (L1 Ranking)",
    "Forward SFS",
    "RFE",
    "ElasticNet",
    "Exhaustive Search",
]


def load_frame(source: str) -> pd.DataFrame:
    return pd.read_csv(source)


def encode(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"找不到目標欄位: {target}. 可用欄位: {list(df.columns)}")

    y_raw = df[target]
    y = pd.to_numeric(y_raw, errors="coerce")
    predictors = df.drop(columns=[target])
    encoded = pd.DataFrame(index=df.index)
    schema = []
    reference = {}

    for col in predictors.columns:
        numeric = pd.to_numeric(predictors[col], errors="coerce")
        if numeric.notna().mean() >= 0.9:
            encoded[col] = numeric
            schema.append(
                {
                    "key": col,
                    "label": col,
                    "kind": "number",
                    "min": float(numeric.min(skipna=True)),
                    "max": float(numeric.max(skipna=True)),
                    "default": float(numeric.median(skipna=True)),
                }
            )
            continue

        levels = sorted(predictors[col].dropna().astype(str).unique().tolist())
        ref = levels[0]
        reference[col] = ref
        dummies = {}
        for level in levels[1:]:
            name = f"{col}_{level}"
            encoded[name] = (predictors[col].astype(str) == level).astype(float)
            dummies[level] = name
        schema.append(
            {
                "key": col,
                "label": col,
                "kind": "category",
                "options": levels,
                "reference": ref,
                "dummies": dummies,
            }
        )

    mask = y.notna() & encoded.notna().all(axis=1)
    return encoded.loc[mask].astype(float), y.loc[mask].astype(float), schema, reference


def adjusted_r2(r2: float, n: int, p: int):
    if n <= p + 1:
        return None
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def test_metrics(y_true, y_pred, p: int):
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
        "adjR2": adjusted_r2(r2, len(y_true), p),
    }


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame.to_numpy(dtype=float)])


def ols_inference(X_train: pd.DataFrame, y_train: pd.Series, features: list[str]):
    """Coefficient tests for an OLS fit. The point estimates come from the
    same normal equation sklearn LinearRegression uses."""
    subset = X_train[features]
    model = LinearRegression().fit(subset, y_train)
    names = ["Intercept", *features]
    beta = np.r_[model.intercept_, model.coef_]
    design = design_matrix(subset)
    n, k = design.shape
    fitted = design @ beta
    resid = y_train.to_numpy(dtype=float) - fitted
    df_resid = n - k
    sigma2 = float(resid @ resid) / df_resid
    xtx_inv = np.linalg.inv(design.T @ design)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))
    t_values = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), df_resid)
    t_crit = float(stats.t.ppf(0.975, df_resid))
    ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
    ss_reg = ss_tot - float(resid @ resid)
    f_stat = (ss_reg / (k - 1)) / (float(resid @ resid) / df_resid)
    f_p = float(stats.f.sf(f_stat, k - 1, df_resid))

    vif_values = [None]
    for idx in range(len(features)):
        others = [name for name in features if name != features[idx]]
        if not others:
            vif_values.append(1.0)
            continue
        aux = LinearRegression().fit(X_train[others], X_train[features[idx]])
        r2_aux = float(aux.score(X_train[others], X_train[features[idx]]))
        vif_values.append(None if r2_aux >= 1 - 1e-12 else 1 / (1 - r2_aux))

    coefficients = []
    for name, coef, error, t_stat, p_value, vif in zip(
        names, beta, se, t_values, p_values, vif_values
    ):
        coefficients.append(
            {
                "name": name,
                "coef": float(coef),
                "se": float(error),
                "t": float(t_stat),
                "p": float(p_value),
                "ciLow": float(coef - t_crit * error),
                "ciHigh": float(coef + t_crit * error),
                "vif": None if vif is None else float(vif),
            }
        )
    return coefficients, float(f_stat), f_p


def top_residuals(index, y_true, y_pred, limit=10):
    rows = []
    for idx, actual, pred in zip(index, np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)):
        resid = float(actual - pred)
        rows.append(
            {
                "id": int(idx) if str(idx).isdigit() else str(idx),
                "actual": float(actual),
                "predicted": float(pred),
                "residual": resid,
                "absError": abs(resid),
                "pctError": (abs(resid) / abs(float(actual)) * 100) if actual != 0 else None,
            }
        )
    rows.sort(key=lambda row: row["absError"], reverse=True)
    for rank, row in enumerate(rows[:limit], start=1):
        row["rank"] = rank
    return rows[:limit]


def evaluate_subset(features, X_train, X_test, y_train, y_test):
    features = list(features)
    model = LinearRegression().fit(X_train[features], y_train)
    pred = model.predict(X_test[features])
    metrics = test_metrics(y_test, pred, len(features))
    coefficients, f_stat, f_p = ols_inference(X_train, y_train, features)
    metrics.update(
        {
            "k": len(features),
            "features": features,
            "fStat": f_stat,
            "fP": f_p,
            "coefficients": coefficients,
            "residuals": top_residuals(y_test.index, y_test, pred),
            "testIndex": [int(i) if str(i).isdigit() else str(i) for i in y_test.index],
            "testActual": [float(v) for v in y_test.to_numpy()],
            "testPredicted": [float(v) for v in pred],
        }
    )
    return metrics


def selection_paths(X_train, y_train, X_test, y_test, features: list[str]):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=features, index=X_train.index
    )
    p = len(features)
    paths = {}

    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=100000).fit(X_scaled, y_train)
    lasso_rank = [name for _, name in sorted(zip(np.abs(lasso.coef_), features), reverse=True)]
    paths["Lasso (L1 Ranking)"] = [lasso_rank[:k] for k in range(1, p + 1)]

    sfs_subsets = []
    for k in range(1, p + 1):
        if k == p:
            sfs_subsets.append(list(features))
        else:
            sfs = SequentialFeatureSelector(
                LinearRegression(), n_features_to_select=k, direction="forward"
            )
            sfs.fit(X_train, y_train)
            sfs_subsets.append(list(X_train.columns[sfs.get_support()]))
    paths["Forward SFS"] = sfs_subsets

    rfe_subsets = []
    for k in range(1, p + 1):
        if k == p:
            rfe_subsets.append(list(features))
        else:
            rfe = RFE(LinearRegression(), n_features_to_select=k)
            rfe.fit(X_scaled, y_train)
            rfe_subsets.append(list(X_train.columns[rfe.get_support()]))
    paths["RFE"] = rfe_subsets

    elastic = ElasticNetCV(cv=5, random_state=RANDOM_STATE, max_iter=100000).fit(
        X_scaled, y_train
    )
    elastic_rank = [name for _, name in sorted(zip(np.abs(elastic.coef_), features), reverse=True)]
    paths["ElasticNet"] = [elastic_rank[:k] for k in range(1, p + 1)]

    exhaustive = []
    for k in range(1, p + 1):
        best_rmse = np.inf
        best_subset = None
        for combo in itertools.combinations(features, k):
            chosen = list(combo)
            pred = LinearRegression().fit(X_train[chosen], y_train).predict(X_test[chosen])
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_subset = chosen
        exhaustive.append(best_subset)
    paths["Exhaustive Search"] = exhaustive
    return paths


def save_plots(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Lasso (L1 Ranking)": "#1f77b4",
        "Forward SFS": "#ff7f0e",
        "RFE": "#2ca02c",
        "ElasticNet": "#d62728",
        "Exhaustive Search": "#9467bd",
    }
    k_values = list(range(1, len(result["encodedFeatures"]) + 1))

    primary = result["algorithms"]["Lasso (L1 Ranking)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k_values, [m["rmse"] for m in primary], marker="o", color="#1f77b4")
    axes[0].set_title("RMSE by Number of Features")
    axes[0].set_xlabel("Number of Features")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(k_values)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[1].plot(k_values, [m["r2"] for m in primary], marker="o", color="#1f77b4")
    axes[1].set_title("R-squared by Number of Features")
    axes[1].set_xlabel("Number of Features")
    axes[1].set_ylabel("R-squared")
    axes[1].set_xticks(k_values)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "lasso_progression.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name in ALGORITHMS:
        models = result["algorithms"][name]
        axes[0].plot(k_values, [m["rmse"] for m in models], marker="o", label=name, color=colors[name])
        axes[1].plot(k_values, [m["r2"] for m in models], marker="o", label=name, color=colors[name])
    axes[0].set_title("RMSE by Number of Features")
    axes[0].set_xlabel("Number of Features")
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[0].set_xticks(k_values)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8)
    axes[1].set_title("R-squared by Number of Features")
    axes[1].set_xlabel("Number of Features")
    axes[1].set_ylabel("R-squared (higher is better)")
    axes[1].set_xticks(k_values)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "all_methods_comparison.png", dpi=160)
    plt.close(fig)

    deployed = result["deployed"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    actual = np.array(deployed["testActual"])
    predicted = np.array(deployed["testPredicted"])
    ax.scatter(actual, predicted, s=36, color="#1f77b4")
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    ax.plot([low, high], [low, high], "r--", label="y = x")
    ax.set_xlabel(f"Actual {result['target']}")
    ax.set_ylabel(f"Predicted {result['target']}")
    ax.set_title(f"Test set: {deployed['algorithm']} (k={deployed['k']})")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def save_tables(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ALGORITHMS:
        for model in result["algorithms"][name]:
            rows.append(
                {
                    "Method": name,
                    "Number of Features": model["k"],
                    "Selected Features": model["features"],
                    "RMSE": model["rmse"],
                    "MAE": model["mae"],
                    "R-squared": model["r2"],
                    "Adjusted R-squared": model["adjR2"],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "feature_selection_comparison.csv", index=False)
    for name in ALGORITHMS:
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        table = pd.DataFrame(
            [
                {
                    "Number of Features": model["k"],
                    "Selected Features": model["features"],
                    "RMSE": model["rmse"],
                    "R-squared": model["r2"],
                }
                for model in result["algorithms"][name]
            ]
        )
        table.to_csv(output_dir / f"{slug}_progression.csv", index=False)


def public_model(model: dict) -> dict:
    """Drop per-row test vectors before writing the web payload."""
    keep = (
        "k",
        "features",
        "rmse",
        "mae",
        "r2",
        "adjR2",
        "fStat",
        "fP",
        "coefficients",
        "residuals",
    )
    return {key: model[key] for key in keep}


def run_crispdm(
    *,
    dataset_id: str,
    title: str,
    subtitle: str,
    source: str,
    target: str,
    target_unit: str,
    business_goal: str,
    output_dir: Path,
    notes: str,
    example_inputs: dict | None = None,
) -> dict:
    print("=" * 72)
    print(title)
    print("CRISP-DM  |  sklearn Multiple Linear Regression")
    print("=" * 72)

    print("\nPHASE 1  Business Understanding  商業理解")
    print(business_goal)

    frame = load_frame(source)
    print("\nPHASE 2  Data Understanding  資料理解")
    print(f"Source: {source}")
    print(f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print("Columns:", ", ".join(map(str, frame.columns)))
    print("Missing values:")
    print(frame.isna().sum().to_string())
    print("\nHead:")
    print(frame.head().to_string(index=False))

    X, y, schema, reference = encode(frame, target)
    features = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("\nPHASE 3  Data Preparation  資料準備")
    print(f"Target: {target}")
    if reference:
        print("Categorical reference levels (dropped):")
        for col, level in reference.items():
            print(f"  {col} -> {level}")
    print("Encoded predictors:", ", ".join(features))
    print(f"Train size: {len(X_train)}    Test size: {len(X_test)}")
    print(f"Split: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")

    print("\nPHASE 4  Modeling  建模")
    print("Estimator: sklearn.linear_model.LinearRegression")
    print("Feature selection: LassoCV, Forward SFS, RFE, ElasticNetCV, Exhaustive Search")
    paths = selection_paths(X_train, y_train, X_test, y_test, features)
    algorithms = {}
    best = None
    for name in ALGORITHMS:
        models = [evaluate_subset(subset, X_train, X_test, y_train, y_test) for subset in paths[name]]
        algorithms[name] = models
        winner = min(models, key=lambda item: item["rmse"])
        print(
            f"  {name:24s}  best k={winner['k']:<2d}  "
            f"RMSE={winner['rmse']:.4f}  R2={winner['r2']:.4f}  {winner['features']}"
        )
        if best is None or winner["rmse"] < best["model"]["rmse"]:
            best = {"algorithm": name, "model": winner}

    print("\nPHASE 5  Evaluation  評估")
    focus = algorithms["Lasso (L1 Ranking)"]
    print("\nLasso path on the test set:")
    print(f"{'k':>3} {'RMSE':>12} {'R2':>10}  Selected Features")
    for model in focus:
        print(f"{model['k']:>3} {model['rmse']:12.4f} {model['r2']:10.6f}  {model['features']}")
    print(
        f"\nLowest test RMSE: {best['algorithm']} k={best['model']['k']} "
        f"RMSE={best['model']['rmse']:.4f}"
    )
    full = algorithms["Lasso (L1 Ranking)"][-1]
    print(f"Full model k={full['k']} RMSE={full['rmse']:.4f} R2={full['r2']:.4f}")

    # Prefer the simplest model within 1% of the best test RMSE.
    candidates = []
    for name in ALGORITHMS:
        for model in algorithms[name]:
            candidates.append((name, model))
    best_rmse = min(model["rmse"] for _, model in candidates)
    near = [(name, model) for name, model in candidates if model["rmse"] <= best_rmse * 1.01]
    deployed_name, deployed = sorted(near, key=lambda item: (item[1]["k"], item[1]["rmse"]))[0]
    print(
        f"Deployed model (simplest within 1% of best RMSE): "
        f"{deployed_name} k={deployed['k']} {deployed['features']}"
    )
    print("Coefficients:")
    for coef in deployed["coefficients"]:
        vif = "" if coef["vif"] is None else f"  VIF={coef['vif']:.3f}"
        print(
            f"  {coef['name']:24s} {coef['coef']:12.6f}  "
            f"se={coef['se']:.6f}  t={coef['t']:.3f}  p={coef['p']:.4g}{vif}"
        )

    print("\nPHASE 6  Deployment  部署")
    encoded_example = {}
    if example_inputs:
        for field in schema:
            if field["kind"] == "number":
                encoded_example[field["key"]] = float(example_inputs.get(field["key"], field["default"]))
            else:
                chosen = str(example_inputs.get(field["key"], field["reference"]))
                for level, dummy in field["dummies"].items():
                    encoded_example[dummy] = 1.0 if chosen == level else 0.0
    else:
        for field in schema:
            if field["kind"] == "number":
                encoded_example[field["key"]] = field["default"]
            else:
                for dummy in field["dummies"].values():
                    encoded_example[dummy] = 0.0

    prediction = 0.0
    for coef in deployed["coefficients"]:
        if coef["name"] == "Intercept":
            prediction += coef["coef"]
        else:
            prediction += coef["coef"] * float(encoded_example.get(coef["name"], 0.0))
    print(f"Example prediction for {target}: {prediction:.4f} {target_unit}")
    print(f"Artifacts: {output_dir}")

    result = {
        "id": dataset_id,
        "title": title,
        "subtitle": subtitle,
        "target": target,
        "targetUnit": target_unit,
        "source": source,
        "notes": notes,
        "businessGoal": business_goal,
        "n": int(len(X)),
        "nTrain": int(len(X_train)),
        "nTest": int(len(X_test)),
        "split": f"test_size={TEST_SIZE}, random_state={RANDOM_STATE}",
        "reference": reference,
        "encodedFeatures": features,
        "predictors": schema,
        "best": {
            "algorithm": best["algorithm"],
            "k": best["model"]["k"],
            "features": best["model"]["features"],
            "rmse": best["model"]["rmse"],
            "r2": best["model"]["r2"],
        },
        "fullModel": {
            "k": full["k"],
            "rmse": full["rmse"],
            "r2": full["r2"],
            "features": full["features"],
        },
        "deployed": {
            "algorithm": deployed_name,
            "k": deployed["k"],
            "features": deployed["features"],
            "rmse": deployed["rmse"],
            "r2": deployed["r2"],
            "coefficients": deployed["coefficients"],
            "exampleInputs": example_inputs or {},
            "prediction": prediction,
            "testActual": deployed["testActual"],
            "testPredicted": deployed["testPredicted"],
        },
        "algorithms": {
            name: [public_model(model) for model in models] for name, models in algorithms.items()
        },
    }
    result["deployed"].update(
        {
            "testActual": deployed["testActual"],
            "testPredicted": deployed["testPredicted"],
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_tables(result, output_dir)
    save_plots(result, output_dir)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def write_preset(result: dict, preset_path: Path):
    presets = []
    if preset_path.exists():
        presets = json.loads(preset_path.read_text(encoding="utf-8"))
    web_result = json.loads(json.dumps(result))
    web_result["deployed"].pop("testActual", None)
    web_result["deployed"].pop("testPredicted", None)
    presets = [item for item in presets if item.get("id") != result["id"]]
    presets.append(web_result)
    order = {"startups": 0, "boston": 1}
    presets.sort(key=lambda item: order.get(item["id"], 9))
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text(json.dumps(presets, ensure_ascii=False), encoding="utf-8")
