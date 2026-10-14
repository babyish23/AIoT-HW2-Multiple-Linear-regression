"""Boston Housing uses the same CRISP-DM workflow as 50 Startups.

Nothing in the modeling steps is rewritten here. This file only supplies a
different dataset, target, and business question.

Run:
    python solve_boston_housing.py
"""

from pathlib import Path

from mlr_workflow import run_crispdm, write_preset

ROOT = Path(__file__).resolve().parent


def main():
    result = run_crispdm(
        dataset_id="boston",
        title="Boston Housing",
        subtitle="預測住宅價格中位數 medv",
        source="https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        target="medv",
        target_unit="1000 USD",
        business_goal=(
            "用社區特徵預測住宅價格中位數 medv（單位：千美元）。"
            "13 個變數不必全部保留；用同一套 sklearn 特徵選擇，"
            "看測試 RMSE 在哪個特徵數停止下降。"
        ),
        output_dir=ROOT / "output" / "boston",
        notes="目標欄位是 medv。chas 已是 0/1，其餘欄位為數值。",
        example_inputs=None,
    )
    write_preset(result, ROOT / "data" / "presets.json")
    deployed = result["deployed"]
    print("\nDecision")
    print(
        f"最低測試 RMSE 是 {result['best']['algorithm']} k={result['best']['k']} "
        f"RMSE={result['best']['rmse']:.4f}。"
    )
    print(
        f"上線模型取最簡單且誤差落在最佳值 1% 以內的子集: "
        f"{deployed['algorithm']} k={deployed['k']}  "
        f"預測 medv = {deployed['prediction']:.4f}"
    )
    print("特徵:", ", ".join(deployed["features"]))


if __name__ == "__main__":
    main()
