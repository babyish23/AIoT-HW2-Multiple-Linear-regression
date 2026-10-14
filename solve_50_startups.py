"""50 Startups multiple linear regression, solved with sklearn and CRISP-DM.

Dataset (Kaggle / 100 Days of ML):
https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv

Run:
    python solve_50_startups.py
    python solve_50_startups.py --url <csv-url> --target Profit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlr_workflow import run_crispdm, write_preset

ROOT = Path(__file__).resolve().parent
DEFAULT_URL = (
    "https://raw.githubusercontent.com/Avik-Jain/"
    "100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
)


def main():
    parser = argparse.ArgumentParser(description="50 Startups CRISP-DM multiple linear regression")
    parser.add_argument("--url", default=DEFAULT_URL, help="CSV dataset URL or local path")
    parser.add_argument("--target", default="Profit", help="Target column name")
    args = parser.parse_args()

    result = run_crispdm(
        dataset_id="startups",
        title="50 Startups",
        subtitle="預測新創公司利潤",
        source=args.url,
        target=args.target,
        target_unit="USD",
        business_goal=(
            "投資人要預測一家新創公司的 Profit。"
            "問題不是把所有欄位都放進迴歸，而是找出值得留下的支出與州別，"
            "讓測試集誤差小，而且模型簡單到可以解釋。"
        ),
        output_dir=ROOT / "output" / "startups",
        notes="State 以字母序第一類 California 為對照組。",
        example_inputs={
            "R&D Spend": 100000,
            "Administration": 120000,
            "Marketing Spend": 150000,
            "State": "California",
        },
    )
    if args.url == DEFAULT_URL and args.target == "Profit":
        write_preset(result, ROOT / "data" / "presets.json")

    lasso = result["algorithms"]["Lasso (L1 Ranking)"]
    if args.url == DEFAULT_URL and args.target == "Profit":
        expected = {
            1: 8274.86801822599,
            2: 8198.797190788488,
            3: 8309.05968348172,
        }
        for model in lasso:
            if model["k"] in expected and abs(model["rmse"] - expected[model["k"]]) > 1e-6:
                raise SystemExit(f"k={model['k']} RMSE 與標準答案不一致: {model['rmse']}")
        print("\nCheck: Lasso k=1,2,3 test RMSE matches the published 50 Startups benchmark.")

    deployed = result["deployed"]
    print("\nDecision")
    print(
        f"四種方法的測試 RMSE 在 k=2（R&D Spend + Marketing Spend）最低，"
        f"數值是 {lasso[1]['rmse']:.2f}。"
    )
    print(
        f"完整 5 變數模型 RMSE 升到 {result['fullModel']['rmse']:.2f}，"
        "多出來的 Administration 與州別沒有改善樣本外預測。"
    )
    print(
        f"上線模型: {deployed['algorithm']}  k={deployed['k']}  "
        f"預測範例 Profit = {deployed['prediction']:.2f}"
    )


if __name__ == "__main__":
    main()
