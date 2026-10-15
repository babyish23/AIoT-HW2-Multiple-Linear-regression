# HW2 多元線性迴歸

這份專案從 HW1（50 Startups）複製後延伸。主程式改為 King County 房價的多元線性迴歸，仍走 CRISP-DM，並保留 HW1 的 `solve_50_startups.py`。

## 為什麼用這個資料集

作業要求 10–20 個特徵，補充條件是 14–39 個。

Kaggle [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) 去掉 `id`、`date` 後有 **18 個原始預測變數**，兩個範圍都符合。目標是 `price`。

補充建議的 Predict the Flood 本來更合適（20 個數值特徵）。不需 Kaggle 登入就能下載的公開 CSV，目標剛好等於 `0.005 × 特徵總和`，線性回歸的測試 R² 會變成 1，特徵選擇沒有意義，所以沒有採用。

## 執行

```bash
pip install -r requirements.txt
python hw2.py
```

分割是 `test_size=0.2`、`random_state=0`。特徵選擇只用訓練集。

## 結果

全部 18 個特徵的 OLS：

- Test MSE：36,326,416,754
- Test RMSE：190,595
- Test R²：0.6950

測試 MSE 最低的是 Forward SFS，k=14：

- Test MSE：36,274,683,097
- Test RMSE：190,459
- Test MAE：122,125
- Test R²：0.6954
- 特徵：`sqft_living`, `lat`, `view`, `grade`, `yr_built`, `waterfront`, `bedrooms`, `bathrooms`, `zipcode`, `long`, `condition`, `sqft_above`, `sqft_living15`, `yr_renovated`

RidgeCV 的測試 R² 也是 0.6950，和全部特徵的 OLS 幾乎一樣。

預測圖在 `output/kingcounty/prediction_interval.png`：深色是 95% 信賴區間，淺色是 95% 預測區間。殘差圖在 `output/kingcounty/residual_diagnostics.png`。殘差右偏，而且房價越高誤差越大。

報告：`report/hw2_report.pdf`。NotebookLM 摘要已換入正文，截圖在同一份 PDF。ChatGPT 對話以 ChatGPT Exporter 匯出，標成「GPT 輔助內容」，全文在 `report/GPT輔助內容.pdf`，並附在報告後面。

主程式請改名為 `學號_hw2.py`，連同報告壓成 `學號_hw2.zip`。
