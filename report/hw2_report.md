資料來源
Kaggle「House Sales in King County, USA」。
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
本地檔 data/kc_house_data.csv，共 21613 筆、21 欄。目標是 price。去掉 id 與 date 後有 18 個原始預測變數，落在作業要求的 10–20，也落在補充條件的 14–39。

為什麼不是 Predict the Flood
補充建議的第一名是 Kaggle Playground S4E5，原始 20 個數值特徵，本來最合適。不需登入就能下載的那份公開 CSV，FloodProbability 恰好等於 0.005 乘上 20 個特徵的總和，線性回歸測試 R² 為 1。那種資料做特徵選擇沒有意義，所以沒有拿來當成果。

【CRISP-DM】
1. Business Understanding：買家與估價需要一個可解釋的成交價模型，而不是只追求黑盒分數。
2. Data Understanding：21613 筆、無缺值。價格最小 75000、中位數 450000、最大 7700000。sqft_living、sqft_above、sqft_basement 尺度大且彼此相關。
3. Data Preparation：刪除 id、date。80/20 分割，random_state=0，訓練 17290、測試 4323。特徵選擇只用訓練集。
4. Modeling：sklearn LinearRegression。五條路徑是 Lasso 係數排序、Forward SFS、RFE、SelectKBest、Permutation Importance。另報 RidgeCV 當收縮對照，不混進 OLS 排名。
5. Evaluation：全部特徵的測試 MSE 是 36326416754.06，R² 是 0.6950。最好的 OLS 子集是 Forward SFS，k=14，測試 MSE 36274683097.02，RMSE 190459.14，MAE 122124.90，R² 0.6954。RidgeCV 測試 MSE 36323730877.54，R² 0.6950。
6. Deployment：網頁用這組係數做即時估價，並畫出 95% 信賴區間與 95% 預測區間。

採用的特徵
sqft_living, lat, view, grade, yr_built, waterfront, bedrooms, bathrooms, zipcode, long, condition, sqft_above, sqft_living15, yr_renovated

回歸假設
殘差偏態 3.540。絕對殘差與配適值的相關 0.496。Durbin-Watson 2.010。最大 VIF 7.854。
房價右偏，殘差不是常態，而且大房子的誤差比較大，變異數不齊。這是這類資料的常見現象，不是程式算錯。對數價格是公開分析裡常見的下一步；本報告主模型仍用原始美元，讓區間可以直接讀成價格。VIF 若偏高，通常來自家中面積欄位彼此重疊，特徵選擇就是在處理這件事。

【主流解法比較】
King County 的公開筆記本多半把 price 取對數，再把 zipcode 做成目標編碼或分組，然後用 Random Forest 或 Gradient Boosting。那些模型的測試 R² 常常高於單純線性回歸，但係數不好直接解釋，也沒有線性模型這種預測區間。這份作業保留線性回歸，用五種特徵選擇處理共線性，並把區間畫出來。Ridge 用來對照收縮；它的測試誤差和全部特徵的 OLS 接近，因為這批資料的樣本數遠大於特徵數，收縮的幫助有限。

【NotebookLM 摘要】
以下是依公開的 King County 房價回歸做法整理的研究摘要，超過 100 字。正式繳交前，請把同主題資料貼進 NotebookLM，用它自己的摘要取代這一段，並在報告標明「NotebookLM 摘要」。King County 成交價幾乎都先看居住面積、等級 grade、臨水 waterfront 與經緯度。多數寫法會先取對數價格，因為原始價格右偏、殘差隨房價變大。zipcode 不適合直接當連續數字，常見做法是目標編碼或只留緯經度。線性回歸能給出每個特徵的美元係數，R² 多半落在 0.65 到 0.75；樹模型與梯度提升通常更高，但難以做預測區間。面積欄位高度相關，所以 Lasso、RFE 或手動刪掉 sqft_above 與 sqft_basement 其中一個，是公開解法裡反覆出現的步驟。本作業沿用這個脈絡，但主模型維持原始價格的 OLS，以便區間用美元呈現。

【GPT 輔助內容】
程式架構、CRISP-DM 分段、五種特徵選擇的比較、預測區間圖，以及這份報告草稿，是在 Cursor 裡由 GPT 協助完成。評分要求的 pdfCrowd ChatGPT 對話 PDF 無法由這裡代為登入匯出。請自行開啟 ChatGPT 對話，用 pdfCrowd 匯出 PDF，放進以學號命名的 zip，並在報告封面標明「GPT 輔助內容」。主程式檔名請改成學號_hw2.py 再繳交。
