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
King County 房價分析顯示，居住面積（sqft_living）與房屋等級（grade）為最關鍵變數。因 sqft_living 為 sqft_above 加地下室坪數，兩者高度相關，來源透過剔除 sqft_above 消除多重共線性；zipcode 因類別過多且無數字順序，不宜作為連續變數，需轉為類別編碼。為修正右偏與異質變異，來源對價格與坪數取對數，並以 Q-Q plot 與殘差圖檢查常態性。線性回歸具備高係數解釋力，但非線性預測力有限。樹模型（Random Forest、Gradient Boosting）預測力更高但缺乏直接解釋力。若作業規定使用線性回歸與特徵選擇，應保留 OLS 的顯著係數、p 值、VIF 及帶有區間的預測圖。在比較段落中，則應承認樹模型在捕捉非線性關係與預測精準度上通常優於線性回歸。

【GPT 輔助內容】
完整對話已用 ChatGPT Exporter 匯出為 PDF，附在本報告後面，共 23 頁，並另存為 report/GPT輔助內容.pdf。對話先討論符合條件的 Kaggle 資料集與 CRISP-DM 報告架構，接著依 King County 的實際設定撰寫正文：18 個特徵的 OLS、Forward SFS 的 14 個特徵、RidgeCV 對照、測試 MSE、RMSE、R²、五大回歸假設，以及信賴區間和預測區間的差別。對話也比較了對數價格、把 zipcode 當類別，以及 Random Forest、Gradient Boosting。這些模型常常預測得更好，但這次仍以可解釋的 OLS 作為主要結果。。
