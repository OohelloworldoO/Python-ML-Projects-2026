# Logistic Regression from Scratch

## 目標

使用 NumPy 從零開始實作 Logistic Regression，不依賴 sklearn 的模型 API

目標是深入理解：

- Linear model（線性模型）
- Sigmoid function
- Binary Cross-Entropy Loss
- Gradient Descent
- Decision Boundary

---

## 核心概念

Logistic Regression 是一種用來做二元分類（Binary Classification）的模型：

$$
P(y=1|x) = \sigma(w^T x + b)
$$

其中：

- \($\sigma$)：Sigmoid function
- \($w$)：權重（weights）
- \($b$)：偏差（bias）

---

## 🔧 實作內容

### model.py（模型本體）

實作內容包含：

- Sigmoid function
- Binary Cross-Entropy Loss
- Gradient Descent（參數更新）
- predict / predict_prob

這部分是「模型核心」，可以在不同資料集重複使用

---

### utils.py（工具）

包含：

- 資料視覺化（plot dataset）
- decision boundary 繪製

這部分是「輔助工具」，不同實驗可以共用

---

### experiment.ipynb（實驗流程）

包含：

- 載入 Iris dataset
- 將問題轉為 binary classification
- 訓練模型
- 計算 accuracy
- 視覺化 decision boundary

這部分是「本次實驗」，會隨 project 改變

---

## 實驗結果

- 模型成功學到 linear decision boundary
- 在此簡化資料上有良好表現
- Loss 隨訓練穩定下降

---

## 觀察（Observations）

- Logistic Regression 本質是 **線性分類器（linear classifier）**
- Sigmoid function 將 linear output 轉為機率（probability）
- 如果資料不是線性可分（non-linear separable），模型效果會受限
- Decision boundary 一定是直線

---

## 學到的東西

- Gradient Descent 如何更新參數
- Loss function 對訓練的影響
- 機率輸出（probability）與分類（classification）的差異
- 如何將模型與實驗流程分離（model vs experiment）

---

## 未來改進方向

- 加入 Regularization（L2）
- 測試不同 learning rate
- 延伸到 Multi-class classification
- 實作 Neural Network（MLP）

---

## 為什麼這樣拆（重點）

本專案將程式拆成三個部分：

### model.py（模型）

放「不會因資料改變的邏輯」

例如：

- sigmoid
- loss
- gradient
- fit / predict

---

### utils.py（工具）

放「會重複用到的功能」

例如：

- 畫圖
- decision boundary

---

### experiment.ipynb（實驗）

放「這次實驗的流程」

例如：

- 使用 iris dataset
- 訓練模型
- 觀察結果

---

## 核心原則

把「不變的」和「會變的」分開

- 模型 = 不變（可重用）
- 實驗 = 會變（每次不同）
- 工具 = 常用（可共用）

---

## 個人心得

這個專案讓我理解：

- Machine Learning 不只是套用 library
- 而是理解「模型如何運作」
- 以及如何把程式結構整理成可維護、可重用的形式
