# Define Loss from Training Data

_How good a set of values is. The smaller, the better._

Define:

$$L(b, w)$$

$y_{i} = b + wx_i$

- $b$: bias
- $w$: weight

label 在 machine learning 中代表正確答案。

_Loss is defined as:_

$$
L = \frac{1}{n}\sum_{i=1}^{n} e_i
$$

- $e$ 為每筆資料的預設跟實際的誤差，n 為總資料數

  ![](<./images/Loss function_2.png>)
  ![loss function](<./images/Define Loss.png>)

$e$有兩種方法:

### **均方誤差MAE(Mean absolute error)**

$$
e_{i} = |y_{i} - \hat{y}_{i}|
$$

缺點:

- 當兩筆資料的誤差相同，相減為零再取絕對值，此時會出現 `loss = 0` 的問題 (e.g $y=100，\hat{y}=100$)
- optimization比較難
- 在誤差為 0 附近不可微（non-differentiable）
- 對 outlier 不敏感（不像 MSE 會放大誤差）

### **平方誤差MSE(Mean square error)**

$$
MSE = \cfrac{1}{n} \sum_{i=1}^{n} e_i
$$

$$
e_{i} = (y_{i} - \hat{y}_{i})^2
$$

缺點:在單位上較難 or 無法解釋數據，且平方放大誤差

### **RMSE(Root Mean square error)**

$$
RMSE =\sqrt{\cfrac{1}{n} \sum_{i=1}^{n}e_i}
$$

$$
e_{i} = (y_{i} - \hat{y}_{i})^2
$$

If d
$y$
and
$\hat{y}$
are both probability distributions => **Cross-entropy**  
分類問題經常使用**Cross-entropy**

## Cross-entropy

cross-entropy 是用來觀測預測的機率分布與實際機率分布的誤差範圍  
corss-entropy 越高，代表內涵的資訊量越大，不確定越多，誤差越高  
[何謂 Cross-Entropy (交叉熵)](https://r23456999.medium.com/%E4%BD%95%E8%AC%82-cross-entropy-%E4%BA%A4%E5%8F%89%E7%86%B5-b6d4cef9189d)

| Loss          | 特點                   | 適用           |
| ------------- | ---------------------- | -------------- |
| MAE           | robust to outliers     | regression     |
| MSE           | penalizes large errors | regression     |
| Cross-entropy | probability comparison | classification |
