# Define Loss from Training Data

How good a set of values is. The smaller, the better.

Define:

$L(b, w)$

$y_{i} = b + wx_i$

- $b$: bias
- $w$: weight

label 在 machine learning 中代表正確答案。

Loss is defined as:

$$
L = \frac{1}{N}\sum_{n=1}^{n} e_i
$$

![](<./images/Loss function_2.png>)
![loss function](<./images/Define Loss.png>)

e有兩種方法:

### **均方誤差MAE(Mean absolute error)**

$$
e_{i} = |y_{i} - \hat{y}_{i}|
$$

缺點:

- 當兩筆資料的誤差為一正一負，相加為零再取絕對值，此時會出現 `loss = 0` 的問題
- optimization比較難

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

# Optimization

Grandient Descent  
 ![Gradient descent](./images/Gradient%20descent.png "Gradient descent")  
會有 local minimum, global minimum 的問題(假議題 之後再更新為何為假議題） 
