# Define Loss from Training Data

How good a set of values is. The smaller, the better.

Define:

$$L(b, w)$$

$y = b + wx_1$

- $b$: bias
- $w$: weight

label 在 machine learning 中代表正確答案。

Loss is defined as:
$$
L = \frac{1}{N}\sum_{n=1}^{N} e_n
$$  
![](<./images/Loss function_2.png>)
![loss function](<./images/Define Loss.png>)

e有兩種方法:  
### **均方誤差MAE(Mean absolute error)**  
$$
e_{1} = |y_{1} - \hat{y}_{1}|
$$

缺點:  
當兩筆資料的誤差為一正一負，相加為零，此時會出現 `loss = 0` 的問題  

### **絕對值誤差MSE(Mean square error)**
$$
e_{1} = \cfrac{1}{N} \sum_{i=1}^{n}(y_{1} - \hat{y}_{1})^2
$$
缺點:在單位上較難 or 無法解釋數據

### **RMSE(Root Mean square error)**
$$
e_{1} =\sqrt{\cfrac{1}{N} \sum_{i=1}^{n}(y_{1} - \hat{y}_{1})^2}
$$

If y and 
$ \hat{y} $
are both probability distributions => Cross-entropy