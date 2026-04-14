## Piecewise Linear Curves

_Piecewise linear curves = constant + sum set of activation functions_

![Linear Curves](<./images/Linear Curves.png>)

可以用piecewise linear curves去逼近任何連續的曲線，而piecewise linear又可以用各種activation functions組合而成

# Machine Learning steps

Machine Learning tasks include:

- Regression
- Classification
- Structured prediction (optional)

## Step 1. Function with unknown

**_Model:_**

$$y=b+wx$$

## Step 2. Define Loss from Training data

Loss is a function of parameters e.g. $L(b,w)$  
 Loss: $L=\frac{1}{N}( Σ(e) ) e$ 為每筆資料的預設跟實際的誤差，N 為總資料數  
 Loss 越大代表參數越差  
 計算誤差的方式:  
 $e=|y-y'|$ L is mean absolute error(MAE)  
 $e=(y-y')^2$ L is mean square error(MSE)  
 如果 y 為機率表示的話=>Cross-entropy

## Step 3. Optimization

Gradient Descent  
 ![Gradient descent](./images/Gradient%20descent.png "Gradient descent")

- (Randomly) Pick an initial value $w_1$
- Compute $\frac{L'}{w'}| w = w_{1}$ , Negative => Increase Positive => decrease $w$
- Update $w$ iteratively

![Global minima & Local minima](./images/Global%20minima%20&%20Local%20minima.png "Global minima & Local minima")  
![Optimization](./images/Optimization.png "Optimization")
