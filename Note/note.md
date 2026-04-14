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
