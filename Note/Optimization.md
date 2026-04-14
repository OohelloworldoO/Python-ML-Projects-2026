# Optimization

_Gradient Descent_

![Gradient descent](./images/Gradient%20descent.png "Gradient descent")  
會有 local minimum, global minimum 的問題假議題 之後再更新為何為假議題  
partial的定義:  
對 $w$ partial(把另一個未知數當作常數)  
$$\frac{\partial f}{\partial w}|_{w = w^0, b = b^0}$$

對 $b$ partial  
$$\frac{\partial f}{\partial b}|_{w = w^0, b = b^0}$$

$$
Gradient = \begin{bmatrix}
\frac{\partial L}{\partial \theta_1}\\
\frac{\partial L}{\partial \theta_2}\\
\frac{\partial L}{\partial \theta_3}\\
\frac{\partial L}{\partial \theta_4}\\
.\\
.\\
.
\end{bmatrix}
= g
$$

$$
\theta = \begin{bmatrix}
\theta_1|_{\theta = \theta_0}\\
\theta_2|_{\theta = \theta_0}\\
\theta_3|_{\theta = \theta_0}\\
\theta_4|_{\theta = \theta_0}\\
.\\
.\\
.
\end{bmatrix}
$$

$$
g = \nabla L(\theta^0)
$$

$$
\theta^1 = \theta^0 - \eta g
$$

### Optimization of New Model

(Randomly) Pick initial values $\theta^0$

Compute gradient $g =\nabla L(\theta^0)$

$$
\theta^1 = \theta^0 - \eta g
$$

Compute gradient $g =\nabla L(\theta^1)$

$$
\theta^2 = \theta^1 - \eta g
$$

Compute gradient $g =\nabla L(\theta^2)$

$$
\theta^3 = \theta^2 - \eta g
$$
