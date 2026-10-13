from pathlib import Path
import nbformat as nbf
out = Path.cwd() / "Logistic_Regression"
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
md("""
# Boston Housing：單變量線性迴歸
姓名：＿＿＿＿　學號：＿＿＿＿

本作業以 MEDV 為應變數，對其餘 13 個欄位各建立一個模型，共 **13 個單變量線性迴歸模型**。14 個欄位包含 13 個輸入及目標 MEDV；這是線性迴歸，不是邏輯斯迴歸。

資料來源為提供的 Data/housing(2).data，欄位定義依據 housing(2).names 與所附 UCI PDF 第 1–2 頁。每筆為地區層級的統計；MEDV 為自有住宅價值中位數，單位是千美元。

## 執行方式
在 Jupyter 或 VS Code 開啟本檔，選擇具有 NumPy、pandas、Matplotlib 與 ipykernel 的 Python 核心，執行 **Restart Kernel and Run All**。工作目錄可為本檔所在資料夾或 NFU 根目錄。若缺少套件，取消下一格的註解執行，完成後重啟核心。
""")
code("# %pip install numpy pandas matplotlib ipykernel")
derivation = r"""
## 一、手寫推導
設資料為 $(x_i,y_i), i=1,\ldots,n$，預測值為 $\hat y_i=w_1x_i+w_0$。最小化平方誤差和：
$$S(w_1,w_0)=\sum_{i=1}^n(y_i-w_1x_i-w_0)^2.$$

**步驟 1：對截距偏微分並令其為零。**
$$\frac{\partial S}{\partial w_0}=-2\sum_{i=1}^n(y_i-w_1x_i-w_0)=0.$$
$$\sum y_i=w_1\sum x_i+n w_0.$$
定義 $\bar x=\frac{\sum x_i}{n}$、$\bar y=\frac{\sum y_i}{n}$，得到
$$\boxed{w_0=\bar y-w_1\bar x}.$$

**步驟 2：對斜率偏微分並令其為零。**
$$\frac{\partial S}{\partial w_1}=-2\sum_{i=1}^n x_i(y_i-w_1x_i-w_0)=0.$$
$$\sum x_iy_i=w_1\sum x_i^2+w_0\sum x_i.$$

**步驟 3：代入截距並整理。**
$$\sum x_iy_i=w_1\sum x_i^2+(\bar y-w_1\bar x)\sum x_i.$$
$$w_1\left(\sum x_i^2-n\bar x^2\right)=\sum x_iy_i-n\bar x\bar y.$$
因此
$$\boxed{w_1=\frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}
=\frac{n\sum x_iy_i-(\sum x_i)(\sum y_i)}{n\sum x_i^2-(\sum x_i)^2}},$$
$$\boxed{w_0=\bar y-w_1\bar x}.$$

**步驟 4：最小值條件。**
平方誤差和是凸二次函數；若 $x$ 不全相同，其 Hessian
$$H=2\begin{pmatrix}\sum x_i^2&\sum x_i\\\sum x_i&n\end{pmatrix}$$
為正定，因為 $n>0$ 且 $\det(H)=4n\sum(x_i-\bar x)^2>0$。因此上述解為唯一全域最小值。若 $x$ 全相同，分母為零，斜率與截距無法分別唯一決定。

## 二、RM 數值代入範例
第一項要求推導公式，沒有要求逐筆手算 506 筆資料。以下加總可作手寫補充，程式也會重算供核對。

|數量|數值|
|---|---:|
|$n$|506|
|$\sum x_i$|3180.025|
|$\sum y_i$|11401.600|
|$\sum x_i^2$|20234.598247|
|$\sum x_iy_i$|73924.077600|

$$w_1=\frac{506(73924.077600)-3180.025(11401.600)}{506(20234.598247)-(3180.025)^2}\approx9.102109.$$
$$w_0=\frac{11401.600}{506}-9.102108981\frac{3180.025}{506}\approx-34.670621.$$
$$\boxed{\widehat{MEDV}=9.102109\,RM-34.670621}.$$
若 $RM=6$，$\widehat{MEDV}\approx19.942033$ 千美元。這是地區統計量的關聯，不是房間數增加造成房價上升的因果結論。
"""
md(derivation)
md("""## 三、讀取與檢查資料
檔案沒有標題列，且以空白分隔。使用 header=None 避免遺失第一筆資料。""")
code(r"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda v: f'{v:.6f}')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
candidates = [Path('Data/housing(2).data'),
              Path('Logistic_Regression/Data/housing(2).data')]
data_path = next((p.resolve() for p in candidates if p.is_file()), None)
if data_path is None:
    raise FileNotFoundError('請從 Notebook 資料夾或 NFU 根目錄執行。')
df = pd.read_csv(data_path, sep=r'\s+', header=None)
assert df.shape == (506, 14), f'非預期的資料大小：{df.shape}'
df.columns = columns
assert np.isfinite(df.to_numpy(dtype=float)).all(), '資料存在缺值或非有限數值'
features = columns[:-1]
y = df['MEDV'].to_numpy(dtype=float)
output_dir = data_path.parent.parent / 'housing_results'
output_dir.mkdir(exist_ok=True)
print(f'樣本數：{len(df)}；輸入欄位數：{len(features)}；缺值數：{df.isna().sum().sum()}')
display(df.head())
""")
md(r"""## 四、以公式建立 13 個模型
直接實作手寫推導的公式。$R^2=1-SSE/SST$，$RMSE=\sqrt{SSE/n}$。

本次以全資料擬合並畫圖，$R^2$ 與 RMSE 都是**樣本內指標**，不代表對新資料的預測表現。RMSE 單位與 MEDV 相同，都是千美元。""")
code(r"""
def fit_simple_linear(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    denominator = np.dot(xc, xc)
    if denominator == 0:
        raise ValueError('x 不可全部相同。')
    w1 = np.dot(xc, yc) / denominator
    w0 = y.mean() - w1 * x.mean()
    return w1, w0

models, rows = {}, []
for feature in features:
    x = df[feature].to_numpy(dtype=float)
    w1, w0 = fit_simple_linear(x, y)
    y_hat = w1 * x + w0
    sse = np.sum((y - y_hat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    models[feature] = (w1, w0)
    rows.append({'Feature': feature, 'w1': w1, 'w0': w0,
                 'R2_in_sample': 1 - sse / sst,
                 'RMSE_in_sample': np.sqrt(sse / len(y)),
                 'Equation': f'MEDV = {w1:.6f} * {feature} {w0:+.6f}'})
results = pd.DataFrame(rows)
display(results)
results.to_csv(output_dir / 'regression_results.csv', index=False, encoding='utf-8-sig')
""")
md("## 五、獨立核對係數與手算加總\n以 NumPy 矩陣最小平方法核對全部 13 組係數。")
code(r"""
for feature in features:
    x = df[feature].to_numpy(dtype=float)
    design = np.column_stack([x, np.ones(len(x))])
    reference, *_ = np.linalg.lstsq(design, y, rcond=None)
    np.testing.assert_allclose(models[feature], reference, rtol=1e-10, atol=1e-10)
assert len(models) == 13
print('13 組係數皆與 np.linalg.lstsq 一致。')
x = df['RM'].to_numpy(dtype=float)
display(pd.Series({'n': len(x), 'sum_x': x.sum(), 'sum_y': y.sum(),
                   'sum_x_squared': x @ x, 'sum_xy': x @ y,
                   'x_mean': x.mean(), 'y_mean': y.mean()}, name='RM 加總'))
w1, w0 = models['RM']
print(f'RM = 6 時，預測 MEDV = {w1 * 6 + w0:.6f} 千美元')
""")
md("""## 六、全部 13 組散佈圖與迴歸線
橫軸為單一輸入，縱軸為 MEDV。用遞增的 x_line 畫迴歸線，避免將未排序的樣本連成折線。CHAS 是二元變數，散點集中在 0 與 1。""")
code(r"""
def draw_regression(ax, feature):
    x = df[feature].to_numpy(dtype=float)
    w1, w0 = models[feature]
    r2 = results.loc[results['Feature'] == feature, 'R2_in_sample'].iloc[0]
    ax.scatter(x, y, s=15, alpha=0.45, color='#2878a0', edgecolors='none', label='Observed')
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, w1 * x_line + w0, color='#d64b35', linewidth=2, label='OLS fit')
    ax.set_title(f'{feature}: MEDV = {w1:.3f} x {w0:+.3f}\nIn-sample R² = {r2:.3f}')
    ax.set_xlabel(feature)
    ax.set_ylabel('MEDV (thousand USD)')
    ax.grid(alpha=0.2)
    ax.legend(loc='best', fontsize=8)

fig, axes = plt.subplots(5, 3, figsize=(16, 21), constrained_layout=True)
for ax, feature in zip(axes.flat, features):
    draw_regression(ax, feature)
for ax in list(axes.flat)[len(features):]:
    ax.set_visible(False)
fig.suptitle('Boston Housing: 13 separate simple linear regressions', fontsize=18)
fig.savefig(output_dir / 'all_13_regressions.png', dpi=160)
plt.show()
plt.close(fig)
""")
md("## 七、RM 放大圖與個別圖檔\n顯示 RM 範例，並將全部 13 張個別圖存入 housing_results。")
code(r"""
for feature in features:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    draw_regression(ax, feature)
    fig.savefig(output_dir / f'MEDV_vs_{feature}.png', dpi=160)
    if feature == 'RM':
        plt.show()
    plt.close(fig)
print('輸出資料夾：', output_dir)
""")
md("## 八、結果比較與討論\n不同欄位單位不同，不能直接以斜率絕對值比較重要性。以下按樣本內 R² 排序。")
code(r"""
display(results[['Feature', 'w1', 'w0', 'R2_in_sample', 'RMSE_in_sample']]
        .sort_values('R2_in_sample', ascending=False).reset_index(drop=True))
print('MEDV 最大觀測值：', y.max(), '；等於該值的筆數：', int((y == y.max()).sum()))
""")
md(r"""
- LSTAT 的單變量線性模型有最高的樣本內 $R^2$（約 0.544），斜率為負；RM 次之（約 0.484），斜率為正。
- RM 每增加 1，模型擬合值增加約 9.102 千美元。這是地區平均房間數和地區房價中位數的關聯，不代表因果效應。
- CHAS 的 0、1 預測值分別等於兩組 MEDV 的平均數。介於 0 與 1 的值不符合該欄位原始定義。
- 散點可見非線性與離群值；MEDV 有多筆集中在最大觀測值 50，單條直線只能描述部分變化。
- B 涉及原始資料的種族比例轉換，LSTAT 是來源中的社經分類；本作業按原始欄位完成歷史資料練習，不能據此對群體做因果判斷。
- 若另要評估對新資料的預測能力，需要切分訓練與測試資料，另算測試集指標。

## 繳交前檢查
1. 手寫：損失函數、兩個偏微分、代入整理、斜率與截距公式，以及分母非零條件。RM 數值範例可作補充。
2. Notebook：補上姓名學號，重新執行全部儲存格，確認 13 組方程式和圖表完整，再儲存。
3. 保留 Data/housing(2).data 的相對路徑，讓老師能重新執行；housing_results 是額外匯出的圖片和結果表。
4. 題目提到的 RM 範例圖未出現在提供的 UCI PDF 中；本檔依文字要求繪圖，未假定老師範例的排版。

資料依據：Data/housing(2).names 與 Data/UCI Machine Learning Repository_ Housing Data Set.pdf。
""")
nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python', 'version': '3.12'}}
nbf.write(nb, out / 'python.ipynb')
(out / 'handwritten_derivation.md').write_text('# 手寫推導參考\n\n' + derivation.strip() + '\n', encoding='utf-8')
print(out / 'python.ipynb')
