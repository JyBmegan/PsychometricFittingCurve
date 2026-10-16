# Psychometric-Function Analysis for SE-AlexNet

从 21 点模型预测与人类行为数据复现论文中的 PSE、拟合质量、ΔPSE 与 ANOVA 表格。

## 复现

```bash
pip install -r requirements.txt
python3 Code/reproduce_psychometric_results.py
```

脚本校验 136 个预测文件，并把全部结果表写入 `Results/`。

## 输入

| 路径 | 内容 |
| --- | --- |
| `Data/model_predictions_21point/` | 136 个预测文件：120 个 SE-AlexNet 条件 + 16 个 AlexNet/VGG16 基线，每个文件含 21 个反应概率 |
| `Data/human-data-9point.csv` | 9 个刺激强度上的人类平均反应概率 |
| `Code/Human_PSE_40_Subjects.csv` | 40 名被试各自的 PSE 与曲线参数 |

分析从每个 21 点序列中取第 1、5、7、9、11、13、15、17、21 个点，对应刺激强度 `0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0`。

掩码标签为 `Full`（完整）、`E`（遮眼）、`N`（遮鼻）、`M`（遮嘴）。`Location-1/2/3` 是三个全连接 SE 插入位置（前序/中序/后序）。压缩比为 2、4、8、16、32。

## 输出

`Results/` 下共 11 张表，其中论文直接使用三张：

| 输出 | 内容 |
| --- | --- |
| `Paper_Table1_PSE_RMSE.csv` | 论文表 1：各模型 PSE、t、p、Cohen's d、模型-人类曲线 RMSE |
| `Paper_Table2_Fit_Quality.csv` | 论文表 2：拟合 R² 与 RMSE |
| `Paper_Table3_Occlusion.csv` | 论文表 3：遮挡条件下各模型的 PSE 统计 |

其余为中间数据：`Model_Psychometric_Results.csv`（全部 136 个模型的完整统计）、`Human_Psychometric_Summary.csv`、`DeltaPSE_Long/Wide.csv`、`Repeated_Measures_ANOVA.csv`、三个 `PostHoc_*.csv`。

## 心理物理函数与统计量

累积高斯心理物理函数（对 9 个点拟合）：

$$
y = g + (1-g-l)\cdot\frac12\left[1+\operatorname{erf}\left(\frac{x-u}{\sqrt{2v^2}}\right)\right]
$$

- $g$：下渐近线（guess）；$l$：上渐近线损失（lapse）；$u$：主观相等点 PSE；$v$：曲线尺度。
- PSE 处斜率：$\displaystyle\frac{1-g-l}{\sqrt{2\pi}\,v}$。

拟合边界与初值：

```text
上界 [0.05, 0.05, 1.0, 100.0]
初值 [0.01, 0.02, 0.5,   5.0]
下界 [0.00, 0.00, 0.0,   0.1]
```

模型 PSE 对 40 个人类 PSE 的双侧单样本 t 检验：

$$
t=\frac{\bar{PSE}_{human}-PSE_{model}}{SD_{human}/\sqrt{N}},
\qquad
|d|=\frac{|PSE_{model}-\bar{PSE}_{human}|}{SD_{human}}
$$

个体级 ΔPSE：

$$
\Delta PSE_{ij}=|PSE_{model,i}-PSE_{human,j}|
$$

对 ΔPSE 做 2（预训练类型）× 3（插入位置）× 5（压缩比）重复测量 ANOVA，以及 Bonferroni 校正的事后配对 t 检验。
