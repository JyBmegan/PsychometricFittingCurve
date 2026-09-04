# Fitting Log for Psychometric Curve in SE-AlexNet Analysis

<p align="center">
    <a href="https://jybmegan.github.io/SE-AlexNet/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>


<p align="center">
    <a href="./0_InputImages/"><img src="https://img.shields.io/badge/Dataset-Github-orange" alt="Dataset: DFEW"></a>
    <a href="https://huggingface.co/JiayuMBao/SE-AlexNet"><img src="https://img.shields.io/badge/Model%20Weights-Hugging%20Face-yellow" alt="Model Weights: Hugging Face"></a>
</p>

## 1. Preparation

### 1.1 File Structure

**Folder - FitCurve:**

``` text
│  log.md
│
├─1_Coding
│      FitPsycheCurveWH.m
│      PsychometricFittingCurve.m
│
├─0_Data
│  │  affect_human.xlsx
│  │  human-data-9point.csv
│  ├─Conv
│  ├─AlexNet
│         AlexNet-FaceBased-E-raw.csv
│         AlexNet-FaceBased-Full-raw.csv
│         ...
│  ├─Location-1
│         Location-1-FaceBased-E-16.csv
│         Location-1-FaceBased-E-2.csv
│         ...
│  ├─Location-2
│         Location-2-FaceBased-E-16.csv
│         Location-2-FaceBased-E-2.csv
│         ...
│  ├─Location-3
│         Location-3-FaceBased-E-16.csv
│         Location-3-FaceBased-E-2.csv
│          ...
│  └─VGG-16
│         VGG-16-FaceBased-E-raw.csv
│         VGG-16-FaceBased-Full-raw.csv
│         ...
├─2_ConvResults
├─3_Results
│      Final_Analysis_Results.xlsx
│      Human_Fitting_Parameters.xlsx
│      Location-1-FaceBased-E.fig
│      Location-1-FaceBased-E.png
│      ...
├─4_Results_LineChart
│      Fitting_Results.xlsx
│      Human_Fitting_Parameters.xlsx
│      Location-1-FaceBased-E.fig
│      Location-1-FaceBased-E.png
│      ...
├─5_HumanPSECalculation
│      Final_Analysis_Results.xlsx
│  ├─Results                      # predict results of All Models (data for this analysis)
│  └─Z_Score_Analysis             # print results
│         Z_Score_FullFace_Table.xlsx
│         Human_PSE_40_Subjects.csv
│         Human_PSE_Summary_Log.txt
│         ZScore_FullFace_SE-Conv-L1.png
│         ...
```

### 1.2 Condition


**Model Type**: 
* FC Layers: SE-FC-L1, SE-FC-L2, SE-FC-L3
* Conv Layers (New): SE-Conv-L1, SE-Conv-L2, SE-Conv-L3, SE-Conv-L4

**Base**: FaceBased, ObjectBased

**Masked Area**: Eyes, Mouth, Nose, Full (no mask)

**Reduction (For SE-AlexNet Only)**: 2, 4, 8, 16, 32 

**Baseline**: Human (50 Participants)

## 2. Function: FitPsycheCurveWH.m

### 2.1 Function Specification

Cumulative Gaussian 进行非线性最小二乘拟合；使用 `erf` 为误差函数，反映被试在不同刺激强度下的反应概率。

$$y = g + (1 - g - l) \cdot \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x - u}{\sqrt{2v^2}} \right) \right]$$

* $g$: Guess Rate，对于 2AFC 实验，通常固定为0.5；对于其他 n-AFC，通常为 $1/n$
* $l$: Lapse Rate失误率，反映高强度刺激下的非感知性错误
* $u$: PSE or Threshold，主观相等点或感知阈值
* $v$: Sigma, 噪声参数，决定曲线斜率


### 2.2 Variable

#### Input

| 符号 | 定义 |
| :--- | :--- |
| `xAxis` | 刺激强度 (Stimulus Intensity) |
| `yData` | 反应正确率 (Proportion Correct) |

#### Parameter
| 符号 | 定义 |
| :--- | :--- |
| `u` | PSE / 阈值 (累积函数中心点) |
| `v` | 标准差 (控制曲线平滑度) |
| `g` | 猜测率 (下限值) |
| `l` | 失误率 (上限偏移) |

#### Output
| 符号 | 定义 |
| :--- | :--- |
| `slope` | 中点斜率 (敏感度) |
| `curve` | 高分辨率拟合曲线 |
| `ffit` | 拟合对象 (含 $R^2$, 残差) |



## 3. How to Compile

**Run** - `PsychometricFittingCurve.m`

**Automation Steps performed by the script:**

* Auto-Fetch Data: Checks ../ConvResults and automatically copies relevant CSVs to ../Data/Conv.

* Adaptive Loading: Automatically detects if data has 21, 11, or 9 rows and maps them to the correct stimulus intensity points.

* Fitting & Stats: Calculates PSE/Slope, runs T-tests against human data, calculates Cohen's d.

* Plotting: Generates smoothed psychometric curves for all models and human data.

**Output** - Originally Saved in Folder `Results`

## 4. Notes

### 4.1 About Constraints

无论是行为学实验还是model的任务，都要求被试/model在“快乐”与“悲伤”中做出分类判断，因此本实验属于 **2AFC Discrimination** Task （2AFC辨别任务）；此外，y轴被定义为Proportion of Happy Response。在极端悲伤 (Intensity=0) 的情况下，判定为“快乐”的概率应趋近于 0 (False Alarm Rate $\approx$ 0)，而非 0.5。

因此对于constraints, 本研究使用：

* *GuessRate* （$g$）[0, 0.05]：极端非快乐点误按“快乐”的概率，限制虚报率在 5% 以内。

* *LapseRate* （$l$） [0, 0.05]：极端快乐点因走神导致的漏报概率。上限设为心理物理学标准 0.05。

* *PSE* （$u$） [0, 1]：50%响应时的强度（分类边界）。预期值（humandata）约 0.51。0.5 为真正的中性判定点。

* *Slope* [0.001, 100]：知觉切换的敏锐度，允许极高的分类精度。预期值（humandata）约 4.19。

In Code ~Line 24~27：

``` Matlab
% Para Limitation
UL = [0.05, 0.05, 1, 100]; 
SP = [0.01, 0.02, 0.5, 5.0];  
LM = [0, 0, 0, 0.1]; 
```

注：其他任务
* Yes/No 任务：$ g \approx 0$，反映信号检测阈限（False Alarm）；
* 2AFC 检测：$ g = 0.5 $，曲线从半山腰开始，反映猜测正确率。

### 4.2 About Line / FittingCurve

共有三组地方需要改动（SE-AlexNet, AlexNet, VGG），每组地方有两处需要修改；以VGG16这一组为例↓：

``` Matlab
% Convert "curveV" into "~" if you want to draw LINE CHART for the Models 
            [ffit, ~, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM); %Comment this Line if you are drawing CURVE
            [ffit, curveV, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM); %Comment this Line if you are drawing LINE chart

% Change parameters for plotting if you want to draw LINE CHART for the Models
            plot(paraX, ySampled, '-o', 'Color', colors.vgg16, 'MarkerSize', 5, ...
                'MarkerFaceColor', colors.vgg16, 'LineWidth', 2.5, 'DisplayName', 'VGG 16'); %Comment this Line if you are drawing CURVE
            plot(curveV(:,1), curveV(:,2), '-', 'Color', colors.vgg16, 'LineWidth', 2.5, ...
                'HandleVisibility', 'on', 'DisplayName', 'VGG 16'); %Comment this Line if you are drawing LINE chart
            plot(paraX, ySampled, 'o', 'MarkerEdgeColor', colors.vgg16, 'MarkerFaceColor',...
                 colors.vgg16, 'MarkerSize', 5, 'HandleVisibility', 'off'); %Comment this Line if you are drawing LINE chart
```
### 4.3 Statistical Analysis

Added **One-Sample T-Test** to compare Model PSE vs. Human Population PSEs.

* Human Baseline: Fitted individual PSEs for 50 participants to derive Mean & SD.

* Metrics Exported (in Final_Analysis_Results.xlsx):

    * t_stat: T-value
    * p_value: Significance level
    * Cohens_d: Effect size ($d = |PSE_{model} - Mean_{human}| / SD_{human}$)

#### Z-Score Analysis

$$Z_{model} = \frac{PSE_{model} - \mu_{human}}{\sigma_{human}}$$

Z 值代表模型的决策阈值（PSE）距离 40 名普通人类平均水平有几个标准差。

图表中的浅灰色背景带界定了 95% 置信区间


### 4.4 Naming & Data Management

Using locConfigs structure to manage mixed naming conventions:

1. Old Format (Hyphen): `Location-1-FaceBased-E-2.csv`

    Mapped to Display Name: SE-FC-L1, SE-FC-L2, SE-FC-L3

2. New Format (Underscore): `SeC1_FaceBased_squeeze2_Full.csv`

    Mapped to Display Name: SE-Conv-L1, SE-Conv-L2, SE-Conv-L3, SE-Conv-L4

This ensures clean plot titles and Excel reports despite different underlying file naming rules.


## 5. Supp Analysis

### 5.1 Model Fit & Similarity Metrics (RMSE & R²)

To comprehensively evaluate the models beyond the single decision boundary (PSE), we introduced Root Mean Square Error (RMSE) and $R^2$. 

See in `7_SuppExp/0_Code/Comprehensive_RMSE_RSquare.m`

(Code in `5_HumanPSECalculation/RMSE_RSquare.m` saved the `1. Goodness of Fit (Model's Internal Consistency)` and it not include new rmse)

Please prepare related files before replicating:

```Matlab
dataRoot = '../../0_Data'; 
savePath = '..Results'; 
```

These metrics are divided into two distinct categories to answer different research questions:

#### 1. Goodness of Fit (Model's Internal Consistency)
Evaluates how well the model's output probabilities across the facial intensities fit a standard psychometric sigmoid curve. A low RMSE and high $R^2$ indicate the model performs stable, rule-based classification rather than random guessing.


* **RMSE (Fit) Formula:**

$$RMSE_{fit} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_{sampled, i} - y_{predicted, i})^2}$$

* **$R^2$ Formula:**

$$R^2 = 1 - \frac{\sum (y_{sampled, i} - y_{predicted, i})^2}{\sum (y_{sampled, i} - \bar{y}_{sampled})^2}$$

  *(Where $y_{sampled}$ is the model's actual output probability, and $y_{predicted}$ is the theoretical value on the fitted sigmoid curve).*

```Matlab
% Part A: 计算实际点与拟合曲线的差异 (Goodness of Fit)

% 1. 利用一维插值法(interp1)，从高分辨率的拟合曲线 (curveM) 中提取出实验 X 坐标对应的“理论预测 Y 值”
yPredicted = interp1(curveM(:,1), curveM(:,2), paraX', 'linear', 'extrap');

% 2. 计算残差 (真实点 和 拟合曲线上的点 的差距)
residuals_fit = ySampled - yPredicted;

% 3. 计算均方根误差 (Fit_RMSE)
Fit_RMSE = sqrt(mean(residuals_fit.^2));

% 4. 计算 R-squared (R方)
SST = sum((ySampled - mean(ySampled)).^2);
SSE = sum(residuals_fit.^2);
if SST ~= 0
    R_Square = 1 - (SSE / SST);
else
    R_Square = NaN; % 防止数学上分母为0
end
```

#### 2. Model-Human Deviation (Biological Plausibility)
Evaluates how far the model's actual reaction curve deviates from the average human baseline curve across all stimulus intensities. This proves whether the model processes extreme and neutral emotions similarly to humans, compensating for the limitations of comparing PSE alone.

* **RMSE (Human Similarity) Formula:**
  $$RMSE_{human} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(P_{model, i} - P_{human\_ave, i})^2}$$
  *(Where $P_{model}$ is the model's output probability, and $P_{human\_ave}$ is the average response probability of 40 human subjects at the same intensity).*

```Matlab
% Part B：计算模型与人类基线曲线的相似度 (Human_Sim_RMSE)
                    
% 1. 计算模型在9个点上的真实输出 (ySampled) 与 人类平均反应点 (humanAve) 之间的残差
residuals_human = ySampled - humanAve;

% 2. 计算均方根误差 (Human_Sim_RMSE)
Human_Sim_RMSE = sqrt(mean(residuals_human.^2));          
```

### 5.2 Delta-PSE ANOVA

**Code**:

`7_SuppExp/0_Code/ANOVA_Delta_PSE.m`

`7_SuppExp/0_Code/ANOVA_FollowUp_Analysis.m`

* **因变量 (DV)**：Delta PSE（模型与人类被试 PSE 的绝对差值，数值越小表示越像人）。
* **组内因子**：
    1. **Pretrain** (2个水平: Face, Object)
    2. **Location** (3个水平: L1, L2, L3)
    3. **Ratio** (5个水平: R2, R4, R8, R16, R32)

* **标准**：显著性水平 $\alpha = .05$。对于违反球形假设的情况，报告 Greenhouse-Geisser ($p_{GG}$) 校正后的结果。

#### 1. 三因素重复测量方差分析 (RM-ANOVA)

``表 1：RM-ANOVA 整体结果``

See in `7_SuppExp/2_ANOVA_Results/RM_ANOVA_Results.xlsx`

| 效应来源 (Source) | $SS$ | $df$ | $MS$ | $F$ | $p_{GG}$ | $\eta_p^2$ |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrain | 0.009 | 1 | 0.009 | 1.370 | .249 | .034 |
| Location | 0.006 | 2 | 0.003 | 1.235 | .279 | .031 |
| **Ratio** | 0.400 | 4 | 0.100 | 49.150 | **<.001** | **.558** |
| **Pre $\times$ Loc** | 0.124 | 2 | 0.062 | 7.293 | **.010** | **.158** |
| **Pre $\times$ Rat** | 0.808 | 4 | 0.202 | 135.840 | **<.001** | **.777** |
| **Loc $\times$ Rat** | 0.535 | 8 | 0.067 | 46.034 | **<.001** | **.541** |
| **Pre $\times$ Loc $\times$ Rat** | 0.687 | 8 | 0.086 | 26.918 | **<.001** | **.408** |

**分析结论**：由于**三阶交互作用显著** ($p_{GG} < .001$)，说明压缩比例的效果高度依赖于预训练背景和模型位置的组合。


#### 2. 二阶简单效应分析

由于所有二阶交互作用均显著，我们分别从三个维度进行拆解。

##### 2.1 Pretrain $\times$ Location (固定预训练看位置)

**分析发现**：面孔预训练（Face）对位置极度敏感，而物体预训练（Object）在不同位置间表现一般。并不恒定，但对于多数model适用

`表 2：基于 Pretrain 的位置简单效应比较`

See `7_SuppExp/2_ANOVA_Results/PostHoc_2Way_Pretrain_Location.csv`

| Pretrain | Comparison (Loc) | Difference | Std. Err | $p_{bonf}$ |
| --- | --- | --- | --- | --- |
| **Face** | **L1 vs L3** | 0.0233 | 0.0015 | **<.001** |
| **Face** | **L2 vs L3** | 0.0277 | 0.0051 | **<.001** |
| Object | L1 vs L3 | -0.0127 | 0.0069 | .221 |

##### 2.2 Pretrain $\times$ Ratio (固定预训练看压缩比)


`表 3：基于 Pretrain 的 Ratio 简单效应比较 (以 R16 为基准)`

See `7_SuppExp/2_ANOVA_Results/PostHoc_2Way_Pretrain_Ratio.csv`

| Pretrain | Comparison | Difference | Std. Err | $p_{bonf}$ |
| --- | --- | --- | --- | --- |
| **Face** | **R16 vs R2** | -0.0554 | 0.0084 | **<.001** |
| **Face** | **R16 vs R8** | -0.0947 | 0.0103 | **<.001** |
| **Object** | R16 vs R2 | -0.0590 | 0.0084 | **<.001** |
| **Object** | R16 vs R8 | 0.0313 | 0.0045 | **<.001** |

##### 2.3 Location $\times$ Ratio (固定位置看压缩比)

在 L3 位置，R16 相对于其他 Ratio 展现了显著的类人化优势。

See `7_SuppExp/2_ANOVA_Results/PostHoc_2Way_Location_Ratio.csv`

| Location | Comparison | Difference | Std. Err | $p_{bonf}$ |
| --- | --- | --- | --- | --- |
| **L3** | **R16 vs R2** | -0.0308 | 0.0057 | **<.001** |
| **L3** | **R16 vs R4** | 0.0127 | 0.0035 | **.008** |
| **L3** | **R16 vs R8** | -0.0350 | 0.0039 | **<.001** |


#### 3. 三阶简单简单效应分析 (3-way Simple Simple Effects)

##### 3.1 基于 Pretrain + Location 拆解 Ratio (寻找各配置下的最佳 Ratio)

`Face 组在不同位置下的 Ratio 关键比较`

See `7_SuppExp/2_ANOVA_Results/PostHoc_3Way_Pretrain_Location_Ratio.csv`

| Location | Comparison | Difference | Std. Err | $p_{bonf}$ |
| --- | --- | --- | --- | --- |
| **L3** | **R16 vs R2** | **-0.04396** | **0.01116** | **.003** |
| **L3** | **R16 vs R8** | -0.13913 | 0.01466 | **<.001** |
| **L1** | R16 vs R2 | -0.03013 | 0.01031 | .058 |
| **L1** | R16 vs R8 | -0.13373 | 0.01503 | **<.001** |


##### 3.2 基于 Location + Ratio 拆解 Pretrain (面孔 vs 物体)


`表：相同物理架构下 Pretrain 的直接对比 (Face vs Object)`

| Location | Ratio | Mean Diff (F-O) | Std. Err | $p_{bonf}$ | 结论 |
| --- | --- | --- | --- | --- | --- |
| **L3** | **R16** | **-0.08012** | **0.01315** | **<.001** | **Face 显著优于 Object** |
| **L3** | **R2** | -0.04403 | 0.01235 | **.002** | Face 优于 Object |
| L1 | R16 | -0.00108 | 0.00125 | .393 | 无显著差异 |


---
#### Notes:
1. Face 预训练 + L3 位置 + R16 压缩。该组合下误差最小，且显著优于物体预训练组。
2. 面孔模型的类人化优势在 **L3 (晚期语义位置)** 表现最明显。在 L1 (早期位置) 时，预训练的影响微乎其微 ($p = .393$)。
3. R16 在 Face-L3 架构下表现出了显著优于 R2 ($p=.003$) 和 R8 ($p<.001$) 的类人度，验证了适度信息压缩的必要性。
