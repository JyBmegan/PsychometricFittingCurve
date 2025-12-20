# Fitting Log for Psychometric Curve in SE-AlexNet Analysis

## 1. Preparation

### 1.1 File Structure

**Folder - FitCurve:**

│  log.md
│
├─Coding
│      FitPsycheCurveWH.m
│      PsychometricFittingCurve.m
│
├─Data
│  │  affect_human.xlsx
│  │  human-data-9point.csv
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
├─Results
│      Fitting_Results.xlsx
│      Human_Fitting_Parameters.xlsx
│      Location-1-FaceBased-E.fig
│      Location-1-FaceBased-E.png
│      ...
├─Results_LineChart
│      Fitting_Results.xlsx
│      Human_Fitting_Parameters.xlsx
│      Location-1-FaceBased-E.fig
│      Location-1-FaceBased-E.png
│      ...

### 1.2 Condition


**Model Type**: AlexNet, VGG-16, SE-Location-1, SE-Location-2, SE-Location-3
**Base**: FaceBased, ObjectBased
**Masked Area**: Eyes, Mouth, Nose, Full (no mask)
**Reduction (For SE-AlexNet Only)**: 2, 4, 8, 16, 32 
**Baseline**: Human (40 Participants)

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

**Output** - Originally Saved in Folder `Results`

## 4. Notes

### 4.1 About Constraints

无论是行为学实验还是model的任务，都要求被试/model在“快乐”与“悲伤”中做出分类判断，因此本实验属于 **2AFC Discrimination** Task （2AFC辨别任务）；此外，y轴被定义为Proportion of Happy Response。

因此对于constraints, 本研究使用：

* *GuessRate* （$g$）[0, 0.1]：极端非快乐点误按“快乐”的概率，不设 0.5 是因为这是辨别任务而非正确率检测。

* *LapseRate* （$l$） [0, 0.05]：极端快乐点因走神导致的漏报概率。上限设为心理物理学标准 0.05。

* *PSE* （$u$） [0, 1]：50%响应时的强度（分类边界）。预期值（humandata）约 0.51。

* *Slope* [0.1, 100]：知觉切换的敏锐度，允许极高的分类精度。预期值（humandata）约 4.19。

In Code ~Line 24~27：

``` Matlab
% Para Limitation
UL = [0.1, 0.05, 1, 100]; 
SP = [0.02, 0.02, 0.5, 5.0];  
LM = [0, 0, 0, 0.1]; 
```

注：其他任务
* Yes/No 任务：$g \approx 0$，反映信号检测阈限（False Alarm）；
* 2AFC 检测：$g = 0.5$，曲线从半山腰开始，反映猜测正确率。

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
