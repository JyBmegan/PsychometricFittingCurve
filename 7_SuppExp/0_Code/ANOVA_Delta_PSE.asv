clear; clc; close all;

currentScriptPath = fileparts(mfilename('fullpath'));

dataRoot = fullfile(currentScriptPath, '..', '..', '0_Data'); 
HumanDataRoot = fullfile(currentScriptPath, '..', '..', '5_HumanPSECalculation'); 
savePath = fullfile(currentScriptPath, '..', '2_ANOVA_Results');

if ~exist(savePath, 'dir'), mkdir(savePath); end

humanCsvPath = fullfile(HumanDataRoot, 'Human_PSE_40_Subjects.csv');
if exist(humanCsvPath, 'file')
    humanData = readtable(humanCsvPath);
    humanIndiPSE = humanData.PSE_u; % 40x1
    subjectIDs = humanData.SubjectID;
else
    error('未找到 Human_PSE_40_Subjects.csv，请检查路径！');
end
numParticipants = length(humanIndiPSE);

%% Data Preparation
% RM-ANOVA: 2x3x5
bases = {'FaceBased', 'ObjectBased'}; % 因素 A pretrain权重: 2 水平（face/object based）
locConfigs = [
    struct('ID', 'SE-FC-L1', 'Folder', 'Location-1', 'Prefix', 'Location-1');
    struct('ID', 'SE-FC-L2', 'Folder', 'Location-2', 'Prefix', 'Location-2');
    struct('ID', 'SE-FC-L3', 'Folder', 'Location-3', 'Prefix', 'Location-3');
]; % 因素 B 插入位置: 3 水平（前序 中序 后序）
ratios = [2, 4, 8, 16, 32]; % 因素 C SEmodule压缩比: 5 水平

mask = 'Full'; % 仅比较无遮挡的标准情绪面孔

% 拟合参数（见心理物理曲线拟合code）
targetRowIdx = [1, 5, 7, 9, 11, 13, 15, 17, 21];
paraX = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0];
UL = [0.05, 0.05, 1, 100];
SP = [0.01, 0.02, 0.5, 5.0];
LM = [0, 0, 0, 0.1];


% 第一列  被试 ID
wideFormatData = table(subjectIDs, 'VariableNames', {'SubjectID'});

for b = 1:length(bases)
    currBase = bases{b};
    for l = 1:length(locConfigs)
        currCfg = locConfigs(l);
        for r = 1:length(ratios)
            currR = ratios(r);

            filePattern = sprintf('%s*%s*%s*-%d.csv', currCfg.Prefix, currBase, mask, currR);
            files = dir(fullfile(dataRoot, currCfg.Folder, filePattern));

            if ~isempty(files)
                % 读取当前条件下的模型数据并拟合出该模型唯一的PSE
                yDataRaw = readmatrix(fullfile(files(1).folder, files(1).name));
                ySampled = yDataRaw(targetRowIdx, 2);
                [ffit, ~, ~] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM);
                modelPSE = ffit.u;

                % 计算40个人在这个模型下的绝对偏差 Delta PSE
                % 每一个人的实际PSE减去 模型的拟合PSE，取绝对值
                deltaPSE_array = abs(humanIndiPSE - modelPSE);

                % 列名例如: Face_L3_R16
                colName = sprintf('%s_L%d_R%d', strrep(currBase, 'Based', ''), l, currR);
                wideFormatData.(colName) = deltaPSE_array;
            end
        end
    end
end

% save current data for preparation
writetable(wideFormatData, fullfile(savePath, 'DeltaPSE_ANOVA_Wide.csv'));

%% RM-ANOVA

measNames = wideFormatData.Properties.VariableNames(2:end);

% 命名循环顺序：Base (2) -> Location (3) -> Ratio (5)
Pretrain_Factor = categorical(repelem({'Face', 'Object'}, 15)');
Location_Factor = categorical(repmat(repelem({'L1', 'L2', 'L3'}, 5)', 2, 1));
Ratio_Factor    = categorical(repmat({'R2', 'R4', 'R8', 'R16', 'R32'}', 6, 1));

withinDesign = table(Pretrain_Factor, Location_Factor, Ratio_Factor, ...
    'VariableNames', {'Pretrain', 'Location', 'Ratio'});

% 从第一列到最后一列的 30 个重复测量变量，由 1 (常数/整体截距) 预测
rmFormula = sprintf('%s-%s ~ 1', measNames{1}, measNames{end});
rm = fitrm(wideFormatData, rmFormula, 'WithinDesign', withinDesign);

% Pretrain*Location*Ratio
ranovatbl = ranova(rm, 'WithinModel', 'Pretrain*Location*Ratio');

% Partial Eta Squared
% MATLAB需手动计算，公式: Partial Eta^2 = SS_effect / (SS_effect + SS_error)
% 在 ranovatbl 中，每一个效应(例如 Pretrain)的下一行就是它的误差项(Error(Pretrain))

numRows = height(ranovatbl);
Partial_Eta_Sq = zeros(numRows, 1);

for i = 1:2:(numRows-1) % 步长为2，因为效应和它的Error是成对出现的
    SS_effect = ranovatbl.SumSq(i);
    SS_error  = ranovatbl.SumSq(i+1);
    Partial_Eta_Sq(i) = SS_effect / (SS_effect + SS_error);
end

ranovatbl.Partial_Eta_Sq = Partial_Eta_Sq;

disp('========================================================================');
disp('                    2 x 3 x 5 RM-ANOVA 统计结果');
disp('========================================================================');
disp(ranovatbl(:, {'SumSq', 'DF', 'MeanSq', 'F', 'pValue', 'pValueGG', 'Partial_Eta_Sq'}));

writetable(ranovatbl, fullfile(savePath, 'RM_ANOVA_Results.xlsx'), 'WriteRowNames', true);
