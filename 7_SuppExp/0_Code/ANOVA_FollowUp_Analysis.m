clear; clc; close all;
currentScriptPath = fileparts(mfilename('fullpath'));
savePath = fullfile(currentScriptPath, '..', '2_ANOVA_Results');
dataFile = fullfile(savePath, 'DeltaPSE_ANOVA_Wide.csv');
if exist(dataFile, 'file')
    wideFormatData = readtable(dataFile);
else
    error('找不到原始数据 %s，请检查路径！', dataFile);
end
measNames = wideFormatData.Properties.VariableNames(2:end);
Pretrain_Factor = categorical(repelem({'Face', 'Object'}, 15)');
Location_Factor = categorical(repmat(repelem({'L1', 'L2', 'L3'}, 5)', 2, 1));
Ratio_Factor    = categorical(repmat({'R2', 'R4', 'R8', 'R16', 'R32'}', 6, 1));
withinDesign = table(Pretrain_Factor, Location_Factor, Ratio_Factor, 'VariableNames', {'Pretrain', 'Location', 'Ratio'});
rmFormula = sprintf('%s-%s ~ 1', measNames{1}, measNames{end});
rm = fitrm(wideFormatData, rmFormula, 'WithinDesign', withinDesign);
%% 1. Main Effect - Ratio
% 根据ANOVA的结果：(Intercept):Ratio pGG < 0.001
post_Main_Ratio = multcompare(rm, 'Ratio', 'ComparisonType', 'bonferroni');
writetable(post_Main_Ratio, fullfile(savePath, 'PostHoc_Main_Ratio.csv'));
%% 2. 2-way Interactions
% 根据ANOVA的结果：Pretrain:Location pGG = 0.010
% 在不同的 Pretrain 下，比较 Location
post_2way_Pre_Loc = multcompare(rm, 'Location', 'By', 'Pretrain', 'ComparisonType', 'bonferroni');
writetable(post_2way_Pre_Loc, fullfile(savePath, 'PostHoc_2Way_Pretrain_Location.csv'));
% 根据ANOVA的结果：Pretrain:Ratio pGG < 0.001
% 在不同的 Pretrain 下，比较 Ratio
post_2way_Pre_Rat = multcompare(rm, 'Ratio', 'By', 'Pretrain', 'ComparisonType', 'bonferroni');
writetable(post_2way_Pre_Rat, fullfile(savePath, 'PostHoc_2Way_Pretrain_Ratio.csv'));
% 根据ANOVA的结果：Location:Ratio pGG < 0.001
% 在不同的 Location 下，比较 Ratio
post_2way_Loc_Rat = multcompare(rm, 'Ratio', 'By', 'Location', 'ComparisonType', 'bonferroni');
writetable(post_2way_Loc_Rat, fullfile(savePath, 'PostHoc_2Way_Location_Ratio.csv'));
%% 3. 3-way Interaction
% 根据ANOVA的结果：Pretrain:Location:Ratio pGG < 0.001
% 固定 Pretrain 和 Location 两个背景变量，去两两比较最核心的参数 Ratio

% --- 修复报错：通过循环 Pretrain 水平来处理多个 By 变量 ---
uPre = {'Face', 'Object'};
post_3way = table();
for i = 1:length(uPre)
    % 提取当前预训练水平下的索引
    idx = (withinDesign.Pretrain == uPre{i});
    % 建立临时子模型
    sub_rm = fitrm(wideFormatData, sprintf('%s-%s ~ 1', measNames{find(idx,1)}, measNames{find(idx,1,'last')}), ...
                   'WithinDesign', withinDesign(idx, {'Location', 'Ratio'}));
    % 执行简单简单效应比较
    tmp = multcompare(sub_rm, 'Ratio', 'By', 'Location', 'ComparisonType', 'bonferroni');
    % 添加标识列
    tmp.Pretrain = repmat(uPre(i), height(tmp), 1);
    post_3way = [post_3way; tmp];
end
writetable(post_3way, fullfile(savePath, 'PostHoc_3Way_Pretrain_Location_Ratio.csv'));

% 反向拆解补充 (用于直接比较 面孔 vs 物体)：
% 固定 Location 和 Ratio，比较 Pretrain (面孔 vs 物体)

% --- 修复报错：通过循环 Location 水平来处理多个 By 变量 ---
uLoc = {'L1', 'L2', 'L3'};
post_3way_Pretrain = table();
for j = 1:length(uLoc)
    idx = (withinDesign.Location == uLoc{j});
    % 修复 horzcat 维度问题：将 find 结果转置为行向量后再拼接
    colIndices = [1, (find(idx)+1)']; 
    % 提取不连续的列
    subData = wideFormatData(:, colIndices);
    subNames = subData.Properties.VariableNames(2:end);
    sub_rm = fitrm(subData, sprintf('%s-%s ~ 1', subNames{1}, subNames{end}), ...
                   'WithinDesign', withinDesign(idx, {'Pretrain', 'Ratio'}));
    % 执行比较
    tmp = multcompare(sub_rm, 'Pretrain', 'By', 'Ratio', 'ComparisonType', 'bonferroni');
    tmp.Location = repmat(uLoc(j), height(tmp), 1);
    post_3way_Pretrain = [post_3way_Pretrain; tmp];
end
writetable(post_3way_Pretrain, fullfile(savePath, 'PostHoc_3Way_Compare_Pretrain.csv'));

disp('Done!');