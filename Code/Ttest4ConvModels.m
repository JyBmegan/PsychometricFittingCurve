clear; clc; close all;

%% ================= 1. 路径与基础配置 =================
% 我们刚刚计算出的 40 个人类独立 PSE 数据文件
humanPSEFile = 'Human_PSE_40_Subjects.csv'; 

% 模型的原始预测数据所在的根目录
dataRoot = '../0_data'; 

% 分析结果保存路径
savePath = '../Results/Statistical_Analysis';
if ~exist(savePath, 'dir'), mkdir(savePath); end
outputFile = fullfile(savePath, 'Comprehensive_T_Test_Results.xlsx');

% 所有需要分析的模型文件夹列表
foldersToAnalyze = {
    'AlexNet', 'VGG-16', ...
    'Location-1', 'Location-2', 'Location-3', ...
    'SeC1', 'SeC2', 'SeC3', 'SeC4'
};

% 拟合参数（与之前完全保持一致，保证 PSE 的绝对严谨）
targetRowIdx = [1, 5, 7, 9, 11, 13, 15, 17, 21]; 
paraX = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0];
UL = [0.05, 0.05, 1, 100];    
SP = [0.01, 0.02, 0.5, 5.0];  
LM = [0, 0, 0, 0.1];  

%% ================= 2. 读取人类基准数据 =================
if ~exist(humanPSEFile, 'file')
    error('未找到 %s 文件！请先运行计算人类 PSE 的代码。', humanPSEFile);
end

humanTable = readtable(humanPSEFile);
humanData = humanTable.PSE_u; % 提取那 40 个真实人类的 PSE 值
humanMean = mean(humanData);
humanSD   = std(humanData);

fprintf('\n✅ 成功加载人类基准数据：N = %d, Mean = %.6f, SD = %.6f\n', ...
    length(humanData), humanMean, humanSD);

%% ================= 3. 遍历模型并执行 T 检验 =================
% 初始化结果 Cell 数组
summaryResults = {};

fprintf('\n⏳ 开始进行曲线拟合与 T 检验统计...\n');

for f = 1:length(foldersToAnalyze)
    currFolder = foldersToAnalyze{f};
    folderPath = fullfile(dataRoot, currFolder);
    
    if ~exist(folderPath, 'dir')
        warning('未找到文件夹 %s，跳过...', folderPath);
        continue;
    end
    
    % 获取该文件夹下所有的 CSV 文件
    csvFiles = dir(fullfile(folderPath, '*.csv'));
    
    for i = 1:length(csvFiles)
        fileName = csvFiles(i).name;
        filePath = fullfile(folderPath, fileName);
        
        % 读取 CSV 数据
        data = readmatrix(filePath);
        
        try
            % ❌ 严禁使用 mean(data(:,2))
            % ✅ 正确做法：必须通过标准的心理物理曲线拟合提取模型的真实 PSE
            ySampled = data(targetRowIdx, 2);
            [ffit, ~, ~] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM);
            Model_PSE = ffit.u;
            
            % --- 核心统计：单样本 T 检验 ---
            % 检验假设：人类群体的均值 是否等于 这个模型的 PSE？
            [~, p_val, ~, stats] = ttest(humanData, Model_PSE);
            
            % 提取 t 值与自由度 (df)
            t_val = abs(stats.tstat);
            df = stats.df;
            
            % --- 核心统计：Cohen's d 效应量 ---
            % 衡量差异的绝对强度：(人类均值 - 模型PSE) / 人类标准差
            cohens_d = abs(humanMean - Model_PSE) / humanSD;
            
            % 保存这条结果
            summaryResults(end+1, :) = {
                currFolder, fileName, Model_PSE, t_val, df, p_val, cohens_d
            };
            
        catch ME
            warning('文件 %s 拟合或计算失败: %s', fileName, ME.message);
        end
    end
end

%% ================= 4. 保存为 Excel 报表 =================
% 转换为 Table 以便直观保存
resultTable = cell2table(summaryResults, 'VariableNames', ...
    {'Model_Folder', 'FileName', 'Model_PSE', 't_Value', 'df', 'p_Value', 'Cohens_d'});

writetable(resultTable, outputFile);

fprintf('\n==================================================\n');
fprintf('🎉 统计分析全部完成！\n');
fprintf('📁 共处理了 %d 个模型条件。\n', size(resultTable, 1));
fprintf('📊 结果已清晰保存至: %s\n', outputFile);
fprintf('==================================================\n');