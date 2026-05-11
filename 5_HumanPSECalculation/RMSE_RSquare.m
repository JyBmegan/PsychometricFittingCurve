clear; clc; close all;

% If you are replicating, you don't have to do so
rawSourceDir = '../2_ConvResults';
convDestDir  = '../0_Data/Conv';
if ~exist(convDestDir, 'dir'), mkdir(convDestDir); end

if exist(rawSourceDir, 'dir')
    subFolders = dir(rawSourceDir);
    subFolders = subFolders([subFolders.isdir] & ~startsWith({subFolders.name}, '.'));
    for i = 1:length(subFolders)
        predictPath = fullfile(rawSourceDir, subFolders(i).name, 'predict');
        if exist(predictPath, 'dir')
            csvFiles = dir(fullfile(predictPath, '*.csv'));
            for k = 1:length(csvFiles)
                sourceFile = fullfile(predictPath, csvFiles(k).name);
                destFile   = fullfile(convDestDir, csvFiles(k).name);
                copyfile(sourceFile, destFile);
            end
        end
    end
end

dataRoot = '../0_Data'; 
savePath = '..Results'; 
if ~exist(savePath, 'dir'), mkdir(savePath); end

% Select 0,4,6,8,10,12,14,16,20
targetRowIdx = [1, 5, 7, 9, 11, 13, 15, 17, 21]; 
paraX = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0];

% Color
colors.human = [0 0 0];
colors.vgg16 = [58 191 153]/255;
colors.alexnet = [145 205 200]/255;
colors.reductions = [
    111 185 208; % R2
    84 153 189;  % R4
    57 129 175;  % R8
    56 97 149;   % R16
    50 76 99     % R32
]/255;

% Para Limitation
UL = [0.05, 0.05, 1, 100];    
SP = [0.01, 0.02, 0.5, 5.0];  
LM = [0, 0, 0, 0.1];       

% Load Human Data
humanPath = fullfile(dataRoot, 'affect_human.xlsx'); 
exlData = readmatrix(humanPath);
exlData = sortrows(exlData, [2 9]); 
numParticipants = 50;
humanIndiPSE = zeros(numParticipants, 1);
allHumanRaw = zeros(21, numParticipants);

for index = 1 : numParticipants
    res_col = exlData((index-1)*42+1 : index*42, 6);
    for i = 1 : 21
        % 每2行对应一个刺激强度
        allHumanRaw(i,index) = 2 - sum(res_col((i-1)*2+1 : (i-1)*2+2))/2;
    end
    % Fit for each Participant (Preparation for t test)
    pData = allHumanRaw(targetRowIdx, index);
    [fitP, ~, ~] = FitPsycheCurveWH(paraX', pData, UL, SP, LM);
    humanIndiPSE(index) = fitP.u;
end

humanMeanPSE = mean(humanIndiPSE);
humanSDPSE = std(humanIndiPSE);

humanRaw9 = allHumanRaw(targetRowIdx, :);
humanAve = mean(humanRaw9, 2); 
humanSe  = std(humanRaw9, 0, 2) ./ sqrt(50);
[ffitH, curveH, slopeH] = FitPsycheCurveWH(paraX', humanAve, UL, SP, LM);

% Model
locConfigs = [
    % --- 原来的 FC 层位置 (Location 1-3) ---
    struct('ID', 'SE-FC-L1', 'Folder', 'Location-1', 'Prefix', 'Location-1', 'Type', 'Old_Hyphen');
    struct('ID', 'SE-FC-L2', 'Folder', 'Location-2', 'Prefix', 'Location-2', 'Type', 'Old_Hyphen');
    struct('ID', 'SE-FC-L3', 'Folder', 'Location-3', 'Prefix', 'Location-3', 'Type', 'Old_Hyphen');
    
    % --- 新的 Conv 层位置 (SeC 1-4) ---
    struct('ID', 'SE-Conv-L1', 'Folder', 'Conv', 'Prefix', 'SeC1', 'Type', 'New_Underscore');
    struct('ID', 'SE-Conv-L2', 'Folder', 'Conv', 'Prefix', 'SeC2', 'Type', 'New_Underscore');
    struct('ID', 'SE-Conv-L3', 'Folder', 'Conv', 'Prefix', 'SeC3', 'Type', 'New_Underscore');
    struct('ID', 'SE-Conv-L4', 'Folder', 'Conv', 'Prefix', 'SeC4', 'Type', 'New_Underscore');
];

bases = {'FaceBased', 'ObjectBased'};
masks = {'E', 'M', 'N', 'Full'};
summaryResults = {}; 

for l = 1:length(locConfigs)
    currCfg = locConfigs(l);
    currID = currCfg.ID; 
    currFolder = currCfg.Folder; 
    currPrefix = currCfg.Prefix; 
    
    for b = 1:length(bases)
        for m = 1:length(masks)
            currBase = bases{b};
            currMask = masks{m};
            
            figName = sprintf('%s_%s_%s', currID, currBase, currMask);
            figure('Color', 'w', 'Name', figName, 'Units', 'normalized', 'Position', [0.1 0.1 0.5 0.7]); hold on;
            
            % Load Models Data
            modelConfigs = {};
            for r = 1:5
                ratioVal = [2, 4, 8, 16, 32];
                currR = ratioVal(r);
                if strcmp(currCfg.Type, 'Old_Hyphen')
                    filePattern = sprintf('%s*%s*%s*-%d.csv', currPrefix, currBase, currMask, currR);                    
                elseif strcmp(currCfg.Type, 'New_Underscore')
                    filePattern = sprintf('%s_%s_squeeze%d_%s.csv', currPrefix, currBase, currR, currMask);
                end
                
                modelConfigs{end+1} = struct(...
                    'folder', currFolder, ...
                    'pattern', filePattern, ...
                    'name', sprintf('%s-R%d', currID, currR), ... 
                    'color', colors.reductions(r,:));
            end
            modelConfigs{end+1} = struct('folder', 'AlexNet', 'pattern', ['AlexNet-' currBase '-' currMask '-raw.csv'], 'name', 'AlexNet', 'color', colors.alexnet);
            modelConfigs{end+1} = struct('folder', 'VGG-16', 'pattern', ['VGG-16-' currBase '-' currMask '-raw.csv'], 'name', 'VGG 16', 'color', colors.vgg16);
            
            for idx = 1:length(modelConfigs)
                cfg = modelConfigs{idx};
                files = dir(fullfile(dataRoot, cfg.folder, cfg.pattern));
                
                if ~isempty(files)
                    yDataRaw = readmatrix(fullfile(files(1).folder, files(1).name));
                    ySampled = yDataRaw(targetRowIdx, 2); % 真实的测定点数据
                    
                    [ffit, curveM, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM);
                    
                    % T-test & Cohen's d
                    modelPSE = ffit.u;
                    [~, p_val, ~, stats] = ttest(humanIndiPSE, modelPSE);
                    t_stat = abs(stats.tstat);
                    cohen_d = abs(modelPSE - humanMeanPSE) / humanSDPSE;
                    
                    % ========================================================
                    % --- 新增：计算实际点与拟合曲线的差异 (Goodness of Fit) ---
                    
                    % 1. 利用一维插值法(interp1)，从高分辨率的拟合曲线 (curveM) 
                    % 中提取出实验 X 坐标对应的“理论预测 Y 值”
                    yPredicted = interp1(curveM(:,1), curveM(:,2), paraX', 'linear', 'extrap');
                    
                    % 2. 计算残差 (真实点 和 拟合曲线上的点 的差距)
                    residuals = ySampled - yPredicted;
                    
                    % 3. 计算均方根误差 (RMSE)
                    RMSE = sqrt(mean(residuals.^2));
                    
                    % 4. 计算 R-squared (R方)
                    SST = sum((ySampled - mean(ySampled)).^2);
                    SSE = sum(residuals.^2);
                    if SST ~= 0
                        R_Square = 1 - (SSE / SST);
                    else
                        R_Square = NaN; % 防止数学上分母为0
                    end
                    % ========================================================

                    % 绘图逻辑保持不变
                    plot(curveM(:,1), curveM(:,2), '-', 'Color', cfg.color, 'LineWidth', 2.5, ...
                        'HandleVisibility', 'on', 'DisplayName', cfg.name);
                    plot(paraX, ySampled, 'o', 'MarkerEdgeColor', cfg.color, 'MarkerFaceColor', cfg.color, ...
                        'MarkerSize', 5, 'HandleVisibility', 'off');
                    
                    % --- 更新输出表：将 RMSE 和 R_Square 一起记录下来 ---
                    summaryResults = [summaryResults; {currID, currBase, currMask, cfg.name, modelPSE, slope, t_stat, p_val, cohen_d, RMSE, R_Square}];
                end
            end
            
            % Plot Human Data
            plot(curveH(:,1), curveH(:,2), 'k-', 'LineWidth', 2.5, 'DisplayName', 'Human Data');
            errorbar(paraX, humanAve, humanSe, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
                     'LineWidth', 2, 'HandleVisibility', 'off');
            
            % General Plot Setting
            title(sprintf('%s: %s - %s', currID, currBase, currMask), 'FontSize', 12, 'FontWeight', 'bold');
            xlabel('Proportion of "Happy-ness" in Face (The Physical Features in Face)'); 
            ylabel('Proportion of "Happy" Response');
            axis([0 1 0 1]); 
            axis square;
            set(gca, 'XTick', 0:0.1:1, 'LineWidth', 1.2);
            legend('Location', 'southeast', 'FontSize', 8, 'NumColumns', 1);
            
            % Save Figure
            savefig(gcf, fullfile(savePath, [figName '.fig']));
            exportgraphics(gcf, fullfile(savePath, [figName '.png']), 'Resolution', 300);
            close(gcf); 
        end
    end
end

% --- 新增：更新导出的 Excel 表头，包含 RMSE 和 R_Square ---
resultTable = cell2table(summaryResults, 'VariableNames', ...
    {'Location', 'BaseType', 'MaskType', 'ModelVariant', 'PSE', 'Slope', 't_stat', 'p_value', 'Cohens_d', 'RMSE', 'R_Square'});
writetable(resultTable, fullfile(savePath, 'Final_Analysis_Results.xlsx'));

fprintf('✅ 所有分析计算完成！包含 RMSE 和 R-Squared 的结果已保存至 Final_Analysis_Results.xlsx。\n');