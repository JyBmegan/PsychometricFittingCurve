clear; clc; close all;

% ================= 1. 基础基准数据 =================
human_mean = 0.514269;
human_sd   = 0.092421;
N_human    = 40; 
df_human   = N_human - 1; 
human_se   = human_sd / sqrt(N_human); % 严谨的标准误计算 (SE = SD / sqrt(N))

% ================= 2. 路径与配置 =================
dataRoot = '../0_data'; 
savePath = './T_Test_Analysis_Plots'; 
if ~exist(savePath, 'dir'), mkdir(savePath); end

targetRowIdx = [1, 5, 7, 9, 11, 13, 15, 17, 21]; 
paraX = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0];
UL = [0.05, 0.05, 1, 100];    
SP = [0.01, 0.02, 0.5, 5.0];  
LM = [0, 0, 0, 0.1];  

color_vgg16   = [58  191 153] / 255;
color_alexnet = [145 205 200] / 255;
color_reductions = [
    111 185 208; % R2
    84  153 189; % R4
    57  129 175; % R8
    56  97  149; % R16
    50  76  99   % R32
] / 255;

locConfigs = [
    struct('ID', 'SE-Conv-L1', 'Folder', 'SeC1',       'Prefix', 'SeC1',       'Type', 'New_Underscore');
    struct('ID', 'SE-Conv-L2', 'Folder', 'SeC2',       'Prefix', 'SeC2',       'Type', 'New_Underscore');
    struct('ID', 'SE-Conv-L3', 'Folder', 'SeC3',       'Prefix', 'SeC3',       'Type', 'New_Underscore');
    struct('ID', 'SE-Conv-L4', 'Folder', 'SeC4',       'Prefix', 'SeC4',       'Type', 'New_Underscore');
    struct('ID', 'SE-FC-L1',   'Folder', 'Location-1', 'Prefix', 'Location-1', 'Type', 'Old_Hyphen');
    struct('ID', 'SE-FC-L2',   'Folder', 'Location-2', 'Prefix', 'Location-2', 'Type', 'Old_Hyphen');
    struct('ID', 'SE-FC-L3',   'Folder', 'Location-3', 'Prefix', 'Location-3', 'Type', 'Old_Hyphen');
];

bases = {'ObjectBased', 'FaceBased'}; 
currMask = 'Full'; 
reductions = [2, 4, 8, 16, 32];
summaryData = {}; 

% ================= 3. 开始遍历与统计分析 =================
for l = 1:length(locConfigs)
    currCfg = locConfigs(l);
    
    model_names = {'VGG-16', 'AlexNet', 'SE-AlexNet-2', 'SE-AlexNet-4', 'SE-AlexNet-8', 'SE-AlexNet-16', 'SE-AlexNet-32'};
    n_models = length(model_names);
    t_vector = nan(1, 2*n_models); % 用 t_vector 替代 z_vector
    
    for b = 1:length(bases)
        currBase = bases{b};
        base_offset = (b-1)*n_models; 
        
        % VGG-16
        vFile = dir(fullfile(dataRoot, 'VGG-16', ['VGG-16-' currBase '-' currMask '-raw.csv']));
        if ~isempty(vFile)
            yDataRaw = readmatrix(fullfile(vFile(1).folder, vFile(1).name));
            [ffit_V, ~, ~] = FitPsycheCurveWH(paraX', yDataRaw(targetRowIdx, 2), UL, SP, LM);
            
            t_val = (ffit_V.u - human_mean) / human_se;
            p_val = 2 * (1 - tcdf(abs(t_val), df_human));
            cohen_d = abs(ffit_V.u - human_mean) / human_sd;
            
            t_vector(base_offset + 1) = t_val;
            summaryData = [summaryData; {currCfg.ID, currBase, currMask, 'VGG-16', ffit_V.u, t_val, df_human, p_val, cohen_d}];
        end
        
        % AlexNet
        aFile = dir(fullfile(dataRoot, 'AlexNet', ['AlexNet-' currBase '-' currMask '-raw.csv']));
        if ~isempty(aFile)
            yDataRaw = readmatrix(fullfile(aFile(1).folder, aFile(1).name));
            [ffit_A, ~, ~] = FitPsycheCurveWH(paraX', yDataRaw(targetRowIdx, 2), UL, SP, LM);
            
            t_val = (ffit_A.u - human_mean) / human_se;
            p_val = 2 * (1 - tcdf(abs(t_val), df_human));
            cohen_d = abs(ffit_A.u - human_mean) / human_sd;
            
            t_vector(base_offset + 2) = t_val;
            summaryData = [summaryData; {currCfg.ID, currBase, currMask, 'AlexNet', ffit_A.u, t_val, df_human, p_val, cohen_d}];
        end
        
        % SE-Block
        for r = 1:5
            currR = reductions(r);
            if strcmp(currCfg.Type, 'Old_Hyphen')
                filePattern = sprintf('%s*%s*%s*-%d.csv', currCfg.Prefix, currBase, currMask, currR);                    
            elseif strcmp(currCfg.Type, 'New_Underscore')
                filePattern = sprintf('%s_%s_squeeze%d_%s.csv', currCfg.Prefix, currBase, currR, currMask);
            end
            
            rFile = dir(fullfile(dataRoot, currCfg.Folder, filePattern));
            if ~isempty(rFile)
                yDataRaw = readmatrix(fullfile(rFile(1).folder, rFile(1).name));
                [ffit_R, ~, ~] = FitPsycheCurveWH(paraX', yDataRaw(targetRowIdx, 2), UL, SP, LM);
                
                t_val = (ffit_R.u - human_mean) / human_se;
                p_val = 2 * (1 - tcdf(abs(t_val), df_human));
                cohen_d = abs(ffit_R.u - human_mean) / human_sd;
                
                t_vector(base_offset + 2 + r) = t_val;
                summaryData = [summaryData; {currCfg.ID, currBase, currMask, sprintf('SE-R%d', currR), ffit_R.u, t_val, df_human, p_val, cohen_d}];
            end
        end
    end
    
    % ================= 4. 画图 (高度还原你的精美排版) =================
%     figName = sprintf('T_Value_FullFace_%s', currCfg.ID);
%     figure('Color', 'w', 'Name', figName, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7]);
%     hold on;
%     
%     n_total = length(t_vector);
%     n_half = n_total/2;
%     
%     % Object组：虚线边框
%     bar_object = bar(1:n_half, t_vector(1:n_half), 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 2);
%     bar_object.LineStyle = '--'; 
%     bar_object.CData = [color_vgg16; color_alexnet; color_reductions];
%     
%     % Face组：实线边框
%     bar_face = bar(n_half+1:n_total, t_vector(n_half+1:n_total), 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 2);
%     bar_face.LineStyle = '-'; 
%     bar_face.CData = [color_vgg16; color_alexnet; color_reductions];
%     
%     % 基线
%     yline(0, 'k-', 'LineWidth', 2, 'Label', 'Human Baseline (\mu)', 'LabelHorizontalAlignment', 'left', 'FontSize', 11);
%     
%     % 填充 95% 置信区间 (自由度39的 t 检验临界值为 ±2.02)
%     t_crit = 2.02; 
%     x_lim = xlim; 
%     fill([x_lim(1), x_lim(2), x_lim(2), x_lim(1)], [-t_crit, -t_crit, t_crit, t_crit], [0.9 0.9 0.9], ...
%          'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
%     uistack(findobj(gca, 'Type', 'patch'), 'bottom'); 
%     
%     yline(t_crit, 'k--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'Label', '+2.02 (p=0.05)');
%     yline(-t_crit, 'k--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'Label', '-2.02 (p=0.05)');
%     
%     ylabel('t-Value (Deviation from Human PSE)', 'FontSize', 13, 'FontWeight', 'bold');
%     
%     xtick_labels = [model_names, model_names];
%     set(gca, 'XTick', 1:length(t_vector), 'XTickLabel', xtick_labels, 'FontSize', 11, 'LineWidth', 1.2);
%     xtickangle(45); 
%     
%     y_lim = ylim;
%     % 确保 Y 轴范围至少包住 t_crit 区域，防止图形挤压
%     ylim([-max(abs(y_lim)), max(abs(y_lim)) * 1.15]); 
%     y_lim = ylim;
%     
%     y_title_pos = y_lim(2) * 0.92; 
%     text(4, y_title_pos, 'Object Based Models', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
%     text(11, y_title_pos, 'Face Based Models', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
%     
%     title(sprintf(currCfg.ID), 'FontSize', 16, 'FontWeight', 'bold', 'Position', [7.5, y_lim(2)*1.02, 0]);
%     
%     grid off; 
%     box off;
%     hold off;
%     
%     exportgraphics(gcf, fullfile(savePath, [figName '.png']), 'Resolution', 300);
%     savefig(gcf, fullfile(savePath, [figName '.fig']));
%     close(gcf); 
end

% ================= 5. 保存完美统计报表 =================
resultTable = cell2table(summaryData, 'VariableNames', ...
    {'Location', 'BaseType', 'MaskType', 'ModelVariant', 'Raw_PSE', 't_Value', 'df', 'p_Value', 'Cohens_d'});
writetable(resultTable, fullfile(savePath, 'T_Test_FullFace_Table.xlsx'));

fprintf('✅ T检验图表与数据表已生成完毕！保存在 %s 文件夹中。\n', savePath);