clear; clc; close all;

human_mean = 0.514269;
human_sd   = 0.092421;

dataRoot = '../0_data'; 
savePath = './Z_Score_Analysis'; 
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

for l = 1:length(locConfigs)
    currCfg = locConfigs(l);
    
    % 重新组织数据：所有模型按顺序排列 [VGG-16, AlexNet, SE-2, SE-4, SE-8, SE-16, SE-32]
    model_names = {'VGG-16', 'AlexNet', 'SE-AlexNet-2', 'SE-AlexNet-4', 'SE-AlexNet-8', 'SE-AlexNet-16', 'SE-AlexNet-32'};
    n_models = length(model_names);
    z_vector = nan(1, 2*n_models); % [Object组7个, Face组7个]
    
    for b = 1:length(bases)
        currBase = bases{b};
        base_offset = (b-1)*n_models; % Object: 0, Face:7
        
        % VGG-16
        vFile = dir(fullfile(dataRoot, 'VGG-16', ['VGG-16-' currBase '-' currMask '-raw.csv']));
        if ~isempty(vFile)
            yDataRaw = readmatrix(fullfile(vFile(1).folder, vFile(1).name));
            [ffit_V, ~, ~] = FitPsycheCurveWH(paraX', yDataRaw(targetRowIdx, 2), UL, SP, LM);
            z_vector(base_offset + 1) = (ffit_V.u - human_mean) / human_sd * sqrt(39);
            summaryData = [summaryData; {currCfg.ID, currBase, currMask, 'VGG-16', ffit_V.u, z_vector(base_offset + 1)}];
        end
        
        % AlexNet
        aFile = dir(fullfile(dataRoot, 'AlexNet', ['AlexNet-' currBase '-' currMask '-raw.csv']));
        if ~isempty(aFile)
            yDataRaw = readmatrix(fullfile(aFile(1).folder, aFile(1).name));
            [ffit_A, ~, ~] = FitPsycheCurveWH(paraX', yDataRaw(targetRowIdx, 2), UL, SP, LM);
            z_vector(base_offset + 2) = (ffit_A.u - human_mean) / human_sd * sqrt(39);
            summaryData = [summaryData; {currCfg.ID, currBase, currMask, 'AlexNet', ffit_A.u, z_vector(base_offset + 2)}];
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
                z_vector(base_offset + 2 + r) = (ffit_R.u - human_mean) / human_sd * sqrt(39);
                summaryData = [summaryData; {currCfg.ID, currBase, currMask, sprintf('SE-R%d', currR), ffit_R.u, z_vector(base_offset + 2 + r)}];
            end
        end
    end

    figName = sprintf('ZScore_FullFace_%s', currCfg.ID);
    figure('Color', 'w', 'Name', figName, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7]);
    hold on;
    
    % 分别绘制Object和Face组的柱状图（实现不同线型）
    n_total = length(z_vector);
    n_half = n_total/2;
    
    % Object组：虚线边框，加粗
    bar_object = bar(1:n_half, z_vector(1:n_half), 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 2);
    bar_object.LineStyle = '--'; % 虚线
    bar_object.CData = [color_vgg16; color_alexnet; color_reductions];
    
    % Face组：实线边框，加粗
    bar_face = bar(n_half+1:n_total, z_vector(n_half+1:n_total), 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 2);
    bar_face.LineStyle = '-'; % 实线
    bar_face.CData = [color_vgg16; color_alexnet; color_reductions];
    
    % 绘制Human Baseline线
    yline(0, 'k-', 'LineWidth', 2, 'Label', 'Human Baseline (\mu)', 'LabelHorizontalAlignment', 'left', 'FontSize', 11);
    
    % 填充±1.96之间的全部灰色区域（完全铺满横轴，无空隙）
    % 获取当前坐标轴的x范围，基于此填充
    x_lim = xlim; % 获取横轴实际显示范围
    fill([x_lim(1), x_lim(2), x_lim(2), x_lim(1)], [-1.96, -1.96, 1.96, 1.96], [0.9 0.9 0.9], ...
         'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    uistack(findobj(gca, 'Type', 'patch'), 'bottom'); 
    
    % 修改标注为SEM，调整线型和宽度
    yline(1.96, 'k--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'Label', '+1.96 SEM');
    yline(-1.96, 'k--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'Label', '-1.96 SEM');
    
    ylabel('Z-Score of Point of Subjective Equality (PSE)', 'FontSize', 13, 'FontWeight', 'bold');

    % 设置横轴标签和旋转
    xtick_labels = [model_names, model_names];
    set(gca, 'XTick', 1:length(z_vector), 'XTickLabel', xtick_labels, 'FontSize', 11, 'LineWidth', 1.2);
    xtickangle(45); % 旋转标签避免重叠
    
    % 分组标题移到图表顶部（基于纵轴最大值定位）
    y_lim = ylim;
    y_title_pos = y_lim(2) * 1.05; % 标题位置在纵轴最大值上方5%
    % Object Based Models 标题（居中在左侧7个柱子上方）
    text(3.5, y_title_pos, 'Object Based Models', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
    % Face Based Models 标题（居中在右侧7个柱子上方）
    text(10.5, y_title_pos, 'Face Based Models', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 调整标题位置（避免和分组标题重叠）
    title(sprintf(currCfg.ID), 'FontSize', 16, 'FontWeight', 'bold', 'Position', [7, y_lim(2)*1.12, 0]);
    
    % 重新设置纵轴范围（留出标题空间）
    ylim([y_lim(1), y_lim(2)*1.15]);
    
    grid off; 
    box off;
    hold off;

    exportgraphics(gcf, fullfile(savePath, [figName '.png']), 'Resolution', 300);
    savefig(gcf, fullfile(savePath, [figName '.fig']));
    close(gcf); 
end

resultTable = cell2table(summaryData, 'VariableNames', ...
    {'Location', 'BaseType', 'MaskType', 'ModelVariant', 'Raw_PSE', 'Z_Score'});
writetable(resultTable, fullfile(savePath, 'Z_Score_FullFace_Table.xlsx'));