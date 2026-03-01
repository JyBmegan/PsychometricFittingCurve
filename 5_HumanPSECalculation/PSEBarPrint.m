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
    z_matrix = nan(2, 7); 
    
    for b = 1:length(bases)
        currBase = bases{b};
        
        vFile = dir(fullfile(dataRoot, 'VGG-16', ['VGG-16-' currBase '-' currMask '-raw.csv']));
        if ~isempty(vFile)
            yDataRaw = readmatrix(fullfile(vFile(1).folder, vFile(1).name));
            [ffit_V, ~, ~] = FitPsycheCurveWH(paraX', yDataRaw(targetRowIdx, 2), UL, SP, LM);
            z_matrix(b, 1) = (ffit_V.u - human_mean) / human_sd;
            summaryData = [summaryData; {currCfg.ID, currBase, currMask, 'VGG-16', ffit_V.u, z_matrix(b, 1)}];
        end
        
        % 2. 获取 AlexNet
        aFile = dir(fullfile(dataRoot, 'AlexNet', ['AlexNet-' currBase '-' currMask '-raw.csv']));
        if ~isempty(aFile)
            yDataRaw = readmatrix(fullfile(aFile(1).folder, aFile(1).name));
            [ffit_A, ~, ~] = FitPsycheCurveWH(paraX', yDataRaw(targetRowIdx, 2), UL, SP, LM);
            z_matrix(b, 2) = (ffit_A.u - human_mean) / human_sd;
            summaryData = [summaryData; {currCfg.ID, currBase, currMask, 'AlexNet', ffit_A.u, z_matrix(b, 2)}];
        end
        
        % 3. 获取 当前 SE-Block 的 5 个 Reduction
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
                z_matrix(b, r+2) = (ffit_R.u - human_mean) / human_sd;
                summaryData = [summaryData; {currCfg.ID, currBase, currMask, sprintf('SE-R%d', currR), ffit_R.u, z_matrix(b, r+2)}];
            end
        end
    end

    figName = sprintf('ZScore_FullFace_%s', currCfg.ID);
    figure('Color', 'w', 'Name', figName, 'Units', 'normalized', 'Position', [0.1 0.1 0.6 0.65]);
    hold on;
    
    b_plot = bar(z_matrix, 'grouped', 'EdgeColor', 'none');
    
    % 上色
    b_plot(1).FaceColor = color_vgg16;   
    b_plot(2).FaceColor = color_alexnet; 
    for r = 1:5
        b_plot(r+2).FaceColor = color_reductions(r, :); 
    end
    
    yline(0, 'k-', 'LineWidth', 2, 'Label', 'Human Baseline (\mu)', 'LabelHorizontalAlignment', 'left', 'FontSize', 11);
    
    fill([0.5, 2.5, 2.5, 0.5], [-1.96, -1.96, 1.96, 1.96], [0.9 0.9 0.9], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    uistack(findobj(gca, 'Type', 'patch'), 'bottom'); 
    
    yline(1.96, 'k--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'Label', '+1.96\sigma');
    yline(-1.96, 'k--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'Label', '-1.96\sigma');
    
    ylabel('Z-Score of Point of Subjective Equality (PSE)', 'FontSize', 13, 'FontWeight', 'bold');

    set(gca, 'XTick', 1:2, 'XTickLabel', {'Object-based Transfer', 'Face-based Transfer'}, 'FontSize', 13, 'LineWidth', 1.2);
    
    title(sprintf(currCfg.ID), 'FontSize', 16, 'FontWeight', 'bold');
    
    max_z = max(abs(z_matrix(:)));
    if ~isnan(max_z)
        y_limit = max(2.5, max_z + 0.5);
        ylim([-y_limit, y_limit]);
    end
    
    legend_labels = {'VGG-16', 'AlexNet', 'SE-AlexNet-2', 'SE-AlexNet-4', 'SE-AlexNet-8', 'SE-AlexNet-16', 'SE-AlexNet-32'};
    lgd = legend(b_plot, legend_labels, 'Location', 'northeast', 'NumColumns', 1, 'FontSize', 11);
     
    set(lgd, 'box', 'on');
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
