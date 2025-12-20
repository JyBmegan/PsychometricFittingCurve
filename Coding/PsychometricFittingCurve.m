clear; clc; close all;


dataRoot = '../Data'; 
savePath = '../Results'; 
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
UL = [0.1, 0.05, 1, 100];    
SP = [0.02, 0.02, 0.5, 5.0];  
LM = [0, 0, 0, 0.1];       

% Load Human Data
humanPath = fullfile(dataRoot, 'affect_human.xlsx'); 
exlData = readmatrix(humanPath);
exlData = sortrows(exlData, [2 9]); 

rawRatings = zeros(21, 50);
for index = 1 : 50 
    res_col = exlData((index-1)*42+1 : index*42, 6);
    for i = 1 : 21
        % 每2行对应一个刺激强度
        rawRatings(i,index) = 2 - sum(res_col((i-1)*2+1 : (i-1)*2+2))/2;
    end
end
humanRaw9 = rawRatings(targetRowIdx, :);
humanAve = mean(humanRaw9, 2); 
humanSe  = std(humanRaw9, 0, 2) ./ sqrt(50);
[ffitH, curveH, slopeH] = FitPsycheCurveWH(paraX', humanAve, UL, SP, LM);

% Model
locList = {'Location-1', 'Location-2', 'Location-3'};
bases = {'FaceBased', 'ObjectBased'};
masks = {'E', 'M', 'N', 'Full'};
summaryResults = {}; 

reductionSuffix = {'-2.csv', '-4.csv', '-8.csv', '-16.csv', '-32.csv'};

for l = 1:length(locList)
    currLoc = locList{l};
    
    for b = 1:length(bases)
        for m = 1:length(masks)
            currBase = bases{b};
            currMask = masks{m};
            
            figName = sprintf('%s-%s-%s', currLoc, currBase, currMask);
            figure('Color', 'w', 'Name', figName, 'Units', 'normalized', 'Position', [0.1 0.1 0.5 0.7]); hold on;
            
            % Load Models Data
            for r = 1:5
                suffix = reductionSuffix{r};
                pattern = sprintf('%s-%s-%s%s', currLoc, currBase, currMask, suffix);
                files = dir(fullfile(dataRoot, currLoc, ['*' currBase '*' currMask '*' suffix]));
                if ~isempty(files)
                    yDataRaw = readmatrix(fullfile(files(1).folder, files(1).name));
                    ySampled = yDataRaw(targetRowIdx, 2);

%                     Convert "curveM" into "~" if you want to draw LINE CHART for the Models 
%                     [ffit, ~, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM);

                    [ffit, curveM, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM);
                    
                    c = colors.reductions(r, :);
                    legName = sprintf('SE-AlexNet-R%s', strrep(strrep(suffix, '.csv', ''), '-', ''));
                    
                    % Plot SE-AlexNet
%                     Change parameters for plotting if you want to draw LINE CHART for the Models
%                     plot(paraX, ySampled, '-o', 'Color', c, 'MarkerSize', 5, ...
%                          'MarkerFaceColor', c, 'LineWidth', 2.5, 'DisplayName', legName);
                    plot(curveM(:,1), curveM(:,2), '-', 'Color', c, 'LineWidth', 2.5, ...
                        'HandleVisibility', 'on', 'DisplayName', legName);
                    plot(paraX, ySampled, 'o', 'MarkerEdgeColor', c, 'MarkerFaceColor', c, ...
                        'MarkerSize', 5, 'HandleVisibility', 'off');

                    summaryResults = [summaryResults; {currLoc, currBase, currMask, legName, ffit.u, slope}];
                end
            end
            
            % Plot Raw AlexNet
            aFile = dir(fullfile(dataRoot, 'AlexNet', ['AlexNet-' currBase '-' currMask '-raw.csv']));
            if ~isempty(aFile)
                yDataRaw = readmatrix(fullfile(aFile(1).folder, aFile(1).name));
                ySampled = yDataRaw(targetRowIdx, 2);

%                 Convert "curveR" into "~" if you want to draw LINE CHART for the Models 
%                 [ffit, ~, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM);
                [ffit, curveR, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM);
                
%                  Change parameters for plotting if you want to draw LINE CHART for the Models
%                  plot(paraX, ySampled, '-o', 'Color', colors.alexnet, 'MarkerSize', 5, ...
%                          'MarkerFaceColor', colors.alexnet, 'LineWidth', 2.5, 'DisplayName', 'AlexNet');
                plot(curveR(:,1), curveR(:,2), '-', 'Color', colors.alexnet, 'LineWidth', 2.5, ...
                    'HandleVisibility', 'on', 'DisplayName', 'AlexNet');
                plot(paraX, ySampled, 'o', 'MarkerEdgeColor', colors.alexnet, 'MarkerFaceColor', colors.alexnet, ...
                    'MarkerSize', 5, 'HandleVisibility', 'off');

                summaryResults = [summaryResults; {currLoc, currBase, currMask, 'AlexNet', ffit.u, slope}];
            end
            
            % Plot VGG
            vFile = dir(fullfile(dataRoot, 'VGG-16', ['VGG-16-' currBase '-' currMask '-raw.csv']));
            if ~isempty(vFile)
                yDataRaw = readmatrix(fullfile(vFile(1).folder, vFile(1).name));
                ySampled = yDataRaw(targetRowIdx, 2);

%                 Convert "curveV" into "~" if you want to draw LINE CHART for the Models 
%                 [ffit, ~, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM);
                [ffit, curveV, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM);

%                  Change parameters for plotting if you want to draw LINE CHART for the Models
%                  plot(paraX, ySampled, '-o', 'Color', colors.vgg16, 'MarkerSize', 5, ...
%                          'MarkerFaceColor', colors.vgg16, 'LineWidth', 2.5, 'DisplayName', 'VGG 16');
                plot(curveV(:,1), curveV(:,2), '-', 'Color', colors.vgg16, 'LineWidth', 2.5, ...
                    'HandleVisibility', 'on', 'DisplayName', 'VGG 16');
                plot(paraX, ySampled, 'o', 'MarkerEdgeColor', colors.vgg16, 'MarkerFaceColor', colors.vgg16, ...
                    'MarkerSize', 5, 'HandleVisibility', 'off');

                summaryResults = [summaryResults; {currLoc, currBase, currMask, 'VGG 16', ffit.u, slope}];
            end

            % Plot Human Data
            plot(curveH(:,1), curveH(:,2), 'k-', 'LineWidth', 2.5, 'DisplayName', 'Human Data');
            errorbar(paraX, humanAve, humanSe, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
                     'LineWidth', 2, 'HandleVisibility', 'off');
            
            % General Plot Setting
            title(sprintf('%s: %s - %s', currLoc, currBase, currMask), 'FontSize', 12, 'FontWeight', 'bold');
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

% Save Models' Data
resultTable = cell2table(summaryResults, 'VariableNames', ...
    {'Location', 'BaseType', 'MaskType', 'ModelVariant', 'PSE', 'Slope'});
writetable(resultTable, fullfile(savePath, 'Fitting_Results.xlsx'));

% Save Humans' Data
humanParamsTable = table({'Human'}, ffitH.g, ffitH.l, ffitH.u, ffitH.v, slopeH, ...
    'VariableNames', {'Model', 'GuessRate_g', 'LapseRate_l', 'PSE_u', 'Sigma_v', 'Slope'});
writetable(humanParamsTable, fullfile(savePath, 'Human_Fitting_Parameters.xlsx'));
