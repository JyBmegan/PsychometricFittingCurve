clear; clc; close all;

currentScriptPath = fileparts(mfilename('fullpath'));
savePath = fullfile(currentScriptPath, '..', '2_ANOVA_Results');
dataFile = fullfile(savePath, 'DeltaPSE_ANOVA_Wide.csv');

if ~exist(dataFile, 'file')
    error('Data file not found: %s. Run ANOVA_Delta_PSE.m first.', dataFile);
end

wideFormatData = readtable(dataFile);
measNames = wideFormatData.Properties.VariableNames(2:end);
nSubj = height(wideFormatData);

% Recreate within-subject design (must match ANOVA_Delta_PSE.m)
Pretrain_Factor = categorical(repelem({'Face', 'Object'}, 15)');
Location_Factor = categorical(repmat(repelem({'L1', 'L2', 'L3'}, 5)', 2, 1));
Ratio_Factor    = categorical(repmat({'R2', 'R4', 'R8', 'R16', 'R32'}', 6, 1));
withinDesign = table(Pretrain_Factor, Location_Factor, Ratio_Factor, ...
    'VariableNames', {'Pretrain', 'Location', 'Ratio'});

%% ===== Section 1: Descriptive Statistics (M, SD per condition) =====
fprintf('========== 1. Descriptive Statistics (M, SD) ==========\n');

dataMatrix = wideFormatData{:, 2:end};
condMeans = mean(dataMatrix, 1)';
condSDs   = std(dataMatrix, 0, 1)';
descStats = table(measNames', condMeans, condSDs, ...
    'VariableNames', {'Condition', 'M', 'SD'});

% Also compute M and SD grouped by meaningful factors for reporting
uPre = {'Face', 'Object'};  uLoc = {'L1', 'L2', 'L3'};
uRat = [2, 4, 8, 16, 32];   uRatLabel = {'R2', 'R4', 'R8', 'R16', 'R32'};

% Print a structured summary
fprintf('%-20s %-10s %-10s\n', 'Condition', 'M', 'SD');
fprintf('%-20s %-10s %-10s\n', '--------------------', '----------', '----------');
for i = 1:length(measNames)
    fprintf('%-20s %-10.4f %-10.4f\n', measNames{i}, condMeans(i), condSDs(i));
end

writetable(descStats, fullfile(savePath, 'Descriptive_Statistics_new.csv'));
fprintf('\nSaved: Descriptive_Statistics_new.csv (%d conditions)\n', length(measNames));

%% ===== Section 2: Simple Two-Way Interaction Tests =====
fprintf('\n========== 2. Simple Two-Way Interaction Tests ==========\n');
fprintf('  (The missing middle layer: fix one factor, test the other two)\n\n');

% --- 2a. Fix Location -> test Pretrain x Ratio simple 2-way interaction ---
fprintf('--- Simple Pretrain x Ratio at each Location ---\n');
rows_byLoc = {};  % cell array to collect rows, avoids table-concat issues
for i = 1:length(uLoc)
    idx = withinDesign.Location == uLoc{i};
    colIdx = [1, find(idx)' + 1];
    subData = wideFormatData(:, colIdx);
    subNames = subData.Properties.VariableNames(2:end);
    subDesign = withinDesign(idx, {'Pretrain', 'Ratio'});
    sub_rm = fitrm(subData, sprintf('%s-%s ~ 1', subNames{1}, subNames{end}), ...
                   'WithinDesign', subDesign);
    tbl = ranova(sub_rm, 'WithinModel', 'Pretrain*Ratio');
    effectNames = tbl.Properties.RowNames;  % save before any modification

    % Partial eta squared
    numRows = height(tbl);
    pes = zeros(numRows, 1);
    for r = 1:2:(numRows-1)
        if tbl.SumSq(r) + tbl.SumSq(r+1) > 0
            pes(r) = tbl.SumSq(r) / (tbl.SumSq(r) + tbl.SumSq(r+1));
        end
    end

    % Print the critical row: Pretrain:Ratio interaction
    pxr_row = find(strcmp(effectNames, 'Pretrain:Ratio'));
    if ~isempty(pxr_row)
        fprintf('  At Location = %s: Pretrain x Ratio  F(%d, %d) = %.3f,  p(GG) = %.4f,  eta_p^2 = %.4f\n', ...
            uLoc{i}, tbl.DF(pxr_row), tbl.DF(pxr_row+1), ...
            tbl.F(pxr_row), tbl.pValueGG(pxr_row), pes(pxr_row));
    end

    % Collect rows into cell array
    for r = 1:numRows
        rows_byLoc{end+1, 1} = uLoc{i};                       % FixFactor
        rows_byLoc{end, 2}   = effectNames{r};                % Effect
        rows_byLoc{end, 3}   = tbl.SumSq(r);
        rows_byLoc{end, 4}   = tbl.DF(r);
        rows_byLoc{end, 5}   = tbl.MeanSq(r);
        rows_byLoc{end, 6}   = tbl.F(r);
        rows_byLoc{end, 7}   = tbl.pValue(r);
        rows_byLoc{end, 8}   = tbl.pValueGG(r);
        rows_byLoc{end, 9}   = pes(r);
    end
end
simple2way_byLoc = cell2table(rows_byLoc, ...
    'VariableNames', {'Location', 'Effect', 'SumSq', 'DF', 'MeanSq', 'F', 'pValue', 'pValueGG', 'Partial_Eta_Sq'});
writetable(simple2way_byLoc, fullfile(savePath, 'Simple2Way_ByLocation_new.csv'));

% --- 2b. Fix Pretrain -> test Location x Ratio simple 2-way interaction ---
fprintf('\n--- Simple Location x Ratio at each Pretrain ---\n');
rows_byPre = {};
for i = 1:length(uPre)
    idx = withinDesign.Pretrain == uPre{i};
    colIdx = [1, find(idx)' + 1];
    subData = wideFormatData(:, colIdx);
    subNames = subData.Properties.VariableNames(2:end);
    subDesign = withinDesign(idx, {'Location', 'Ratio'});
    sub_rm = fitrm(subData, sprintf('%s-%s ~ 1', subNames{1}, subNames{end}), ...
                   'WithinDesign', subDesign);
    tbl = ranova(sub_rm, 'WithinModel', 'Location*Ratio');
    effectNames = tbl.Properties.RowNames;

    numRows = height(tbl);
    pes = zeros(numRows, 1);
    for r = 1:2:(numRows-1)
        if tbl.SumSq(r) + tbl.SumSq(r+1) > 0
            pes(r) = tbl.SumSq(r) / (tbl.SumSq(r) + tbl.SumSq(r+1));
        end
    end

    lxr_row = find(strcmp(effectNames, 'Location:Ratio'));
    if ~isempty(lxr_row)
        fprintf('  At Pretrain = %s: Location x Ratio  F(%d, %d) = %.3f,  p(GG) = %.4f,  eta_p^2 = %.4f\n', ...
            uPre{i}, tbl.DF(lxr_row), tbl.DF(lxr_row+1), ...
            tbl.F(lxr_row), tbl.pValueGG(lxr_row), pes(lxr_row));
    end

    for r = 1:numRows
        rows_byPre{end+1, 1} = uPre{i};
        rows_byPre{end, 2}   = effectNames{r};
        rows_byPre{end, 3}   = tbl.SumSq(r);
        rows_byPre{end, 4}   = tbl.DF(r);
        rows_byPre{end, 5}   = tbl.MeanSq(r);
        rows_byPre{end, 6}   = tbl.F(r);
        rows_byPre{end, 7}   = tbl.pValue(r);
        rows_byPre{end, 8}   = tbl.pValueGG(r);
        rows_byPre{end, 9}   = pes(r);
    end
end
simple2way_byPre = cell2table(rows_byPre, ...
    'VariableNames', {'Pretrain', 'Effect', 'SumSq', 'DF', 'MeanSq', 'F', 'pValue', 'pValueGG', 'Partial_Eta_Sq'});
writetable(simple2way_byPre, fullfile(savePath, 'Simple2Way_ByPretrain_new.csv'));
fprintf('\nSaved: Simple2Way_ByLocation_new.csv, Simple2Way_ByPretrain_new.csv\n');

%% ===== Section 3: Cohen's d (dz) for 3-Way Simple Simple Main Effects =====
fprintf('\n========== 3. Cohen''s d (dz) for 3-Way Pairwise Comparisons ==========\n');
fprintf('  (dz = |M_diff| / SD_diff, using SD of difference scores)\n\n');

% --- 3a. dz for Ratio comparisons within each (Pretrain, Location) ---
% Mirrors PostHoc_3Way_Pretrain_Location_Ratio.csv
fprintf('--- dz for Ratio pairwise comparisons ---\n');
cohensD_ratio = table();
for iP = 1:length(uPre)
    for iL = 1:length(uLoc)
        for r1 = 1:length(uRat)
            for r2 = (r1+1):length(uRat)
                col1 = sprintf('%s_L%d_R%d', uPre{iP}, iL, uRat(r1));
                col2 = sprintf('%s_L%d_R%d', uPre{iP}, iL, uRat(r2));
                % Defensive check
                if ~ismember(col1, measNames) || ~ismember(col2, measNames)
                    continue;
                end
                diffScores = wideFormatData.(col1) - wideFormatData.(col2);
                Mdiff = mean(diffScores);
                SDdiff = std(diffScores);
                dz = abs(Mdiff) / SDdiff;
                row = table({uPre{iP}}, {uLoc{iL}}, uRat(r1), uRat(r2), ...
                    Mdiff, SDdiff, dz, ...
                    'VariableNames', {'Pretrain', 'Location', 'Ratio1', 'Ratio2', 'M_diff', 'SD_diff', 'Cohens_dz'});
                cohensD_ratio = [cohensD_ratio; row];
            end
        end
    end
end
writetable(cohensD_ratio, fullfile(savePath, 'CohensD_3Way_Ratio_new.csv'));
fprintf('  Saved: CohensD_3Way_Ratio_new.csv (%d rows)\n', height(cohensD_ratio));

% --- 3b. dz for Face vs. Object within each (Location, Ratio) ---
% Mirrors PostHoc_3Way_Compare_Pretrain.csv
fprintf('--- dz for Face vs. Object comparisons ---\n');
cohensD_pretrain = table();
for iL = 1:length(uLoc)
    for iR = 1:length(uRat)
        colF = sprintf('Face_L%d_R%d', iL, uRat(iR));
        colO = sprintf('Object_L%d_R%d', iL, uRat(iR));
        if ~ismember(colF, measNames) || ~ismember(colO, measNames)
            continue;
        end
        diffScores = wideFormatData.(colF) - wideFormatData.(colO);
        Mdiff = mean(diffScores);
        SDdiff = std(diffScores);
        dz = abs(Mdiff) / SDdiff;
        row = table({uLoc{iL}}, uRat(iR), Mdiff, SDdiff, dz, ...
            'VariableNames', {'Location', 'Ratio', 'M_diff', 'SD_diff', 'Cohens_dz'});
        cohensD_pretrain = [cohensD_pretrain; row];
    end
end
writetable(cohensD_pretrain, fullfile(savePath, 'CohensD_3Way_Pretrain_new.csv'));
fprintf('  Saved: CohensD_3Way_Pretrain_new.csv (%d rows)\n', height(cohensD_pretrain));

fprintf('\n========== All supplementary analyses complete. ==========\n');
fprintf('Output files in: %s\n', savePath);
