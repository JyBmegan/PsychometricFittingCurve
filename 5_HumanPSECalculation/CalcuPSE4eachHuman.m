clear; clc; close all;

resultsDir = 'results'; 
startSubj = 2;
endSubj = 44;

paraX = linspace(0, 1, 11); 
UL = [0.05, 0.05, 1, 100]; 
SP = [0.01, 0.02, 0.5, 5.0]; 
LM = [0, 0, 0, 0.1];

numTotal = endSubj - startSubj + 1;
subjectIDs = zeros(numTotal, 1);
pse_u      = zeros(numTotal, 1);
sd_v       = zeros(numTotal, 1);
guess_g    = zeros(numTotal, 1);
lapse_l    = zeros(numTotal, 1);

validCount = 0; 

for subjNum = startSubj:endSubj
    folderPath = fullfile(resultsDir, num2str(subjNum));

    matFiles = dir(fullfile(folderPath, '*.mat'));
    if isempty(matFiles)
        warning ('No .mat files in folder %s', folderPath);
        continue;
    end
    
    matData = load(fullfile(folderPath, matFiles(1).name));
    
    if ~isfield(matData, 'result_dummy')
        warning ('No result_dummy in file %s', matFiles(1).name);
        continue;
    end
    
    result_dummy = matData.result_dummy;
    
    rawRatings = zeros(1, 11);
    for c = 1:11
        idx = (result_dummy(:, 2) == c);
        chunk = result_dummy(idx, 4); 
        
        if isempty(chunk)
            continue;
        end
        
        rawRatings(c) = 2 - mean(chunk);
    end
    
    try
        [ffit, ~, ~] = FitPsycheCurveWH(paraX, rawRatings, UL, SP, LM);
        
        validCount = validCount + 1;
        subjectIDs(validCount) = subjNum;
        pse_u(validCount)      = ffit.u;
        sd_v(validCount)       = ffit.v;
        guess_g(validCount)    = ffit.g;
        lapse_l(validCount)    = ffit.l;
    catch ME
        warning ('Error for participant %d because: %s', subjNum, ME.message);
    end
end

subjectIDs = subjectIDs(1:validCount);
pse_u      = pse_u(1:validCount);
sd_v       = sd_v(1:validCount);
guess_g    = guess_g(1:validCount);
lapse_l    = lapse_l(1:validCount);

group_mean = mean(pse_u);
group_sd   = std(pse_u); 
group_se   = group_sd / sqrt(validCount);

T = table(subjectIDs, pse_u, sd_v, guess_g, lapse_l, ...
    'VariableNames', {'SubjectID', 'PSE_u', 'Curve_SD_v', 'GuessRate_g', 'LapseRate_l'});

csvName = 'Human_PSE_40_Subjects.csv';
writetable(T, csvName);

logText = sprintf('\n==================================================\n');
logText = [logText, sprintf(' %d participants finded: %s\n', validCount, csvName)];
logText = [logText, sprintf('Group Mean PSE (Mean value for all PSE results, not PSE for all trials between participants) = %.6f\n', group_mean)];
logText = [logText, sprintf('Group PSE SD = %.6f\n', group_sd)];
logText = [logText, sprintf('Group PSE SE = %.6f\n', group_se)];
logText = [logText, sprintf('==================================================\n')];
logText = [logText, sprintf('Z-score formula for SE-AlexNet:\n')];
logText = [logText, sprintf('Z_AlexNet = (AlexNet_PSE - %.6f) / %.6f\n', group_mean, group_sd)];

fprintf('%s', logText);
txtFileName = 'Human_PSE_Summary_Log.txt';
fileID = fopen(txtFileName, 'w');
if fileID ~= -1
    fprintf(fileID, '%s', logText);
    fclose(fileID);
end