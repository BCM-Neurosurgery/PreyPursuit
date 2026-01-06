%% Load simulation results
clear; close all
fileName = 'empiricalSimResults';
load([fileName '.mat']);

models  = {'p','pv','pf','pvi','pif','pvf'};
nModels = numel(models);

confMatrixCat = [];
confCountsCat = [];

runIdx = unique(T.run_idx);

for sim = 1:numel(runIdx)
    s = runIdx(sim);
    Tsim = T(T.run_idx == s, :);

    % ---- Build elboAllModels ONCE per sim ----
    elboAllModels = table();

    for k = 1:numel(models)
        Tmodel = Tsim(strcmp(Tsim.gen_model, models{k}), :);

        if isempty(Tmodel), continue; end

        elboModel = table();
        n = height(Tmodel);

        % store as string for safety
        elboModel.gen_model = repmat(string(models{k}), n, 1);

        elboModel.elbo_p   = Tmodel.elbo_p;
        elboModel.elbo_pv  = Tmodel.elbo_pv;
        elboModel.elbo_pf  = Tmodel.elbo_pf;
        elboModel.elbo_pvi = Tmodel.elbo_pvi;
        elboModel.elbo_pif = Tmodel.elbo_pif;
        elboModel.elbo_pvf = Tmodel.elbo_pvf;

        elboModel.elbo_gen = Tmodel.elbo_gen;

        elboAllModels = [elboAllModels; elboModel];
    end

    confCounts = zeros(nModels, nModels);

    for i = 1:height(elboAllModels)
        gen = char(elboAllModels.gen_model(i));          % 'p','pv',...
        genIdx = find(strcmp(models, gen));

        % fill generative ELBO into correct column
        genCol = ['elbo_' gen];
        elboAllModels.(genCol)(i) = elboAllModels.elbo_gen(i);

        % extract all candidate elbos
        elbos = nan(1, nModels);
        for m = 1:nModels
            colName = ['elbo_' models{m}];
            elbos(m) = elboAllModels.(colName)(i);
        end

        [~, bestIdx] = max(elbos);
        confCounts(genIdx, bestIdx) = confCounts(genIdx, bestIdx) + 1;
    end

    confMatrix = confCounts ./ sum(confCounts, 2);

    confMatrixCat = cat(3, confMatrixCat, confMatrix);
    confCountsCat = cat(3, confCountsCat, confCounts);
end

meanConfMatrix = mean(confMatrixCat, 3);

%% Heatmap
figure('Color','w');
h = heatmap(models, models, meanConfMatrix, ...
    'CellLabelFormat','%.2f', 'XLabel','Fitted model', 'YLabel','Generating model', ...
    'FontName','Arial', 'FontSize',12);

h.ColorLimits = [0 0.5];
h.GridVisible = 'off';
h.CellLabelColor = 'k';
title('Model Confusion Matrix - Random 1-40 gain regimes');


%% Aggregate counts + stats
confCounts = sum(confCountsCat, 3);

for r = 1:nModels
    successes = confCounts(r,r);
    ntrials   = sum(confCounts(r,:));
    p0        = 1/nModels;  % chance level
    pval(r)   = 1 - binocdf(successes-1, ntrials, p0);
    fprintf('Model %s: %d/%d correct (%.2f), p=%.4f\n', ...
        models{r}, successes, ntrials, successes/ntrials, pval(r));
end

models  = {'p','pv','pf','pvi','pif','pvf'};
nModels = numel(models);

diagAcc   = zeros(1,nModels);
ciLow     = zeros(1,nModels);
ciHigh    = zeros(1,nModels);

for r = 1:nModels
    successes = confCounts(r,r);          % diagonal count
    ntrials   = sum(confCounts(r,:));     % total datasets for this generator
    diagAcc(r) = successes / ntrials;

    % 95% binomial CI (Clopper–Pearson exact)
    [phat, pci] = binofit(successes, ntrials, 0.05);
    ciLow(r)  = pci(1);
    ciHigh(r) = pci(2);

    fprintf('%s: %.2f (95%% CI [%.2f, %.2f])\n', ...
        models{r}, diagAcc(r), ciLow(r), ciHigh(r));
end


