%% Load simulation results
clear; close all 

fileName = 'empiricalSimResults'; 

% load mat file 
load([fileName '.mat']);

%% Extract gains correlation

pModel = T(strcmp(T.gen_model, 'p'), :);
pvModel = T(strcmp(T.gen_model, 'pv'), :);
pfModel = T(strcmp(T.gen_model, 'pf'), :);
pifModel = T(strcmp(T.gen_model, 'pif'), :);
pviModel = T(strcmp(T.gen_model, 'pvi'), :);
pvfModel = T(strcmp(T.gen_model, 'pvf'), :);

runIdx = unique(pvModel.run_idx); 

for s = 1:length(runIdx)

    sim = runIdx(s);
    
    % ---- p ----
    pModSim = pModel(pModel.run_idx == sim, :);

    corrP(sim, :) = [corr(cell2mat(pModSim.fit_gainsNum1), cell2mat(pModSim.real_gainsNum1))...
        corr(cell2mat(pModSim.fit_gainsNum2), cell2mat(pModSim.real_gainsNum2))];

    corrPmean(sim, 1) = corr(mean([cell2mat(pModSim.fit_gainsNum1) cell2mat(pModSim.fit_gainsNum2)], 2),...
        mean([cell2mat(pModSim.real_gainsNum1) cell2mat(pModSim.real_gainsNum2)], 2));

     
    % ---- pv ----
    pvModSim = pvModel(pvModel.run_idx == sim, :);

    A1 = cell2mat(pvModSim.fit_gainsNum1);   % N×2
    B1 = cell2mat(pvModSim.real_gainsNum1);  % N×2
    r1 = corr(A1(:), B1(:), 'rows','pairwise');  

    A2 = cell2mat(pvModSim.fit_gainsNum2);   % N×2
    B2 = cell2mat(pvModSim.real_gainsNum2);  % N×2
    r2 = corr(A2(:), B2(:), 'rows','pairwise');  

    corrPV(sim,:) = [r1, r2];
    corrPVmean(sim,1) = corr([A1(:); A2(:)], [B1(:); B2(:)], 'rows','pairwise');

    % ---- pf ----
    pfModSim = pfModel(pfModel.run_idx == sim, :);

    A1 = cell2mat(pfModSim.fit_gainsNum1);   % N×2
    B1 = cell2mat(pfModSim.real_gainsNum1);  % N×2
    r1 = corr(A1(:), B1(:), 'rows','pairwise');

    A2 = cell2mat(pfModSim.fit_gainsNum2);   % N×2
    B2 = cell2mat(pfModSim.real_gainsNum2);  % N×2
    r2 = corr(A2(:), B2(:), 'rows','pairwise');

    corrPF(sim,:) = [r1, r2];
    corrPFmean(sim,1) = corr([A1(:); A2(:)], [B1(:); B2(:)], 'rows','pairwise');


    % ---- pif ----
    pifModSim = pifModel(pifModel.run_idx == sim, :);

    A1 = cell2mat(pifModSim.fit_gainsNum1);   % N×3
    B1 = cell2mat(pifModSim.real_gainsNum1);  % N×3
    r1 = corr(A1(:), B1(:), 'rows','pairwise');

    A2 = cell2mat(pifModSim.fit_gainsNum2);   % N×3
    B2 = cell2mat(pifModSim.real_gainsNum2);  % N×3
    r2 = corr(A2(:), B2(:), 'rows','pairwise');

    corrPIF(sim,:) = [r1, r2];
    corrPIFmean(sim,1) = corr([A1(:); A2(:)], [B1(:); B2(:)], 'rows','pairwise');

    % ---- pvi ----
    pviModSim = pviModel(pviModel.run_idx == sim, :);

    A1 = cell2mat(pviModSim.fit_gainsNum1);   % N×3
    B1 = cell2mat(pviModSim.real_gainsNum1);  % N×3
    r1 = corr(A1(:), B1(:), 'rows','pairwise');

    A2 = cell2mat(pviModSim.fit_gainsNum2);   % N×3
    B2 = cell2mat(pviModSim.real_gainsNum2);  % N×3
    r2 = corr(A2(:), B2(:), 'rows','pairwise');

    corrPVI(sim,:) = [r1, r2];
    corrPVImean(sim,1) = corr([A1(:); A2(:)], [B1(:); B2(:)], 'rows','pairwise');

      % ---- pvf ----
    pvfModSim = pvfModel(pvfModel.run_idx == sim, :);

    A1 = cell2mat(pvfModSim.fit_gainsNum1);   % N×3
    B1 = cell2mat(pvfModSim.real_gainsNum1);  % N×3
    r1 = corr(A1(:), B1(:), 'rows','pairwise');

    A2 = cell2mat(pvfModSim.fit_gainsNum2);   % N×3
    B2 = cell2mat(pvfModSim.real_gainsNum2);  % N×3
    r2 = corr(A2(:), B2(:), 'rows','pairwise');

    corrPVF(sim,:) = [r1, r2];
    corrPVFmean(sim,1) = corr([A1(:); A2(:)], [B1(:); B2(:)], 'rows','pairwise');
end



%% Plot correlation between and across gains

models = {'p','pv','pf','pif','pvi','pvf'};

summaryCorr = table; 

for m = 1:numel(models)
    modelName = models{m};
    summaryCorr.model{m} = modelName;
    modelTable = T(strcmp(T.gen_model, modelName), :);

    nGains = numel(modelTable.fit_gainsNum1{1});

    corrSimRealFit = [];
    corrSimFitFit = [];
    real_all_mean = [];
    fit_all_mean  = [];

    runIdx = unique(modelTable.run_idx);

    fit_all2plot = [];
    real_all2plot = [];

    for s = 1:length(runIdx)

        sim = runIdx(s);

        modelTableSim = modelTable(modelTable.run_idx == sim, :);

        % Pool both controllers (60×nGains)
        fit_all  = [cell2mat(modelTableSim.fit_gainsNum1); cell2mat(modelTableSim.fit_gainsNum2)];
        real_all = [cell2mat(modelTableSim.real_gainsNum1); cell2mat(modelTableSim.real_gainsNum2)];

        fit_all_mean(sim,:)  = mean(fit_all);
        real_all_mean(sim,:) = mean(real_all);

        % Per-run per-gain recovery
        corrSimRealFit(sim,:) = diag(corr(fit_all, real_all));

        fit_all2plot = [fit_all2plot; fit_all(:)];
        real_all2plot = [real_all2plot; real_all(:)];

    end

    % Fisher-z average across runs
    fisher = @(r) atanh(r);
    invf   = @(z) tanh(z);

    z = fisher(corrSimRealFit);
    mean_z = mean(z, 1, 'omitnan');
    std_z  = std(z, 0, 1, 'omitnan');

    corrSimRealFitFisherAvg = invf(mean_z);
    corrSimRealFitFisherStd = invf(mean_z + std_z) - corrSimRealFitFisherAvg;
    
    colors = winter(nGains);

    % --- Combined figure ---
    figure('Name', sprintf('%s - Recovery', modelName), 'Color', 'w'); hold on

    for g = 1:nGains
        fit_all  = [cell2mat(modelTable.fit_gainsNum1); cell2mat(modelTable.fit_gainsNum2)];
        real_all = [cell2mat(modelTable.real_gainsNum1); cell2mat(modelTable.real_gainsNum2)];
        scatter(real_all(:,g), fit_all(:,g), 25, colors(g,:), 'filled', 'MarkerFaceAlpha', 0.5); 
    end

    if nGains == 1
        legend([modelName(1) ' r= ' num2str(corrSimRealFitFisherAvg(1))]);
    elseif nGains == 2
        legend([modelName(1) ' r= ' num2str(corrSimRealFitFisherAvg(1))], [modelName(2) ' r= ' num2str(corrSimRealFitFisherAvg(2))]);
    elseif nGains == 3
        legend([modelName(1) ' r= ' num2str(corrSimRealFitFisherAvg(1))], [modelName(2) ' r= ' num2str(corrSimRealFitFisherAvg(2))], ...
            [modelName(3) ' r= ' num2str(corrSimRealFitFisherAvg(3))]);
    end

    title(['r = ', num2str(mean(corrSimRealFitFisherAvg)) ' std=' num2str(mean(corrSimRealFitFisherStd))]);
    xlabel(['True gain: ' modelName]);
    ylabel(['Fitted gain: ' modelName]);
    ylim([0 40]); xlim([1 40]);
    % ylim([min([real_all(:); fit_all(:)]) max([real_all(:); fit_all(:)])]); 
    % xlim([min([real_all(:); fit_all(:)]) max([real_all(:); fit_all(:)])]);
    axis square;
    set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 12);
   
    summaryCorr.corr(m) = mean(corrSimRealFitFisherAvg);
    summaryCorr.std(m) = mean(mean(corrSimRealFitFisherStd));

end


%% Compare correlations across gain regimes
load('allCorr.mat')

modelPmean = mean(b.corr(strcmp(b.model, 'p')));
modelPstd = mean(b.std(strcmp(b.model, 'p')));

modelPVmean = mean(b.corr(strcmp(b.model, 'pv')));
modelPVstd = mean(b.std(strcmp(b.model, 'pv')));

modelPFmean = mean(b.corr(strcmp(b.model, 'pf')));
modelPFstd = mean(b.std(strcmp(b.model, 'pf')));

modelPIFmean = mean(b.corr(strcmp(b.model, 'pif')));
modelPIFstd = mean(b.std(strcmp(b.model, 'pif')));

modelPVImean = mean(b.corr(strcmp(b.model, 'pvi')));
modelPVIstd = mean(b.std(strcmp(b.model, 'pvi')));

modelPVFmean = mean(b.corr(strcmp(b.model, 'pvf')));
modelPVFstd = mean(b.std(strcmp(b.model, 'pvf')));

% Assuming your table is called T
models = unique(b.model, 'stable');
sims   = unique(b.sim, 'stable');

% Preallocate matrices
meanVals = nan(numel(models), numel(sims));
stdVals  = nan(numel(models), numel(sims));

% Fill matrices
for i = 1:numel(models)
    for j = 1:numel(sims)
        idx = strcmp(b.model, models{i}) & strcmp(b.sim, sims{j});
        meanVals(i,j) = mean(b.corr(idx), 'omitnan');
        stdVals(i,j)  = mean(b.std(idx), 'omitnan');
    end
end

figure;
% --- Plot grouped bars ---
bb = bar(meanVals, 'grouped'); hold on
colors = lines(numel(sims));
for j = 1:numel(sims)
    bb(j).FaceColor = colors(j,:);
end

% --- Add error bars ---
ngroups = size(meanVals, 1);
nbars   = size(meanVals, 2);
groupwidth = min(0.8, nbars/(nbars + 1.5));

for i = 1:nbars
    % X positions of the bars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, meanVals(:,i), stdVals(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1);
end
ylim([0 1])
set(gca, 'XTick', 1:numel(models), 'XTickLabel', models,'TickDir','out','FontName','Arial','FontSize',12);
xlabel('Controller class');
ylabel('Correlation (r)');
legend(sims, 'Location','northoutside','Orientation','horizontal');
box off; axis square;
