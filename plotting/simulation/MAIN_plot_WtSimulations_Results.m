%% Load simulation results
clear; close all 

fileName = 'empiricalSimResults'; 

% load mat file 
load([fileName '.mat']);

%% Extract Wt simulation correlation and position correlation

for ss = 1:size(T, 1)
    T.wtCorr(ss) = corr(T.wtsim{ss}(1,:)', T.actual_shift{ss}(1,:)');
end

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

    wtCorrP(sim) = mean((pModSim.wtCorr));
    wtCorrPGen(sim) = mean((pModSim.wtcorr_gen));
    
    posCorrP(sim) = mean((pModSim.poscorr_gen));
    
    % ---- pv ----
    pvModSim = pvModel(pvModel.run_idx == sim, :);

    wtCorrPV(sim) = mean((pvModSim.wtCorr));
    wtCorrPVGen(sim) = mean((pvModSim.wtcorr_gen));

    posCorrPV(sim) = mean((pvModSim.poscorr_gen));

    % ---- pf ----
    pfModSim = pfModel(pfModel.run_idx == sim, :);

    wtCorrPF(sim) = mean((pfModSim.wtCorr));
    wtCorrPFGen(sim) = mean((pfModSim.wtcorr_gen));

    posCorrPF(sim) = mean((pfModSim.poscorr_gen));

    % ---- pif ----
    pifModSim = pifModel(pifModel.run_idx == sim, :);

    wtCorrPIF(sim) = mean((pifModSim.wtCorr));
    wtCorrPIFGen(sim) = mean((pifModSim.wtcorr_gen));

    posCorrPIF(sim) = mean((pifModSim.poscorr_gen));

    % ---- pvi ----
    pviModSim = pviModel(pviModel.run_idx == sim, :);
    
    wtCorrPVI(sim) = mean((pviModSim.wtCorr));
    wtCorrPVIGen(sim) = mean((pviModSim.wtcorr_gen));

    posCorrPVI(sim) = mean((pviModSim.poscorr_gen));

    % ---- pvf ----
    pvfModSim = pvfModel(pvfModel.run_idx == sim, :);

    wtCorrPVF(sim) = mean((pvfModSim.wtCorr));
    wtCorrPVFGen(sim) = mean((pvfModSim.wtcorr_gen));

    posCorrPVF(sim) = mean((pvfModSim.poscorr_gen));

end


%% Bar plots with correlations: Wt (Figure 1E) and positions

models  = {'p','pv','pf','pvi','pif','pvf'};
nModels = numel(models);


% Barplot of Wt correlation by gen model - simulated vs recovered Wt (Fig
% 1E empirical results)
avgCorr = [nanmean((wtCorrP)) nanmean((wtCorrPV)) nanmean(wtCorrPF) nanmean(wtCorrPVI) nanmean(wtCorrPIF) nanmean(wtCorrPVF)];

figure('Color','w');
bar(avgCorr);
set(gca,'XTick',1:nModels,'XTickLabel',models);
ylabel('Correlation (r)');
xlabel('Wt recovery');
title('Correlation: real and recovered Wt — mean across targets');
ylim([0 1]); yline(0,'k:'); grid on;
%saveas(gcf, ['WtCorr_' fileName '.svg']);


% Barplot of Wt trajectory correlation by gen model
avgCorrPos = [mean(posCorrP) mean(posCorrPV) mean(posCorrPF) mean(posCorrPVI) mean(posCorrPIF) mean(posCorrPVF)];

figure;
bar(avgCorrPos);
set(gca,'XTick',1:nModels,'XTickLabel',models);
ylabel('Correlation (r)');
xlabel('Position recovery');
title('Correlation: real vs recovered trajectories — mean across targets');
ylim([0 1]); yline(0,'k:'); grid on;
%saveas(gcf, ['PosCorr_' fileName '.svg']);

%% Visualize examples of simulated Wt vs recovered Wt 

for k = 1:35 % be careful this are a lot of plots! change the end index to visualize different Wt examples
    if size(T.actual_shift{k}(1, :), 2) > 100
        figure; hold on;
        plot(T.actual_shift{k}(1, :), 'LineWidth', 1.5);
        plot(T.wtsim{k}(1, :), 'LineWidth', 1.5);
        legend('real', 'recovered');
        r = corr(T.actual_shift{k}(1, :)', T.wtsim{k}(1, :)');
        title(['r = ' num2str(r)]);
        xlabel('Time (ms)');
        ylabel('Wt');
        set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 12);
        axis square
    end
end

%% Scatter plot showing correlation of simulated Wt vs recovered Wt 

models  = {'p','pv','pf','pvi','pif','pvf'};

modelTables = {pModel, pvModel, pfModel, pviModel, pifModel, pvfModel};

for m = 1:numel(models)
    Tm = modelTables{m};

    true_all  = [];
    fitted_all = [];
    trIdx_all = [];

    trials = find(Tm.wtCorr > -1); %selecting all the trials
    n = min(100, numel(trials));
    idx = randsample(numel(trials), n);

    for r = 1:length(idx)
        tr = trials(idx(r));
        true_all   = [true_all,   Tm.actual_shift{tr}(1,:)];
        fitted_all = [fitted_all, Tm.wtsim{tr}(1,:)];
        trIdx_all = [trIdx_all; repmat(tr, length(Tm.actual_shift{tr}(1,:)), 1)];
    end

    figure('Color','w');
    scatter(true_all, fitted_all, 40, trIdx_all)
    xlabel('Simulated Wt');
    ylabel('Recovered Wt');
    title(sprintf('Wt recovery: model %s', models{m}));

end

%% Plot correlation between and across gains 

models     = {'p','pv','pf','pvi','pif','pvf'};
modelsCap  = {'P','PV','PF','PVI','PIF','PVF'};
modelTables = {pModel, pvModel, pfModel, pviModel, pifModel, pvfModel};

wtCorrThresh = -1; % plotting all the correlations if threshold = - 1
epoch_fracs  = [0.10 0.50 0.90];  % start, mid, end as fractions of trial length


figure('Color','w','Position',[100 100 1200 1800]); 
tiledlayout(6,3,'Padding','compact','TileSpacing','compact');

for m = 1:numel(models)
    Tm = modelTables{m};
    if isempty(Tm), continue; end

    % trial selection
    if ismember('wtCorr', Tm.Properties.VariableNames)
        trials = find(Tm.wtCorr > wtCorrThresh);
    else
        trials = 1:height(Tm);  
    end
    if isempty(trials), continue; end

    % collect epoch samples across trials
    True = cell(1,3);  Hat = cell(1,3);   % {Start, Mid, End}
    for r = 1:numel(trials)
        tr = trials(r);

        wt_true = Tm.actual_shift{tr};
        wt_hat  = Tm.wtsim{tr};

        % standardize shape → column vectors
        wt_true = wt_true(:);
        wt_hat  = wt_hat(:);

        T = numel(wt_true);
       
        % epoch indices
        idx = max(1, min(T, round(epoch_fracs*T)));
        for e = 1:3
            ii = idx(e);

            tval = wt_true(ii);
            hval = wt_hat(ii);

            True{e}(end+1,1) = tval;
            Hat{e}(end+1,1)  = hval;
        end
    end

    % plot per-epoch scatters + annotate r and R^2
    titles = {'Start','Mid','End'};
    for e = 1:3
        nexttile;
        scatter(True{e}, Hat{e}, 16, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
        
        % unity line
        lims = [min([True{e};Hat{e}]) max([True{e};Hat{e}])];
        if ~isempty(lims) && all(isfinite(lims)) && diff(lims)>0
            xlim(lims); ylim(lims);
            plot(lims, lims, 'k--'); 
        end

        axis square; 
        xlabel('True W_t'); ylabel('Recovered W_t'); title([titles{e} ' ' modelsCap{m}]);
        set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 12);
        set(gcf, 'Position', [100 100 900 1200]);

        % stats
        if numel(True{e}) >= 3
            r = corr(True{e}, Hat{e}, 'rows','pairwise');
            r2 = r.^2;
            text(mean(xlim), max(ylim), sprintf('r = %.2f, R^2 = %.2f', r, r2), ...
                 'HorizontalAlignment','center','VerticalAlignment','top', ...
                 'FontSize', 11, 'FontWeight', 'bold');
        end
    end
    
    sgtitle('Epoch-based W_t recovery across controller classes');

end

%% Histogram of Wt recovery correlations across controller classes (Figure 1F) 
% --- compute per-model and overall Wt recovery correlations ---

models = {'p','pv','pf','pvi','pif','pvf'};
modelTables = {pModel, pvModel, pfModel, pviModel, pifModel, pvfModel};

figure; hold on
fprintf('\n=== Wt recovery correlations per controller ===\n\n');
all_corr = [];
for m = 1:numel(models)
    Tm = modelTables{m};
    if isempty(Tm) || ~ismember('wtCorr', Tm.Properties.VariableNames)
        continue
    end
    vals = Tm.wtCorr;
    neg(m) = sum(vals < 0);
    dime(m) = size(vals, 1);
    
    histogram(vals, 20, "EdgeColor","none", "EdgeAlpha", 0.5); xlabel('r(W_t true, recovered)'); ylabel('count'); 
    set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 12);
    vals = vals(~isnan(vals));  % remove NaNs
    if isempty(vals), continue; end

    med(m) = median(vals);

    mean_r = mean(vals, 'omitnan');
    std_r  = std(vals, 'omitnan');
    fprintf('%-4s : r = %.3f ± %.3f (n=%d)\n', models{m}, mean_r, std_r, numel(vals));

    all_corr = [all_corr; vals(:)];
end
legend(models)
fprintf('\nOverall : r = %.3f ± %.3f (n=%d)\n', ...
    mean(all_corr,'omitnan'), std(all_corr,'omitnan'), numel(all_corr));

%saveas(gcf, ['/Users/assiachericoni/Documents/Hayden_Lab/WritingPapers/PacManControlPaper_FineChericonietal/Revision/Supp1/' fileName '_WtcorrHist.svg']);

%% Cumulative plot across simulations
load('corrWtall.mat')

% ----- Inputs -----
% corrWt.emp.avgCorr   % 6 x 30   (controllers x sims)
% corrWt.rnd5.avgCorr
% corrWt.rnd40.avgCorr

controllers = {'P','PV','PF','PVI','PIF','PVF'};
regimes     = {'emp','rnd5','rnd40'};
regData     = {corrWt.emp.avgCorr, corrWt.rnd5.avgCorr, corrWt.rnd40.avgCorr};

useFisher = false;   % average r with Fisher z
mu = nan(6, numel(regData));
sd = nan(6, numel(regData));

for k = 1:numel(regData)
    X = regData{k};              % 6 x Nsims_k (Nsims can differ)
    if useFisher
        Z = atanh(X);            % Fisher z
        mu(:,k) = tanh(mean(Z, 2, 'omitnan'));
        % SD in r-space (compute per column back in r, then SD across sims)
        rCols   = tanh(Z);
        sd(:,k) = std(rCols, 0, 2, 'omitnan');
    else
        mu(:,k) = mean(X, 2, 'omitnan');
        sd(:,k) = std(X, 0, 2, 'omitnan');
    end
end

% ---- Plot grouped bars + error bars ----
figure('Color','w','Position',[100 100 900 500]);
b = bar(mu, 'grouped'); hold on;

% x-positions for errorbars
ng = size(mu,1); nb = size(mu,2);
xpos = nan(ng, nb);
for k = 1:nb
    xpos(:,k) = b(k).XEndPoints;
end
for k = 1:nb
    errorbar(xpos(:,k), mu(:,k), sd(:,k), 'k', 'linestyle','none', 'linewidth', 1);
end

set(gca,'XTick',1:ng, 'XTickLabel',controllers, 'TickDir','out', ...
        'FontName','Arial','FontSize',12, 'Box','off');
ylabel('Mean correlation (r)'); xlabel('Controller class');
ylim([0 1]); legend(regimes,'Location','northwest'); legend boxoff;
title('Wt recovery (true vs recovered): mean \pm SD across simulations');

