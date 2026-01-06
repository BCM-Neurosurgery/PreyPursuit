%% Load simulation results
clear; close all 

fileName = 'empiricalSimResults'; 

% load mat file 
load([fileName '.mat']);

%% Plot scatter of correlation across recovered gains

models = {'pv','pf','pif','pvi','pvf'};

for m = 1:numel(models)
    modelName = models{m};
    modelTable = T(strcmp(T.gen_model, modelName), :);

    nGains = numel(modelTable.fit_gainsNum1{1});
    combs = nchoosek(1:nGains, 2);

    corrSimFitFit = [];
    fit_all_mean  = [];

    runIdx = unique(modelTable.run_idx);

    for s = 1:length(runIdx)
        sim = runIdx(s);
        modelTableSim = modelTable(modelTable.run_idx == sim, :);

        % Pool both controllers
        fit_all = [cell2mat(modelTableSim.fit_gainsNum1); cell2mat(modelTableSim.fit_gainsNum2)];
        fit_all_mean(sim,:) = mean(fit_all);

        % Per-run gain–gain correlations
        for c = 1:size(combs,1)
            g1 = combs(c,1); g2 = combs(c,2);
            corrSimFitFit(s,c) = corr(fit_all(:,g1), fit_all(:,g2), 'rows','pairwise');
        end
    end

    % Fisher-z average across runs
    fisher = @(r) atanh(r);
    invf   = @(z) tanh(z);

    z = fisher(corrSimFitFit);
    mean_z = mean(z, 1, 'omitnan');
    std_z  = std(z, 0, 1, 'omitnan');
    corrSimFitFitFisherAvg = invf(mean_z);
    corrSimFitFitFisherStd = invf(mean_z + std_z) - corrSimFitFitFisherAvg;

    fit_all = [cell2mat(modelTable.fit_gainsNum1); cell2mat(modelTable.fit_gainsNum2)];

    % --- Figure sizing: consistent subplot size ---
    nPlots = size(combs,1);
    nCols = nPlots;    % 1 row with all plots side by side
    nRows = 1;

    tileW = 6; tileH = 6;  % cm per tile
    marginL = 1.2; marginR = 0.5; marginT = 1.2; marginB = 1.2;   % cm
    figW = marginL + nCols*tileW + marginR;
    figH = marginB + nRows*tileH + marginT;

    f = figure('Color','w','Name',['Gains_tradeOff_' modelName], ...
        'Units','centimeters','Position',[2 2 figW figH], ...
        'PaperUnits','centimeters','PaperPosition',[0 0 figW figH]);

    tlo = tiledlayout(f, nRows, nCols, 'TileSpacing','compact', 'Padding','compact');
    sgtitle(tlo, upper(modelName));

    % --- Plot each gain–gain pair ---
    for c = 1:size(combs,1)
        g1 = combs(c,1); g2 = combs(c,2);
        nexttile(tlo);
        scatter(fit_all(:,g1), fit_all(:,g2), 25, 'filled', 'MarkerFaceAlpha', 0.5);
        axis square;
        xlabel(['Fitted gain: ' modelName(g1)]);
        ylabel(['Fitted gain: ' modelName(g2)]);
        % title(sprintf('r = %.2f', corrSimFitFitFisherAvg(c)), 'FontWeight', 'bold');
        x = fit_all(:,g1);
        y = fit_all(:,g2);
        r_pooled = corr(x, y, 'rows','pairwise');

        title(sprintf('pooled r = %.2f | mean-run r = %.2f', r_pooled, corrSimFitFitFisherAvg(c)), 'FontWeight','bold');
        set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', ...
            'FontName', 'Arial', 'FontSize', 11);
        xlim([0 40]); ylim([0 40])
    end

end





