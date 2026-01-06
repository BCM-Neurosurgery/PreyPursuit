%% Load simulation results
clear; close all 

fileName = 'empiricalSimResults'; 

% load mat file 
load([fileName '.mat']);

%% Plot correlation between and across gains (one figure per controller class)
models = {'p', 'pv','pf','pif','pvi','pvf'};

% Precompute Wt features once
for s = 1:size(T, 1)
    halfWtidx       = round(length(T.wtsim{s})/2);
    T.midWtfit(s)   = T.wtsim{s}(halfWtidx);
    T.startWtfit(s) = T.wtsim{s}(10);
    T.endWtfit(s)   = T.wtsim{s}(end-10);
end

for m = 1:numel(models)
    modelName  = models{m};
    modelTable = T(strcmp(T.gen_model, modelName), :);

    nGains = numel(modelTable.fit_gainsNum1{1});

    corrMidWtFitGain      = [];
    corrStartWtFitGain    = [];
    corrEndWtFitGain      = [];
    partialCorrMidWtFitGain   = [];
    partialCorrStartWtFitGain = [];
    partialCorrEndWtFitGain   = [];

    runIdx = unique(modelTable.run_idx);

    for s = 1:numel(runIdx)
        sim = runIdx(s);
        modelTableSim = modelTable(modelTable.run_idx == sim, :);

        % --- Pool both controllers row-wise (rows = trials*60*2, cols = nGains)
        fit_all  = [cell2mat(modelTableSim.fit_gainsNum1); cell2mat(modelTableSim.fit_gainsNum2)];
        real_all = [cell2mat(modelTableSim.real_gainsNum1); cell2mat(modelTableSim.real_gainsNum2)];

        % --- Duplicate Wt features to match pooled rows
        Wt_mid_all   = [modelTableSim.midWtfit;   modelTableSim.midWtfit];
        Wt_start_all = [modelTableSim.startWtfit; modelTableSim.startWtfit];
        Wt_end_all   = [modelTableSim.endWtfit;   modelTableSim.endWtfit];

        % --- Per-gain correlations & partial correlations (control for true gain g)
        for g = 1:nGains
            % Raw correlations
            corrMidWtFitGain(s,g)   = corr(Wt_mid_all,   fit_all(:,g), 'rows','pairwise');
            corrStartWtFitGain(s,g) = corr(Wt_start_all, fit_all(:,g), 'rows','pairwise');
            corrEndWtFitGain(s,g)   = corr(Wt_end_all,   fit_all(:,g), 'rows','pairwise');

            % Partial correlations controlling for real gain g
            Z   = real_all(:,g);
            resX_mid   = Wt_mid_all   - Z * (Z \ Wt_mid_all);
            resX_start = Wt_start_all - Z * (Z \ Wt_start_all);
            resX_end   = Wt_end_all   - Z * (Z \ Wt_end_all);
            resY       = fit_all(:,g) - Z * (Z \ fit_all(:,g));

            partialCorrMidWtFitGain(s,g)   = corr(resX_mid,   resY, 'rows','pairwise');
            partialCorrStartWtFitGain(s,g) = corr(resX_start, resY, 'rows','pairwise');
            partialCorrEndWtFitGain(s,g)   = corr(resX_end,   resY, 'rows','pairwise');
        end
    end

    % --- Fisher-z average across runs
    fisher = @(r) atanh(r);
    invf   = @(z) tanh(z);

    corrMidWtFitGainAvg       = invf(mean(fisher(corrMidWtFitGain),         1, 'omitnan'));
    corrStartWtFitGainAvg     = invf(mean(fisher(corrStartWtFitGain),       1, 'omitnan'));
    corrEndWtFitGainAvg       = invf(mean(fisher(corrEndWtFitGain),         1, 'omitnan'));
    partialCorrMidWtFitGainAvg   = invf(mean(fisher(partialCorrMidWtFitGain),   1, 'omitnan'));
    partialCorrStartWtFitGainAvg = invf(mean(fisher(partialCorrStartWtFitGain), 1, 'omitnan'));
    partialCorrEndWtFitGainAvg   = invf(mean(fisher(partialCorrEndWtFitGain),   1, 'omitnan'));

    % Figure: one per model class 
    fit_all_all  = [cell2mat(modelTable.fit_gainsNum1); cell2mat(modelTable.fit_gainsNum2)];
    Wt_start_all = [modelTable.startWtfit; modelTable.startWtfit];
    Wt_mid_all   = [modelTable.midWtfit;   modelTable.midWtfit];
    Wt_end_all   = [modelTable.endWtfit;   modelTable.endWtfit];

    % Common axis limits 
    xlim_common = [0 1];
    ymax = prctile(fit_all_all(:), 99);          
    ymax = max(ymax, 1);                         
    ylim_common = [0, ceil(ymax/5)*5]; 
    
    

    tileW = 6;   
    tileH = 6;  
    nCols = ceil(sqrt(nGains));
    nRows = ceil(nGains / nCols);
    marginL = 1.2; marginR = 0.5; marginT = 1.2; marginB = 1.2;   % cm
    figW = marginL + nCols*tileW + marginR;
    figH = marginB + nRows*tileH + marginT;

    f = figure('Color','w','Name',['Wt_vs_Gain_' modelName], ...
               'Units','centimeters','Position',[2 2 figW figH], ...
               'PaperUnits','centimeters','PaperPosition',[0 0 figW figH]);

    tlo = tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','compact');
    sgtitle(tlo, sprintf('Wt vs Fitted Gains — %s', upper(modelName)));

    cols = [ 1.0 0.6 0.2; 1.0 0.4 0.0; 0.8 0.1 0.1 ];  % Start/Mid/End

    for g = 1:nGains
        ax = nexttile; hold(ax,'on');

        s1 = scatter(ax, Wt_start_all, fit_all_all(:,g), 8, 'filled', ...
            'MarkerFaceColor', cols(1,:), 'MarkerFaceAlpha', 0.5);
        s2 = scatter(ax, Wt_mid_all,   fit_all_all(:,g), 8, 'filled', ...
            'MarkerFaceColor', cols(2,:), 'MarkerFaceAlpha', 0.5);
        s3 = scatter(ax, Wt_end_all,   fit_all_all(:,g), 8, 'filled', ...
            'MarkerFaceColor', cols(3,:), 'MarkerFaceAlpha', 0.5);

        xlabel(ax,'Recovered W_t');
        ylabel(['Recovered gain: ' modelName(g)]);
        title(ax, sprintf('r: Start=%.2f  Mid=%.2f  End=%.2f', ...
            corrStartWtFitGainAvg(g), corrMidWtFitGainAvg(g), corrEndWtFitGainAvg(g)));

        axis(ax,'square');
        xlim(ax, xlim_common);
        ylim(ax, ylim_common);
        set(ax, 'TickDir','out', 'Color','none', 'Box','off', ...
            'FontName','Arial', 'FontSize', 13);

        if g == 1
            legend(ax,[s1 s2 s3], {'Start','Mid','End'}, 'Location','best'); legend(ax,'boxoff');
        end
    end

    % Save one file per controller class
    %saveas(f, sprintf('Empirical_Wt_vs_gain_%s.svg', modelName));  
end


