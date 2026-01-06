%% Load simulation results
clear; close all
fileName = 'empiricalSimResults';
load([fileName '.mat']);

%% PER-MODEL METRICS — continuous Dt_ms (no helper functions)

models  = {'p','pv','pf','pvi','pif','pvf'};
dt_ms   = 1000/60;   % frame duration in ms (adjust if yours differs)

stats_all   = cell(numel(models),1);
stats_nosat = cell(numel(models),1);

halfwin = 5;                 % +/- bins to estimate local W noise
sat_eps = max(1e-3, 10*eps); % saturation threshold

for m = 1:numel(models)
    M = T(strcmp(T.gen_model, models{m}), :);

    W_hat_at_true  = [];
    W_true         = [];
    Dt_ms          = [];
    sat_flag       = [];
    CI_time_lo     = [];
    CI_time_hi     = [];

    for k = 1:height(M)
        w_true = M.actual_shift{k};
        w_hat  = M.wtsim{k};
        if isempty(w_true) || isempty(w_hat), continue; end
        w_true = w_true(1,:);
        w_hat  = w_hat(1,:);
        L = min(numel(w_true), numel(w_hat));
        w_true = w_true(1:L);
        w_hat  = w_hat(1:L);

        % --- Find 0.5 crossing for TRUE W_t ---
        sT = w_true - 0.5;
        zT = find(sT(1:end-1).*sT(2:end) <= 0 & ~isnan(sT(1:end-1)) & ~isnan(sT(2:end)), 1, 'first');
        if isempty(zT), continue; end
        iT = zT; jT = zT + 1;

        tT1 = (iT-1) * dt_ms; tT2 = (jT-1) * dt_ms;
        wT1 = w_true(iT);     wT2 = w_true(jT);

        denomT = (wT2 - wT1);
        if ~isfinite(denomT) || abs(denomT) < eps, continue; end
        fracT = (0.5 - wT1) / denomT;
        t_true_ms = tT1 + fracT * (tT2 - tT1);

        % --- Find 0.5 crossing for ESTIMATED W_t ---
        sH = w_hat - 0.5;
        zH = find(sH(1:end-1).*sH(2:end) <= 0 & ~isnan(sH(1:end-1)) & ~isnan(sH(2:end)), 1, 'first');
        if isempty(zH), continue; end
        iH = zH; jH = zH + 1;

        tH1 = (iH-1) * dt_ms; tH2 = (jH-1) * dt_ms;
        wH1 = w_hat(iH);      wH2 = w_hat(jH);

        denomH = (wH2 - wH1);
        if ~isfinite(denomH) || abs(denomH) < eps, continue; end
        fracH = (0.5 - wH1) / denomH;
        t_hat_ms = tH1 + fracH * (tH2 - tH1);

        % --- Temporal offset (ms)
        Dt_ms(end+1,1) = t_hat_ms - t_true_ms;

        % --- Value of each at the true switch time (nearest sample)
        it = max(1, min(round(t_true_ms/dt_ms)+1, L));
        W_hat_at_true(end+1,1) = w_hat(it);
        W_true(end+1,1)        = w_true(it);

        % --- Saturation flag
        sat_flag(end+1,1) = (w_hat(it) <= sat_eps) || (w_hat(it) >= 1 - sat_eps);

        i1 = max(1, iT - halfwin);
        i2 = min(L, iT + 1 + halfwin);

        local_res = w_hat(i1:i2) - w_true(i1:i2);
        sigma_w   = std(local_res, 0, 'omitnan');

        % local slope of w_hat around iT (central difference, robust at edges)
        if iT <= 1
            slope_hat_local = (w_hat(2) - w_hat(1)) / dt_ms;
        elseif iT >= L
            slope_hat_local = (w_hat(L) - w_hat(L-1)) / dt_ms;
        else
            slope_hat_local = (w_hat(min(L,iT+1)) - w_hat(max(1,iT-1))) / (2*dt_ms);
        end

        sigma_t = sigma_w / max(abs(slope_hat_local), eps);
        CI_time_lo(end+1,1) = -1.96 * sigma_t;
        CI_time_hi(end+1,1) =  1.96 * sigma_t;
    end

    stats_all{m} = struct( ...
        'W_hat_at_true', W_hat_at_true, ...
        'W_true', W_true, ...
        'Dt_ms', Dt_ms, ...
        'sat', sat_flag, ...
        'CI_time_lo', CI_time_lo, ...
        'CI_time_hi', CI_time_hi);

    keep = ~sat_flag;
    stats_nosat{m} = struct( ...
        'W_hat_at_true', W_hat_at_true(keep), ...
        'W_true', W_true(keep), ...
        'Dt_ms', Dt_ms(keep), ...
        'sat', sat_flag(keep), ...
        'CI_time_lo', CI_time_lo(keep), ...
        'CI_time_hi', CI_time_hi(keep));
end


%% Plot bias around simulated Wt = 0.5

models = {'p','pv','pf','pvi','pif','pvf'};   % controller classes
summary = table('Size',[numel(models) 8], ...
                'VariableTypes',{'string','double','double','double','double','double','double','double'}, ...
                'VariableNames',{'model','median_Wthat','PIlow_Wthat','PIhigh_Wthat', ...
                                  'bias_med','median_lag','PIlow_lag','PIhigh_lag'});

figure('Color','w','Position',[100 100 1100 500]);

for k = 1:numel(models)

    summary.model(k) = models{k};

    % pull arrays for this model
    wt_true_all = stats_nosat{k,1}.W_true;          % true W_t at each trial's true switch
    wt_hat_all  = stats_nosat{k,1}.W_hat_at_true;    % recovered W_t evaluated at the true switch index

    % keep only trials near true W_t = 0.5 (exclude "no switch" or degenerate cases)
    keep = (wt_true_all > 0.49 & wt_true_all < 0.51);
    wt_true_keep = wt_true_all(keep);
    wt_hat_keep  = wt_hat_all(keep);


    % --- recovered W_t distribution stats ---
    med_Wthat   = median(wt_hat_keep, 'omitnan');
    PI95_Wthat  = quantile(wt_hat_keep, [0.025 0.975]);   
    bias_med    = med_Wthat - 0.5; 
    mean_Wthat   = mean(wt_hat_keep, 'omitnan');
    bias_mean    = mean_Wthat - 0.5; 


    x = wt_hat_keep;                      
    SEM = std(x)/sqrt(length(x));               
    ts = tinv([0.025  0.975],length(x)-1);      
    CI_Wthat = mean_Wthat + ts*SEM;       

    % store in summary
    summary.median_Wthat(k)  = med_Wthat;
    summary.mean_Wthat(k)  = mean_Wthat;
    summary.PIlow_Wthat(k)   = PI95_Wthat(1);
    summary.PIhigh_Wthat(k)  = PI95_Wthat(2);
    summary.CIlow_Wthat(k)  = CI_Wthat(1);
    summary.CIhigh_Wthat(k)  = CI_Wthat(2);
    summary.bias_med(k)      = bias_med;
    summary.bias_mean(k)      = bias_mean;
 
    
    %   PLOTS
    % Recovered W_t when true W_t ~ 0.5
    nexttile(k); hold on;
    [f,xi] = ksdensity(wt_hat_keep);           
    plot(xi, f, 'b', 'LineWidth', 2);

    yl = ylim;
    % vertical lines
    xline(0.5,        'r', 'LineWidth', 1);   % true W_t
    xline(mean_Wthat,  'b', 'LineWidth', 1);   % mean recovered
    xline(CI_Wthat(1),  'k--', 'LineWidth', 1);   
    xline(CI_Wthat(2),  'k--', 'LineWidth', 1);   

    xlabel('Recovered W_t');
    ylabel('Density');
    title([models{k} ' bias = ' num2str(bias_med)]);
    legend({'Density','95% PI','True W_t=0.5','Mean'}, ...
           'Location','best'); legend boxoff;
    set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 12);

   
end
sgtitle('Switch recovery bias ');

%% Plot time lag when simulated Wt = 0.5

models = {'p','pv','pf','pvi','pif','pvf'};
figure('Color','w','Position',[100 100 1100 500]); tiledlayout(1,numel(models),'Padding','compact','TileSpacing','compact');

for k = 1:numel(models)
    wt_true_all = stats_nosat{k,1}.W_true;
    dtms_all    = stats_nosat{k,1}.Dt_ms;

    keep = (wt_true_all > 0.49 & wt_true_all < 0.51);
    dtms_keep = dtms_all(keep);
    dtms_keep = dtms_keep(~isnan(dtms_keep));

    if isempty(dtms_keep)
        nexttile; title([models{k} ' (no data)']); axis off; continue;
    end

    med_lag  = median(dtms_keep, 'omitnan');

    % Mean uncertainty (t CI for mean)
    mu_lag = mean(dtms_keep, 'omitnan');
    n      = numel(dtms_keep);
    SEM    = std(dtms_keep, 0, 'omitnan') / sqrt(n);
    tScore = tinv(0.975, max(n-1,1));
    CImean = [mu_lag - tScore*SEM, mu_lag + tScore*SEM];

    % Plot
    nexttile; hold on;
    histogram(dtms_keep, 'BinWidth', 50, 'EdgeColor','none');
    xline(med_lag,    'r',  'LineWidth', 1.5);          % median
    xline(CImean(1),  'k--','LineWidth', 1.2);          % mean CI low
    xline(CImean(2),  'k--','LineWidth', 1.2);          % mean CI high

    xlabel('Lag (ms): recovered - true'); ylabel('Count');
    title(sprintf('%s: median=%.0fms', models{k}, med_lag));
    xlim([-2000 2000]);
    legend({'Lag dist','Median','95% CI(mean)','95% CI(mean)'}, 'Location','best'); legend boxoff;
    set(gca,'TickDir','out','Color','none','Box','off','FontName','Arial','FontSize',12);
end

sgtitle('Switch recovery temporal lag');
