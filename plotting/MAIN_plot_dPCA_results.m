%% Plot the outputs of the dPCA decomposition along switch direction 
clear; close all

load('dPCASwitchDirection_SyntheticACC.mat');
%load('/Users/assiachericoni/Documents/PYTHON/ControllerModeling-main_old/PacManData/dPCA_Decoding/dPCA_FiltRew24_e05_95_allRegion_10thrashold.mat')

%% Interaction dynamics = W(t) trajectory (st)

t = 16.67*(1:30) - 15;   % 30 bins
cfg.prewin = 14; % samples before switch
cfg.dt_ms  = 1/60;  
cfg.dt_ms  = 1000/60;                    % ms per bin

t    = 1:30;
t_ms = (t - cfg.prewin - 1) * cfg.dt_ms; % time relative to switch (s)


figure;
Zacc_st = Z_acc.st(1,:,:);                 % [1 x T x 2]
plot(t_ms, squeeze(Zacc_st).', 'LineWidth', 2);
title(sprintf('ACC — W(t) (st)  %.2f%% var', expvar_acc.st(1)*100));
xlabel('bins (warped)'); ylabel('Projection'); box on
set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);

% figure;
% Zhpc_st = Z_hpc.st(1,:,:);
% plot(t_ms, squeeze(Zhpc_st).', 'LineWidth', 2);
% title(sprintf('HPC — W(t) (st)  %.2f%% var', expvar_hpc.st(1)*100));
% xlabel('bins (warped)'); ylabel('Projection'); box on
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% 
% figure;
% Zofc_st = Z_ofc.st(1,:,:);
% plot(t_ms, squeeze(Zofc_st).', 'LineWidth', 2);
% title(sprintf('OFC — W(t) (st)  %.2f%% var', expvar_ofc.st(1)*100));
% xlabel('bins (warped)'); ylabel('Projection'); box on
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);

%% E/F: Condition-independent time (t)
figure;
Zacc_t = Z_acc.t(1,:,:);
plot(t_ms, squeeze(Zacc_t).', 'LineWidth', 2);
title(sprintf('ACC — Condition-independent (t)  %.2f%% var', expvar_acc.t(1)*100));
xlabel('Time (ms, warped)'); ylabel('Projection'); box on
set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 12);
%ylim([-0.3 0.31])


% figure;
% Zhpc_t = Z_hpc.t(1,:,:);
% 
% plot(t_ms, squeeze(Zhpc_t).', 'LineWidth', 2);
% title(sprintf('HPC — Condition-independent (t)  %.2f%% var', expvar_hpc.t(1)*100));
% xlabel('Time (ms, warped)'); ylabel('Projection'); box on
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 12);
% %ylim([-0.3 0.31])
% 
% figure;
% Zofc_t = Z_ofc.t(1,:,:);
% 
% plot(t_ms, squeeze(Zofc_t).', 'LineWidth', 2);
% title(sprintf('OFC — Condition-independent (t)  %.2f%% var', expvar_ofc.t(1)*100));
% xlabel('Time (ms, warped)'); ylabel('Projection'); box on
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 12);
%ylim([-0.3 0.31])

%% Static switch (s) — shows time-independent offset by direction
figure;
Zacc_s = Z_acc.s(1,:,:);
plot(t, squeeze(Zacc_s).', 'LineWidth', 2);
title(sprintf('ACC — Static switch (s)  %.2f%% var', expvar_acc.s(1)*100));
xlabel('Time (ms, warped)'); ylabel('Projection'); box on

% figure;
% Zhpc_s = Z_hpc.s(1,:,:);
% plot(t, squeeze(Zhpc_s).', 'LineWidth', 2);
% title(sprintf('HPC — Static switch (s)  %.2f%% var', expvar_hpc.s(1)*100));
% xlabel('Time (ms, warped)'); ylabel('Projection'); box on
% 
% figure;
% Zofc_s = Z_ofc.s(1,:,:);
% plot(t, squeeze(Zofc_s).', 'LineWidth', 2);
% title(sprintf('OFC — Static switch (s)  %.2f%% var', expvar_ofc.s(1)*100));
% xlabel('Time (ms, warped)'); ylabel('Projection'); box on


%% B: Percent variance per marginalization (sum over components)
% area   = {'HPC'; 'HPC'; 'HPC'; 'ACC'; 'ACC'; 'ACC'; 'OFC'; 'OFC'; 'OFC'};
% pctype = {'Wt'; 'Interaction'; 'CI'; 'Wt'; 'Interaction'; 'CI'; 'Wt'; 'Interaction'; 'CI'};
% perc   = [ sum(expvar_hpc.st)*100; ...   % W(t) = st
%            sum(expvar_hpc.s)*100;  ...   % Interaction
%            sum(expvar_hpc.t)*100; ...
%            sum(expvar_acc.st)*100; ...
%            sum(expvar_acc.s)*100;  ...
%            sum(expvar_acc.t)*100;...
%            sum(expvar_ofc.st)*100; ...
%            sum(expvar_ofc.s)*100;  ...
%            sum(expvar_ofc.t)*100;];
% 
% 
% PERCVAR = table(area, pctype, perc, 'VariableNames', {'area','pctype','PERC'});
% P = unstack(PERCVAR, "PERC", "pctype");  % columns named CI, Interaction, Wt
% 
% figure('Color','w');
% cats = categorical(P.area);
% bar(cats, [P.Wt, P.Interaction, P.CI], 'grouped');
% ylabel('Variance dPCA (%)'); 
% legend({'W(t)','Interaction','Condition-independent'}, 'Location','northoutside','Orientation','horizontal');
% box on

%% C: Percent variance per marginalization (1st components)

area   = {'ACC'; 'ACC'; 'ACC'};
pctype = {'Wt'; 'Interaction'; 'CI'};

perc   = [expvar_acc.st(1)*100; ...
    expvar_acc.s(1)*100;  ...
    expvar_acc.t(1)*100];

PERCVAR = table(area, pctype, perc, 'VariableNames', {'area','pctype','PERC'});
P = unstack(PERCVAR, "PERC", "pctype");  % columns named CI, Interaction, Wt

figure('Color','w');
cats = categorical(P.area);
bar(cats, [P.Wt, P.Interaction, P.CI], 'grouped');
ylabel('Variance dPCA (%)'); 
legend({'W(t)','Interaction','Condition-independent'}, 'Location','northoutside','Orientation','horizontal');

set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Arial', 'FontSize', 12);

%% ACC
obs  = accuracy_acc_dpca{1};         % 1 x T  (observed accuracy per bin)
null = accuracy_acc_perm_dpca{2};    % R x T  (R=1000 permutations)


pvals = mean(null >= obs, 1);        % 1 x T
sig   = pvals < 0.05;

figure; hold on
plot(t_ms, obs, 'k','LineWidth',2); 
plot(t_ms, mean(null,1), 'Color',[0.5 0.5 0.5],'LineWidth',1.5); % still plotting mean of set-averaged null
yline(0.5,'--k'); title('ACC');
xlabel('time (ms)'); ylabel('Decoding accuracy Wt')

ypos = max(obs)+0.03;
plot(t_ms(sig), repmat(ypos,1,sum(sig)), 'k_', 'LineWidth',2);
%xlim([1 30]);

