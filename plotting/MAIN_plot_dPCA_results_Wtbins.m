%% Plot the outputs of the dPCA decomposition along 5 Wt bins

clear; close all

load('dPCA_WtBins_SyntheticACC.mat');

% rwd0 = load('/Users/assiachericoni/Documents/PYTHON/ControllerModeling-main_old/PacManData/dPCA_Decoding/dPCA_rwd0AllRegions_control.mat');
% 
% rwd24 = load('/Users/assiachericoni/Documents/PYTHON/ControllerModeling-main_old/PacManData/dPCA_Decoding/dPCA_rwd24AllRegions_control.mat');


%% Heatmaps for dPCA decomposition along 5 Wt bins

cfg.prewin = 14; % samples before switch
cfg.dt_ms  = 1/60;             
t    = 1:30;
t_ms = (t - cfg.prewin - 1) * cfg.dt_ms; % time relative to switch (s)

wt_edges   = linspace(0,1,6);
wt_centers = (wt_edges(1:end-1) + wt_edges(2:end))/2;

% st: Wt x time interaction as a heatmap
STacc0 = squeeze(Z_acc.st(1,:,:)); 

STacc0z = zscore(STacc0);


figure; imagesc(t_ms, wt_centers, (STacc0z)); axis xy; axis square
title(sprintf('ACC st (%.1f%% var)', 100*expvar_acc.st(1)));
xlabel('time'); ylabel('W_t'); box off; set(gca,'TickDir','out');
set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
colormap(bluered);
colorbar;
caxis([-1.5 1.5]);  

%% Here the original code to plot heatmaps and scatter for all the brain regions as in Figure 4F, G, H

% figure('Color','w','Position',[100 100 1600 500]); 
% subplot(1 ,3, 1)
% scatter(STacc0z(:), SThpc0z(:)); axis square
% [r_AH_eq, p_AH_eq] = corr(STacc0z(:), SThpc0z(:));
% xlabel('acc st'); xlabel('hpc st');
% title(['HPC vs ACC equal reward r = ' num2str(r_AH_eq)])
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% 
% subplot(1 ,3, 2)
% scatter(STacc0z(:), STofc0z(:)); axis square
% [r_AO_eq, p_AO_eq] = corr(STacc0z(:), STofc0z(:));
% xlabel('acc st'); xlabel('ofc st');
% title(['OFC vs ACC equal reward r = ' num2str(r_AO_eq)])
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% 
% subplot(1 ,3, 3); 
% scatter(STofc0z(:), SThpc0z(:)); axis square
% [r_OH_eq, p_OH_eq] = corr(STofc0z(:), SThpc0z(:));
% xlabel('ofc st'); xlabel('hpc st');
% title(['HPC vs OFC equal reward r = ' num2str(r_OH_eq)])
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% saveas(gcf, 'dPCARwd0_correlations_across_regions.svg');
% 
% 
% %% overlayed
% figure('Color','w','Position',[100 100 1600 500]); hold on
% 
% % Get default MATLAB colors
% colors = cool(3);   % 3 nice distinct colors
% 
% ms = 120;  % marker size
% 
% % Symbols for each region pair
% sym = {'o', 's', 'd'};   % circle, square, diamond
% 
% % ACC vs HPC
% scatter(STacc0z(:), SThpc0z(:), ms, ...
%     'Marker', sym{1}, ...
%     'MarkerFaceColor', colors(1,:), ...
%     'MarkerEdgeColor', colors(1,:), ...
%     'MarkerFaceAlpha', 0.4);
% 
% % ACC vs OFC
% scatter(STacc0z(:), STofc0z(:), ms, ...
%     'Marker', sym{2}, ...
%     'MarkerFaceColor', colors(2,:), ...
%     'MarkerEdgeColor', colors(2,:), ...
%     'MarkerFaceAlpha', 0.4);
% 
% % HPC vs OFC
% scatter(STofc0z(:), SThpc0z(:), ms, ...
%     'Marker', sym{3}, ...
%     'MarkerFaceColor', colors(3,:), ...
%     'MarkerEdgeColor', colors(3,:), ...
%     'MarkerFaceAlpha', 0.4);
% 
% axis square
% xlabel('Z_1 (region A)')
% ylabel('Z_1 (region B)')
% set(gca, 'TickDir','out', 'Color','none', 'Box','off', ...
%          'FontName','Helvetica', 'FontSize', 14);
% 
% legend({'ACC vs HPC','ACC vs OFC','HPC vs OFC'}, ...
%        'Location','best', 'Box','off');
% saveas(gcf, 'dPCARwd0_scatter.svg');
% 
% %% Heatmaps for different reward condition
% 
% % st: Wt x time interaction as a heatmap
% STacc24 = squeeze(rwd24.Z_acc.st(1,:,:));  
% SThpc24 = squeeze(rwd24.Z_hpc.st(1,:,:)); 
% STofc24 = squeeze(rwd24.Z_ofc.st(1,:,:)); 
% 
% STacc24z = zscore(STacc24);
% SThpc24z = zscore(SThpc24);
% STofc24z = zscore(STofc24);
% 
% figure('Color','w','Position',[100 100 1600 500]);
% subplot(1,3,1); imagesc(t_ms, wt_centers, (STacc24z)); axis xy; axis square
% title(sprintf('ACC st (%.1f%% var)', 100*rwd24.expvar_acc.st(1)));
% xlabel('time'); ylabel('W_t'); colormap(parula); box off; set(gca,'TickDir','out');
% colormap(bluered);
% colorbar;
% caxis([-1.5 1.5]);  % Center at zero
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% 
% subplot(1,3,2); imagesc(t_ms, wt_centers, (SThpc24z)); axis xy; axis square
% title(sprintf('HPC st (%.1f%% var)', 100*rwd24.expvar_hpc.st(1)));
% xlabel('time'); ylabel('W_t'); colormap(parula); box off; set(gca,'TickDir','out');
% colormap(bluered);
% colorbar;
% caxis([-1.5 1.5]);  % Center at zero
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% 
% subplot(1,3,3); imagesc(t_ms, wt_centers, (STofc24z)); axis xy; axis square
% title(sprintf('OFC st (%.1f%% var)', 100*rwd24.expvar_ofc.st(1)));
% xlabel('time'); ylabel('W_t'); colormap(parula); box off; set(gca,'TickDir','out');
% colormap(bluered);
% colorbar;
% caxis([-1.5 1.5]);  % Center at zero
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% 
% sgtitle('different reward')
% saveas(gcf, 'dPCARwd24_heatmaps.svg');
% 
% figure('Color','w','Position',[100 100 1600 500]);
% subplot(1 ,3, 1)
% scatter(STacc24z(:), SThpc24z(:)); axis square
% [r_AH_diff, p_AH_diff] = corr(STacc24z(:), SThpc24z(:));
% xlabel('acc st'); xlabel('hpc st');
% title(['HPC vs ACC different reward r = ' num2str(r_AH_diff)])
% 
% subplot(1 ,3, 2)
% scatter(STacc24z(:), STofc24z(:)); axis square
% [r_AO_diff, p_AO_diff] = corr(STacc24z(:), STofc24z(:));
% xlabel('acc st'); xlabel('ofc st');
% title(['OFC vs ACC different reward r = ' num2str(r_AO_diff)])
% 
% subplot(1 ,3, 3)
% scatter(STofc24z(:), SThpc24z(:)); axis square
% [r_OH_diff, p_OH_diff] = corr(STofc24z(:), SThpc24z(:));
% xlabel('ofc st'); xlabel('hpc st');
% title(['HPC vs OFC different reward r = ' num2str(r_OH_diff)])
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% saveas(gcf, 'dPCARwd24_correlations_across_regions.svg');
% 
% %%
% figure('Color','w','Position',[100 100 1600 500]); hold on
% 
% % Get default MATLAB colors
% colors = cool(3);   % 3 nice distinct colors
% 
% ms = 120;  % marker size
% 
% % Symbols for each region pair
% sym = {'o', 's', 'd'};   % circle, square, diamond
% 
% % ACC vs HPC
% scatter(STacc24z(:), SThpc24z(:), ms, ...
%     'Marker', sym{1}, ...
%     'MarkerFaceColor', colors(1,:), ...
%     'MarkerEdgeColor', colors(1,:), ...
%     'MarkerFaceAlpha', 0.4);
% 
% % ACC vs OFC
% scatter(STacc24z(:), STofc24z(:), ms, ...
%     'Marker', sym{2}, ...
%     'MarkerFaceColor', colors(2,:), ...
%     'MarkerEdgeColor', colors(2,:), ...
%     'MarkerFaceAlpha', 0.4);
% 
% % HPC vs OFC
% scatter(STofc24z(:), SThpc24z(:), ms, ...
%     'Marker', sym{3}, ...
%     'MarkerFaceColor', colors(3,:), ...
%     'MarkerEdgeColor', colors(3,:), ...
%     'MarkerFaceAlpha', 0.4);
% 
% axis square
% xlabel('Z_1 (region A)')
% ylabel('Z_1 (region B)')
% set(gca, 'TickDir','out', 'Color','none', 'Box','off', ...
%          'FontName','Helvetica', 'FontSize', 14);
% 
% legend({'ACC vs HPC','ACC vs OFC','HPC vs OFC'}, ...
%        'Location','best', 'Box','off');
% saveas(gcf, 'dPCARwd24_scatter.svg');
% 
% %% Comparison within brain region
% 
% figure('Color','w','Position',[100 100 1600 500]);
% subplot(1 ,3, 1)
% scatter(STacc0z(:), STacc24z(:)); axis square
% [r_diff, p_diff]   = corr(STacc0z(:), STacc24z(:))
% xlabel('st rwd0'); xlabel('st rwd24');
% title(['ACC different vs equal reward r = ' num2str(r_diff)])
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% 
% subplot(1 ,3, 2)
% scatter(SThpc0z(:), SThpc24z(:)); axis square
% [r_diff, p_diff]   = corr(SThpc0z(:), SThpc24z(:))
% xlabel('st rwd0'); xlabel('st rwd24');
% title(['HPC different vs equal reward r = ' num2str(r_diff)])
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% 
% subplot(1 ,3, 3)
% scatter(STofc0z(:), STofc24z(:)); axis square
% [r_diff, p_diff]   = corr(STofc0z(:), STofc24z(:))
% xlabel('st rwd0'); xlabel('st rwd24');
% title(['OFC different vs equal reward r = ' num2str(r_diff)])
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% 
% % saveas(gcf, 'dPCARwdComparison_across_regions.svg');
% 
% %% Assessing significance 
% N_eq   = numel(STacc0z);   % same N for all maps
% N_diff = numel(STacc24z);
% 
% [z_AH, p_AH] = compare_correlations(r_AH_eq, r_AH_diff, N_eq, N_diff);
% 
% fprintf('ACC–HPC: equal vs diff reward\n');
% fprintf('  r_eq   = %.3f\n', r_AH_eq);
% fprintf('  r_diff = %.3f\n', r_AH_diff);
% fprintf('  z      = %.3f,  p = %.3g\n\n', z_AH, p_AH);
% 
% %ACC–OFC, HPC–OFC:
% [z_AO, p_AO] = compare_correlations(r_AO_eq, r_AO_diff, N_eq, N_diff);
% fprintf('ACC–OFC: equal vs diff reward\n');
% fprintf('  r_eq   = %.3f\n', r_AO_eq);
% fprintf('  r_diff = %.3f\n', r_AO_diff);
% fprintf('  z      = %.3f,  p = %.3g\n\n', z_AO, p_AO);
% 
% [z_OH, p_OH] = compare_correlations(r_OH_eq, r_OH_diff, N_eq, N_diff);
% fprintf('OFC–HPC: equal vs diff reward\n');
% fprintf('  r_eq   = %.3f\n', r_OH_eq);
% fprintf('  r_diff = %.3f\n', r_OH_diff);
% fprintf('  z      = %.3f,  p = %.3g\n\n', z_OH, p_OH);
