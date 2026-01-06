import dill as pickle
from ChangeOfMind.Figures import Figure_Maker



kwargs = {}
kwargs['C_svm_acc'] = 0.01
kwargs['C_svm_hpc'] = 0.1
kwargs['smoothing'] = 150
kwargs['ncomps_sep']=0
kwargs['prewin_behave'] = 18  # Specifies behavioral data window, extend for delayed GLM
kwargs['behavewin_behave'] = 10  # specifies post event behavioral window
kwargs['all_subjects'] = False
kwargs['plottype'] = 'decode'
kwargs['return_type'] = 'save'
Figure_Maker.Fig_4_EvidenceSubspace_emu_do(**kwargs)


kwargs = {}
kwargs['C_svm_acc'] = 0.01
kwargs['C_svm_hpc'] = 0.1
kwargs['smoothing'] = 100
kwargs['ncomps_sep']=0
kwargs['prewin_behave'] = 18  # Specifies behavioral data window, extend for delayed GLM
kwargs['behavewin_behave'] = 10  # specifies post event behavioral window
kwargs['all_subjects'] = False
kwargs['plottype'] = 'decode'
kwargs['return_type'] = 'save'
Figure_Maker.Fig_4_EvidenceSubspace_emu_do(**kwargs)


kwargs = {}
kwargs['C_svm_acc'] = 0.01
kwargs['C_svm_hpc'] = 0.1
kwargs['smoothing'] = 60
kwargs['ncomps_sep']=0
kwargs['prewin_behave'] = 14  # Specifies behavioral data window, extend for delayed GLM
kwargs['behavewin_behave'] = 15  # specifies post event behavioral window
kwargs['all_subjects'] = False
kwargs['plottype'] = 'decode'
kwargs['return_type'] = 'save'
Figure_Maker.Fig_4_EvidenceSubspace_emu_do(**kwargs)



kwargs = {}
kwargs['smoothing'] = 80
kwargs['cca_reps']=100
kwargs['cca_perm']=0
kwargs['win_smooth_cca']=2
kwargs['d_hpc']=3
kwargs['kmeans_hpc']=4
kwargs['d_acc']=3
kwargs['kmeans_acc']=5
kwargs['ncomps_sep']=0
kwargs['prewin_behave'] = 14  # Specifies behavioral data window, extend for delayed GLM
kwargs['behavewin_behave'] = 15  # specifies post event behavioral window
kwargs['all_subjects'] = False
kwargs['plottype'] = 'ldscluster'
kwargs['return_type'] = 'save'
kwargs['dowhiten'] = True

Figure_Maker.Fig_4_EvidenceSubspace_emu_do(**kwargs)




kwargs = {}
kwargs['C_svm_acc'] = 0.01
kwargs['C_svm_hpc'] = 0.1
kwargs['smoothing'] = 120
kwargs['cca_reps']=100
kwargs['cca_perm']=0
kwargs['win_smooth_cca']=2
kwargs['ncomps_sep']=0
kwargs['prewin_behave'] = 14  # Specifies behavioral data window, extend for delayed GLM
kwargs['behavewin_behave'] = 15  # specifies post event behavioral window
kwargs['all_subjects'] = False
kwargs['plottype'] = 'commsubspace_cca'
kwargs['return_type'] = 'save'
Figure_Maker.Fig_4_EvidenceSubspace_emu_do(**kwargs)


