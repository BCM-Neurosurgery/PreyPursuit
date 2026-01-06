from ChangeOfMind.Figures import Figure_Maker
import dill as pickle
import pandas as pd

if __name__ == "__main__":


    do_partial=[True]
    do_warp=[False]
    do_cut_at_median=[True]

    extra_params={}
    emu_dpca=pd.DataFrame(columns=['outputs','do_warp','do_cut_at_median','do_partial','reg_level'])
    reg_level = [1e-2, 0.05, 0.1, 0.5]

    for i,reg in enumerate(reg_level):
        for partial in do_partial:
            for warp in do_warp:
                for cut_at_median in do_cut_at_median:
                    extra_params['reg_acc']=reg
                    extra_params['reg_hpc']=reg

                    outputs = Figure_Maker.Fig_3_demixedPCA_decoding_emu_do(extra_params,cut_at_median=cut_at_median,do_warp=warp,do_partial=partial,smoothing=100,all_subjects=False)

                    new_row = {
                    'outputs': outputs,
                        'do_warp': warp,
                        'do_cut_at_median': cut_at_median,
                        'do_partial': partial,
                        'reg_level': reg
                }

                    emu_dpca = pd.concat([emu_dpca, pd.DataFrame([new_row])], ignore_index=True)


    dpca_all={}
    dpca_all['emu']=emu_dpca

    with open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/' + 'dpcaEMUSMOOTH.pkl', 'wb') as f:
        pickle.dump(dpca_all, f)




 # reg_level=[1e-4,1e-2,1e-1,0.5]
    # do_partial=[True,False]
    # do_warp=[True,False]
    # do_cut_at_median=[True,False]
    #
    # extra_params={}
    #
    # nhp_dpca=pd.DataFrame(columns=['outputs','do_warp','do_cut_at_median','do_partial','reg_level'])
    # for i,reg in enumerate(reg_level):
    #     for partial in do_partial:
    #         for warp in do_warp:
    #             for cut_at_median in do_cut_at_median:
    #                 extra_params['reg_acc']=reg
    #                 extra_params['reg_pmd']=reg
    #
    #                 outputs=Figure_Maker.Fig_3_demixedPCA_decoding_nhp(extra_params,cut_at_median=cut_at_median,do_warp=warp,do_partial=partial,smoothing=80)
    #
    #                 new_row = {
    #                 'outputs': outputs,
    #                     'do_warp': warp,
    #                     'do_cut_at_median': cut_at_median,
    #                     'do_partial': partial,
    #                     'reg_level': reg
    #             }
    #
    #                 nhp_dpca = pd.concat([nhp_dpca, pd.DataFrame([new_row])], ignore_index=True)
    #
    #
    # extra_params={}
    # emu_dpca=pd.DataFrame(columns=['outputs','do_warp','do_cut_at_median','do_partial','reg_level'])
    #
    # for i,reg in enumerate(reg_level):
    #     for partial in do_partial:
    #         for warp in do_warp:
    #             for cut_at_median in do_cut_at_median:
    #                 extra_params['reg_acc']=reg
    #                 extra_params['reg_hpc']=reg
    #
    #                 outputs=Figure_Maker.Fig_3_demixedPCA_decoding_emu(extra_params,cut_at_median=cut_at_median,do_warp=warp,do_partial=partial,smoothing=120,all_subjects=False)
    #
    #                 new_row = {
    #                 'outputs': outputs,
    #                     'do_warp': warp,
    #                     'do_cut_at_median': cut_at_median,
    #                     'do_partial': partial,
    #                     'reg_level': reg
    #             }
    #
    #                 emu_dpca = pd.concat([emu_dpca, pd.DataFrame([new_row])], ignore_index=True)
    #
    #
    # dpca_all={}
    # dpca_all['nhp']=nhp_dpca
    # dpca_all['emu']=emu_dpca
    #
    #
    # with open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'+'dpca.pkl', 'wb') as f:
    #     pickle.dump(dpca_all, f)
    #
    #
    #
