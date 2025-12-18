from patsy import dmatrix

def get_design_mat(glm_input_df):
    # terms to include in design matrix
    design_terms = ['slope_pursuit_dist', 'average_pursuit_dist', 
                    'slope_ignore_dist', 'average_ignore_dist', 
                    'rel_value', 'rel_time_s']

    # limit kinematic interactions to just one variable per interaction max
    limit_terms = {'slope_pursuit_dist', 'average_pursuit_dist',
                   'slope_ignore_dist', 'average_ignore_dist'}
    
    # create design matrix with main effects and 3rd-order interactions
    formula = f"{' + '.join(design_terms)} + ({' + '.join(design_terms)})**3"
    design_matrix = dmatrix(formula, data=glm_input_df, return_type='dataframe')
    included_terms = []
    for col in design_matrix.columns:
        col_terms = set(col.split(':'))
        num_excluded = len(limit_terms & col_terms)
        if num_excluded <= 1:
            included_terms.append(col)
    
    # filter to included terms
    design_matrix = design_matrix[included_terms].copy()
    # add patient id and label term
    design_matrix['pt_id'] = glm_input_df['pt_id']
    design_matrix['y'] = (~glm_input_df['control']).astype(int)
    return design_matrix


def get_fixed_formulas():
    terms = ['slope_pursuit_dist', 'average_pursuit_dist', 
                'slope_ignore_dist', 'average_ignore_dist', 
                'rel_value', 'rel_time_s']
    
    others = " + ".join([t for t in terms if t != "rel_value"])
    fixed_formula = f"y ~ {others} + rel_value*({others})"
    return fixed_formula