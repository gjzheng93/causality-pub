
# implements the package version

from . import parts


def grangercausalitytests(dta, dtaown, dtajoint, lag, addconst=True, verbose=False):
    
    res2down, res2djoint = parts.fit_regression(dta, dtaown, dtajoint)
    
    result = parts.f_test(res2down, res2djoint, lag)
    
    p_value = result['ssr_ftest'][1]

    return p_value, res2down, res2djoint   