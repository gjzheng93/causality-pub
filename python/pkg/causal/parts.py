


import numpy as np

from scipy import stats

from statsmodels.tsa.stattools import lagmat2ds as lagmat2ds
from statsmodels.tools.tools import add_constant as add_constant
from statsmodels.regression.linear_model import OLS as OLS





def fit_regression(dta, dtaown, dtajoint):

    # Run ols on both models without and with lags of second variable
    res2down = OLS(dta[:, 0], dtaown).fit()
    res2djoint = OLS(dta[:, 0], dtajoint).fit()
  
    return res2down, res2djoint
    
def f_test(res2down, res2djoint, lag):
    
    result = {} 

    # Granger Causality test using ssr (F statistic)
   
    fgc1 = (res2down.ssr - res2djoint.ssr) / res2djoint.ssr / lag * res2djoint.df_resid
            
    result['ssr_ftest'] = (fgc1,
                           stats.f.sf(fgc1, lag, res2djoint.df_resid),
                           res2djoint.df_resid, lag) 
                           
    return result



def get_lagged_data(x, lag, addconst, verbose):

    x = np.asarray(x)

    if x.shape[0] <= 3 * lag + int(addconst):
        raise ValueError("Insufficient observations. Maximum allowable "
                         "lag is {0}".format(int((x.shape[0] - int(addconst)) /
                                                 3) - 1))
    
    if verbose:
        print('\nGranger Causality')
        print('number of lags (no zero)', lag)

    # create lagmat of both time series
    dta = lagmat2ds(x, lag, trim='both', dropex=1)

    #add constant
    if addconst:
        dtaown = add_constant(dta[:, 1:(lag + 1)], prepend=False, has_constant = 'add')
        dtajoint = add_constant(dta[:, 1:], prepend=False, has_constant = 'add')
    else:

        dtaown = dta[:, 1:(lag+1)]
        dtajoint = dta[:, 1:]
        
        
    return dta, dtaown, dtajoint