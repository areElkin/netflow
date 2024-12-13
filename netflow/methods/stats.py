import itertools
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

from multiprocessing import Pool, cpu_count

def t_test(values1, values2, alternative='two-sided', **kwargs):
    """ Calculate the T-test for the means of *two independent* samples of scores.
    
    This is a test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances by default.
    
    Computed via ``scipy.stats.ttest_ind``.
    
    Parameters
    ----------
    values1, values2 : array-like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default), which can be
        specified in ``kwargs``.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
    
        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.
    kwarags : `dict`
        Key-word arguments passed to ``scipy.stats.ttest_ind``.

    Returns
    -------
    p_value : `float`
        The p-value.
    """
    t_stat, p_value = ttest_ind(values1, values2, alternative=alternative, **kwargs)
    return p_value


def mann_whitney_u_test(values1, values2, alternative='two-sided', **kwargs):
    """ Perform the Mann-Whitney U rank test on two independent samples.

    The Mann-Whitney U test is a nonparametric test of the null hypothesis
    that the distribution underlying sample `x` is the same as the
    distribution underlying sample `y`. It is often used as a test of
    difference in location between distributions.
    
    Computed via ``scipy.stats.mannwhitneyu``.
    
    Parameters
    ----------
    values1, values2 : array-like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default), which can be
        specified in ``kwargs``.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
    
        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.
    kwarags : `dict`
        Key-word arguments passed to ``scipy.stats.mannwhitneyu``.

    Returns
    -------
    p_value : `float`
        The p-value.
    """
    u_stat, p_value = mannwhitneyu(values1, values2, alternative=alternative, **kwargs)
    return p_value


def wilcoxon_signed_rank_test(values1, values2=None, alternative='two-sided', **kwargs):
    """ The Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Computed via ``scipy.stats.wilcoxon``.

    Parameters
    ----------
    values1 : array-like
        Either the first set of measurements (in which case ``y`` is the second
        set of measurements), or the differences between two sets of
        measurements (in which case ``y`` is not to be specified.)  Must be
        one-dimensional.
    values2 : array-like
        Optional. Either the second set of measurements (if ``x`` is the first set of
        measurements), or not specified (if ``x`` is the differences between
        two sets of measurements.)  Must be one-dimensional.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
    
        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.
    kwarags : `dict`
        Key-word arguments passed to ``scipy.stats.wilcoxon``.

    Returns
    -------
    p_value : `float`
        The p-value.
    """    
    w_stat, p_value = wilcoxon(values1, values2, alternative=alternative, **kwargs)
    return p_value


# Function to choose and perform the statistical test
def perform_stat_test(values1, values2, test_type, **kwargs):
    if test_type == 't-test':
        return t_test(values1, values2, **kwargs)
    elif test_type == 'MWU':
        return mann_whitney_u_test(values1, values2, **kwargs) 
    elif test_type == 'wilcoxon':
        return wilcoxon_signed_rank_test(values1, values2, **kwargs)
    else:
        raise ValueError("Invalid test type. Choose 't-test', 'MWU', or 'wilcoxon'.")
    

def stat_test(df1, df2, test='MWU', alpha=0.05, method='fdr_bh', **kwargs):
    """ Perform statistical test between datasets with FWER correction.

    The statistical tests are Computed via ``scipy.stats``.

    Parameters
    ----------
    df1, df2 : `pandas.DataFrame`
        The measurements, where rows are features and columns are observations.
        The dataframes  must have the same number of features (rows). 
        If ``test='wilcoxon'``, they must also have the same number of
        observationas (columns).
    test : `str`
        The statistical test that should be performed. Options are:

        - 'MWU' : Mann Whitney-U Test (default).
        - 't-test' : T-test
        - 'wilcoxon' : Wilcoxon Signed Rank Test
    alpha : `float`
        The family-wise error rate (FWER), should be between 0 and 1.
    method : `str`
        Method for multiple test correction, default='fdr_bh'.

        Options:

        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative) 
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)
    kwargs : `dict`
        Key-word arguments passed to ``scipy.stats`` for performing the
        statistical test.

    Returns
    -------
    record : `pandas.DataFrame`
        Record of each feature, p-value, and corrected p-value.
    """
    assert (df1.index == df2.index).all(), "DataFrames must have the same index."
    if test == 'wilcoxon':
        assert (df1.columns == df2.columns).all(), "DataFrames must have the same observations for performing the Wilcoxon Signed Rank Test."
    
    p_values = [perform_stat_test(df1.loc[k].values.astype(float), df2.loc[k].values.astype(float), test, **kwargs) for k in df1.index]

    corrected_p_values = multipletests(p_values, alpha=alpha, method=method)[1]

    record = pd.DataFrame(data=[p_values, corrected_p_values], columns=df1.index.copy(), index=['p-value', 'corrected p-value']).T
    return record


def compute_spearman(row_x, row_y):
    """Compute Spearman correlation with each row in Y for a given row in X.
    
    Parameters
    ----------
    row_x, row_y : array_like
        1-D arrays representing multiple observations of a single variable. 
        The correlation is computed between ``row_x`` and ``row_y``.
        
    Returns
    -------
    correlation : `float`
        The correlation.
    p_value : `float`
        The p-value.
    """    
    correlation, pvalue = spearmanr(row_x, row_y)
    return correlation, pvalue


def compute_spearman_parallel(X, Y, num_processors=None, chunksize=None):
    """Compute Spearman correlation between each row of X and all rows of Y in parallel.
    
    Parameters
    ----------
    X, Y : `pandas.DataFrame` 
        Dataframes containing multiple variables and observations. 
        Each row represents a variable and each column is an observation of each variable.
        `X` and `Y` must have the same number of columns (i.e., the same observations) 
        but they need not have the same number of variables. 
    num_processors : `int`
        Number of processors to use. Defaults to None (uses all available).
        
    Returns
    -------
    correlations : `dict`
        The resulting correlations in the form ``{index_row_X: {index_row_Y: corr}}``
    p_values : `dict`
        The resulting p_values in the form ``{index_row_X: {index_row_Y: p_value}}``
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimension miss-match, expected X and Y to have the same number of columns")
    
    # ensure the columns of X and Y are ordered the same
    Y = Y[X.columns]
    
    if num_processors is None:
        num_processors = cpu_count()  # Use all available processors

    with Pool(processes=num_processors) as pool:                    
        result = pool.starmap(compute_spearman, itertools.product(X.values, Y.values), chunksize=chunksize)

    # correlations = {k: v[0] for k, v in zip(itertools.product(X.index, Y.index), result)}
    # p_values = {k: v[1] for k, v in zip(itertools.product(X.index, Y.index), result)}

    correlations = {k: {} for k in X.index}
    p_values = {k: {} for k in X.index}
    for (x_idx, y_idx), (rho, p_val) in zip(itertools.product(X.index, Y.index), result):
        correlations[x_idx][y_idx] = rho
        p_values[x_idx][y_idx] = p_val

    return correlations, p_values


def compute_pearson(row_x, row_y):
    """Compute Pearson correlation with each row in Y for a given row in X.
    
    Parameters
    ----------
    row_x, row_y : array_like
        1-D arrays representing multiple observations of a single variable. 
        The correlation is computed between ``row_x`` and ``row_y``.
        
    Returns
    -------
    correlation : `float`
        The correlation.
    p_value : `float`
        The p-value.
    """    
    correlation, pvalue = pearsonr(row_x, row_y)
    return correlation, pvalue


def compute_pearson_parallel(X, Y, num_processors=None, chunksize=None):
    """Compute Pearson correlation between each row of X and all rows of Y in parallel.
    
    Parameters
    ----------
    X, Y : `pandas.DataFrame` 
        Dataframes containing multiple variables and observations. 
        Each row represents a variable and each column is an observation of each variable.
        `X` and `Y` must have the same number of columns (i.e., the same observations) 
        but they need not have the same number of variables. 
    num_processors : `int`
        Number of processors to use. Defaults to None (uses all available).
        
    Returns
    -------
    correlations : `dict`
        The resulting correlations in the form ``{index_row_X: {index_row_Y: corr}}``
    p_values : `dict`
        The resulting p_values in the form ``{index_row_X: {index_row_Y: p_value}}``
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimension miss-match, expected X and Y to have the same number of columns")
    
    # ensure the columns of X and Y are ordered the same
    Y = Y[X.columns]
    
    if num_processors is None:
        num_processors = cpu_count()  # Use all available processors
    
    with Pool(processes=num_processors) as pool:                    
        result = pool.starmap(compute_pearson, itertools.product(X.values, Y.values), chunksize=chunksize)

    # correlations = {k: v[0] for k, v in zip(itertools.product(X.index, Y.index), result)}
    # p_values = {k: v[1] for k, v in zip(itertools.product(X.index, Y.index), result)}

    correlations = {k: {} for k in X.index}
    p_values = {k: {} for k in X.index}
    for (x_idx, y_idx), (rho, p_val) in zip(itertools.product(X.index, Y.index), result):
        correlations[x_idx][y_idx] = rho
        p_values[x_idx][y_idx] = p_val

    return correlations, p_values
