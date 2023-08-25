from functools import lru_cache
import numpy as np
import cvxpy as cvx
import ot
import heapq
from scipy.optimize import linprog
from scipy import sparse
import scipy

from ._logging import logger


linprog_status = {0:'Optimization proceeding nominally.',
                  1:'Iteration limit reached.',
                  2:'Problem appears to be infeasible.',
                  3:'Problem appears to be unbounded.',
                  4:'Numerical difficulties encountered.'}



def optimal_transportation_distance(x, y, d, solvr=None):    
    """ Compute the optimal transportation distance (OTD) of the given density distributions by CVXPY.

    Parameters
    ----------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    Returns
    -------
    m : float
        Optimal transportation distance.
    """
    rho = cvx.Variable((len(y), len(x)))  # the transportation plan rho
    # objective function d(x,y) * rho * x, need to do element-wise multiply here
    obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))
    # \sigma_i rho_{ij}=[1,1,...,1]
    source_sum = cvx.sum(rho, axis=0, keepdims=True)
    # constrains = [rho * x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    constrains = [rho @ x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    prob = cvx.Problem(obj, constrains)
    if solvr is None:
        m = prob.solve()
    else:
        m = prob.solve(solver=solvr)
        # m = prob.solve(solver='ECOS', abstol=1e-6,verbose=True)
        # m = prob.solve(solver='OSQP', max_iter=100000,verbose=False)
    return m


def OTD(x, y, d, solvr=None, flag=None):
    """ Compute the optimal transportation distance (OTD) of the given density distributions
    trying first with POT package and then by CVXPY.

    Parameters
    ----------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.
    flag : str, optional
        Optional flag included in logger output to identify inputs.
        

    Returns
    -------
    m : float
        Optimal transportation distance.
    """

    flag = "" if flag is None else flag+": "

    try:
        wasserstein_distance, lg = ot.emd2(x, y, d, log=True)
        if lg['warning'] is not None:
            logger.msg(f"{flag}POT library failed: warning = {lg['warning']}, retry with explicit computation")
            wasserstein_distance = optimal_transportation_distance(x, y, d)
    except cvx.error.SolverError:
        logger.msg(f"{flag}OTD failed, retry with SCS solver")
        wasserstein_distance = optimal_transportation_distance(x, y, d, solvr='SCS')
    return wasserstein_distance


                   
