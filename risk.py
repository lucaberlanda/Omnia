import numpy as np
import pandas as pd

from time import time
from numpy import diag, sqrt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.stattools import pacf
from Omnia.core_primitive import business_day
from scipy.stats import spearmanr, kendalltau


def timeit(f):
    def timed(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()

        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return timed


frequency_converter = {
    'freq_D': 1,
    'freq_W': 12,
    'freq_M': 30,
    'freq_Q': 120,
    'freq_Y': 252,

    'freq_<Day>': 1,
    'freq_<BusinessDay>': 1,
    'Week: weekday=6': 12,

}

catch_optimization_failed = {}


def partial_corr_distance_measure(starting_set, nlags=40, alpha=None):
    return 2 - 2 * pacf(starting_set, nlags=nlags, method='ywunbiased', alpha=alpha)


def standardize_standard_deviation_time(data_frequency_starting='D', data_frequency_output='Y'):
    """

    compute the conversion number to take a series in one frequency to another one

    :param data_frequency_starting: str.  starting frequency
    :param data_frequency_output: str. output frequency
    :return: float. the conversion number
    """

    end_freq = frequency_converter['freq_' + str(data_frequency_output)]
    start_freq = frequency_converter['freq_' + str(data_frequency_starting)]

    return np.sqrt(end_freq / start_freq)


def standard_deviation(x, do_not_standardize=False, data_frequency_starting='D', data_frequency_output='Y'):
    """

    compute the standard deviation optionally standardizing (annualizing) to the chosen dateset

    :param x: pandas series, ndarray. returns
    :param do_not_standardize: bool. whether to standardize or not
    :param data_frequency_starting: str. starting frequency
    :param data_frequency_output:  str. output frequency
    :return: standrda deviation
    """

    if do_not_standardize:
        time_standardizer = 1
    else:
        time_standardizer = standardize_standard_deviation_time(data_frequency_starting,
                                                                data_frequency_output)

    return np.std(x) * time_standardizer


def marginal_percentage_risk_contribution(w, omega):
    """

    return the marginal percentage risk contribution of a given set of weights

    :param w: list/ndarray. the set of weights
    :param omega: pandas dataframe. var cov matrix
    :return: ndarray. the PCTR array
    """

    if isinstance(w, list):
        w = np.array(w)

    if isinstance(w, pd.Series):
        new_w = pd.DataFrame(np.diag(w.values), index=w.index, columns=w.index)
        num = new_w.dot(omega).dot(w)
        den = w.dot(omega).dot(w)
        return num / den

    if isinstance(w, np.ndarray):
        new_w = pd.DataFrame(np.diag(w), index=omega.index, columns=omega.index)
        num = new_w.dot(omega).dot(w)
        den = w.dot(omega).dot(w)
        return num / den

    num = np.diag(w) @ omega @ w
    den = w @ omega @ w

    return num / den


def compute_covariance(returns, method_used='standard', level_division=0):
    """

    :param omega:
    :param method_used:
    :return:
    """

    lw = LedoitWolf()

    if level_division != 0:
        if method_used != 'standard':
            to_ret = returns.groupby(level=level_division, axis=1).apply(lambda x: lw.fit(x).covariance_)
            to_ret.columns = to_ret.columns.droplevel(level=0)
            return to_ret
        else:
            to_ret = returns.groupby(level=level_division, axis=1).apply(lambda x: x.cov())
            to_ret.columns = to_ret.columns.droplevel(level=0)
            return to_ret

    if method_used != 'standard':
        return pd.DataFrame(lw.fit(returns).covariance_, index=returns.columns, columns=returns.columns)
    else:
        return returns.cov()


def compute_corr_std_matrix(returns, do_not_standardize=False, data_frequency_starting='D', data_frequency_output='Y'):
    """

    Compute a correlation matrix with on the diagonal the standardized standard deviation given a specific frequency

    :param returns: pandas  dataframe. return data table
    :param do_not_standardize: bool. if true standard deviations are not standardized
    :param data_frequency_starting: str. the frequency of the data before compting the var-cov matrix
    :param data_frequency_output: str.the frequency of the data after compting the var-cov matrix
    :return: pandas dataframe. a corr_std_matrix
    """

    corr_matrix = returns.corr()
    new_diagonal = np.sqrt(np.diag(returns.cov().as_matrix()))
    if do_not_standardize:
        time_standardizer = 1
    else:
        time_standardizer = standardize_standard_deviation_time(data_frequency_starting,
                                                                data_frequency_output)
    rng = np.arange(len(corr_matrix))
    corr_matrix[rng, rng] = new_diagonal * time_standardizer
    return corr_matrix


def optimize_with_risk_budget_spinu(risk_budgets, omega, eps=0.00001, max_iter=100, do_standardize=True):
    """ Compute weights corresponding to the risk budgets
        If omega is a vector, and not a matrix, then it is interpreted as
        a volatility vector and the closed-form solution is provided.

        Implementing Spinu's algorithm as described in:
        Griveau-Billion, Richard and Roncalli,
        A Fast Algorithm for Computing High-Dimensional Risk Parity Portfolios
        (September 1, 2013).
        Available at SSRN: https://ssrn.com/abstract=2325255

    Args:
        risk_budgets: risk budgets
        omega:  covariance matrix as numpy array or array of standard deviations
        eps: acceptable accuracy on risk budgets (by default, 1bp)
        max_iter: maximum number of iterations before stopping the algorithm

    Returns:
        portfolio weights as numpy array
    """

    if len(omega) != len(risk_budgets):
        print('Risk budgets and covariance matrix have different sizes')
        return None

    if np.any(np.linalg.eigvalsh(omega) < 0):
        print('The covariance matrix is not semi positive definite')
        return None

    if not isinstance(risk_budgets, np.ndarray):
        risk_budgets = np.asarray(risk_budgets)

    # Solve the problem where correlations are not taken into consideration,
    # i.e. omega is an array of standard deviations
    if len(omega.shape) == 1:
        y = np.sqrt(risk_budgets) / omega
        return y / sum(y)

    # Normalize risk budgets so they can be compared to PCTR
    if do_standardize:
        b = [x / sum(risk_budgets) for x in risk_budgets]
    else:
        b = risk_budgets

    # D: main diagonal of covariance matrix
    D = np.diag(np.diag(omega))
    # C: correlation matrix
    C = np.sqrt(np.linalg.inv(D)) @ omega @ np.sqrt(np.linalg.inv(D))

    # Initialize algorithm
    u = np.ones(len(omega))
    x = u / np.sqrt(u @ omega @ u)
    l_star = 0.95 * (3 - np.sqrt(5)) * 0.5

    for i in np.arange(max_iter):
        u = C @ x - b @ np.diag(1 / x)
        H = C + np.diag(b / (x * x))
        Delta_x = np.linalg.solve(H, u)
        l = np.sqrt(u @ Delta_x)
        if l > l_star:
            delta = max(np.abs(Delta_x / x))
            x -= Delta_x / (1 + delta)
        elif l > eps:
            x -= Delta_x
        else:
            break

    inv_sigma = 1 / np.sqrt(np.diag(D))
    w = inv_sigma * x / sum(inv_sigma * x)

    max_error = max(np.abs(marginal_percentage_risk_contribution(w, omega) - risk_budgets))
    if max_error > eps:
        return None
    else:
        return w


def optimize_with_risk_budget_standard(budget, omega, bnds, cons, maxiter=4000, tol=1e-13, do_standardize=True):
    """

    compute the weights to match the risk budgets given element wise and aggregate constraints

    :param budget: list. the matching budgets
    :param omega: pandas dataframe. a var cov matrix
    :param bnds: dict.
    :param cons: dict.
    :return: ndarray. the resulting budget
    """

    def fun(w):
        return sum((marginal_percentage_risk_contribution(w, omega) - budget) ** 2)

    if len(omega) != len(budget):
        return None

    if do_standardize:
        budget = [x / sum(budget) for x in budget]

    if len(omega.shape) == 1:
        y = np.sqrt(budget) / omega
        return y / sum(y)
    else:
        s = np.sqrt(np.diag(omega))
        y = np.sqrt(budget) / s
        w0 = y / sum(y)
        res = minimize(fun, w0, bounds=np.array(bnds), constraints=cons, tol=tol, options={'maxiter': maxiter})
        if res.success:
            return res.x
        else:
            print('Optimization Failed')
            return None


def expected_shortfall(x):
    """

    compute the expected shortfall metric

    :param x:
    :return:
    """
    return x


def drawdown(xs, also_dates=False, ts=False):
    """
    compute the drawdown of a given timeseries

    :param xs: ndarray/pandas timeseries
    :param also_dates:
    :param ts: bool; if True returns the
    :return: float. the drawdown value

    """

    if isinstance(xs, pd.Series):
        xs = xs.dropna()
        _xs = xs.values
    if isinstance(xs, pd.DataFrame):
        xs = xs.dropna()
        _xs = xs.values.flatten()

    if _xs.shape[0] == 0:
        if also_dates:
            return np.nan, np.nan, np.nan
        else:
            return np.nan

    dd_ts = _xs / np.maximum.accumulate(_xs)

    if ts:
        return pd.Series(dd_ts - 1, index=xs.index, name=xs.name)

    i = np.argmin(dd_ts)  # end of the period
    if i == 0:
        return 0
    else:
        j = np.argmax(_xs[:i])
        if also_dates:
            st = xs.index[j]
            end = xs.index[i]
            return abs(_xs[i] / _xs[j] - 1), st, end
        else:
            return abs(_xs[i] / _xs[j] - 1)


def time_to_recovery(xs, end_dd, recovery_value):
    start_ttr = business_day(end_dd, False, shift=1)
    if isinstance(xs, pd.Series):
        xs = xs.dropna().truncate(before=start_ttr)
        _xs = xs.values
    if isinstance(xs, pd.DataFrame):
        xs = xs.dropna().truncate(before=start_ttr)
        _xs = xs.values.flatten()

    i = np.argmax(_xs >= recovery_value)  # end of the period
    if i == 0:
        return None
    else:
        return xs.index[i] - start_ttr


def compute_recovery_period(xs):
    """

    Peak to Peak recovery period

    :param xs: ndarray/pandas timeseries
    :return: int. the days

    """

    if isinstance(xs, pd.Series):
        xs = xs.dropna()
        xs = xs.values
    if isinstance(xs, pd.DataFrame):
        xs = xs.dropna()
        xs = xs.values.flatten()

    # drawdown position
    i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period

    if i == 0:
        return 0
    else:

        j = np.argmax(xs[:i])
        tgt = np.argmax(xs[i:] >= xs[j])

        if tgt == 0:
            return len(xs) - 1
        else:
            new_xs = pd.Series(xs)
            peak_reached_again = new_xs[(new_xs.index > i) & (new_xs >= xs[j])].index[0]
            return peak_reached_again - j


def drawdown_vectorized(xs):
    """
    compute the drawdown of a given timeseries

    :param xs: ndarray/pandas timeseries
    :return: float. the drawdown value
    """

    i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period
    if i == 0:
        return 0
    else:
        j = np.argmax(xs[:i])
        return abs(xs[i] / xs[j] - 1)


def compute_standard_rb_with_constraint(
        df_ret,
        peso,
        method_compute_cov,
        maxiter=4000,
        tol=1e-13,
        relax_negative_constraint=False,
        custom_bounds=None,
        do_not_use_correlations=False,
        precompiled_variance_covariance_structure=None,
        do_standardize=True,
        upper_bound=1
):
    actual_ret_data = df_ret.loc[:, peso.index]

    if precompiled_variance_covariance_structure is not None:
        df_ret_cov = precompiled_variance_covariance_structure.loc[actual_ret_data.columns, actual_ret_data.columns]
    else:

        if do_not_use_correlations:
            df_ret_cov_raw = actual_ret_data.std() * (252 ** 0.5)
            df_ret_cov = pd.DataFrame(
                np.diag(df_ret_cov_raw.values), index=df_ret_cov_raw.index, columns=df_ret_cov_raw.index)
        else:
            if relax_negative_constraint:
                df_ret_cov = compute_covariance(actual_ret_data, method_compute_cov)
            else:
                df_ret_cov = compute_covariance(actual_ret_data, method_compute_cov).applymap(
                    lambda x: 0 if x < 0 else x)

    if isinstance(df_ret_cov.index, pd.MultiIndex):
        df_ret_cov = df_ret_cov.sort_index(level=0, axis=0, sort_remaining=True).sort_index(
            level=0, axis=1, sort_remaining=True)
        if isinstance(peso, pd.Series):
            peso = peso.to_frame().sort_index(level=0, axis=1, sort_remaining=True).iloc[:, 0]
        else:
            peso = peso.sort_index(level=0, axis=1, sort_remaining=True)
    else:
        df_ret_cov = df_ret_cov.sort_index().sort_index(axis=1)
        peso = peso.sort_index()

    if custom_bounds is not None:
        g = custom_bounds.loc[peso.index]
    else:
        g = pd.DataFrame({
            'lower': np.repeat([0], len(peso.index)),
            'upper': np.repeat([upper_bound], len(peso.index))},
            index=peso.index)

    sum_w = 1 if peso.sum() > 1 else peso.sum()
    c_ = [{'type': 'eq', 'fun': lambda w: sum(w) - sum_w}]

    c = optimize_with_risk_budget_standard(
        peso.values.flatten(), df_ret_cov, g, c_, maxiter=maxiter, tol=tol, do_standardize=do_standardize)

    if c is not None:
        return pd.Series(c, index=peso.index)
    else:
        return peso


def compute_spinu_rb_with_constraint(
        df_ret,
        peso,
        method_compute_cov,
        maxiter=100,
        eps=0.0001,
        relax_negative_constraint=False,
        do_not_use_correlations=False,
        precompiled_variance_covariance_structure=None,
        do_standardize=True
):
    actual_ret_data = df_ret.loc[:, peso.index]

    if precompiled_variance_covariance_structure is not None:
        df_ret_cov = precompiled_variance_covariance_structure.loc[actual_ret_data.columns, actual_ret_data.columns]
    else:
        if do_not_use_correlations:
            df_ret_cov_raw = actual_ret_data.std() * (252 ** 0.5)
            df_ret_cov = pd.DataFrame(
                np.diag(df_ret_cov_raw.values), index=df_ret_cov_raw.index, columns=df_ret_cov_raw.index)
        else:
            df_ret_cov = compute_covariance(actual_ret_data, method_compute_cov)

            if not relax_negative_constraint:
                df_ret_cov = df_ret_cov.applymap(lambda x: 0 if x < 0 else x)

    if isinstance(df_ret_cov.index, pd.MultiIndex):
        df_ret_cov = df_ret_cov.sort_index(level=0, axis=0, sort_remaining=True).sort_index(
            level=0, axis=1, sort_remaining=True)
        if isinstance(peso, pd.Series):
            peso = peso.to_frame().sort_index(level=0, axis=1, sort_remaining=True).iloc[:, 0]
        else:
            peso = peso.sort_index(level=0, axis=1, sort_remaining=True)
    else:
        df_ret_cov = df_ret_cov.sort_index().sort_index(axis=1)
        peso = peso.sort_index()

    c = optimize_with_risk_budget_spinu(
        peso.values.flatten(), df_ret_cov, eps=eps, max_iter=maxiter, do_standardize=do_standardize)

    if c is not None:
        return pd.Series(c, index=peso.index)
    else:
        return peso


def barone_adesi_indicator_variant(weights, var_cov_matrix, std_dev_ptf):
    try:
        variance_part = pd.Series(diag(var_cov_matrix), index=[var_cov_matrix.index])
        sum_weighted_variance_part = variance_part.mul(weights ** 2).sum()
        num = sqrt(sum_weighted_variance_part)
        return num / std_dev_ptf
    except:
        variance_part = pd.Series(diag(var_cov_matrix), index=var_cov_matrix.index)
        sum_weighted_variance_part = variance_part.mul(weights ** 2).sum()
        num = sqrt(sum_weighted_variance_part)
        return num / std_dev_ptf


def barone_adesi_indicator(weights, var_cov_matrix, std_dev_ptf):
    std_part = pd.Series(sqrt(diag(var_cov_matrix)), index=var_cov_matrix.index)
    num = std_part.mul(weights).sum()
    return std_dev_ptf / num


def compute_rank_correlations(rank1, rank2, which='spearman'):
    """

    Compute rank correlations

    :param rank1: pandas Series
    :param rank2: pandas Series
    :param which: str or list
    :param do_rank: bool
    :param rank_ascending: bool
    :return:

    https://en.wikipedia.org/wiki/Rank_correlation
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    """

    if isinstance(which, str):
        which = [which]

    results = {}

    df = pd.concat({'rank1': rank1, 'rank2': rank2}, axis=1)

    if 'correlation' in which:
        results['correlation'] = pd.Series({'indicator': df.corr().iloc[0, 1], 'p_value': np.nan})

    if 'spearman' in which:
        rho, pval = spearmanr(df.loc[:, 'rank1'], df.loc[:, 'rank2'])
        results['spearman'] = pd.Series({'indicator': rho, 'p_value': pval})

    if 'kendall' in which:
        tau, pval = kendalltau(df.loc[:, 'rank1'], df.loc[:, 'rank2'])
        results['kendall'] = pd.Series({'indicator': tau, 'p_value': pval})

    return pd.concat(results, axis=1)


#  Optimization

def MCR(vCov, w, ptf_std):
    mcr = w.dot(vCov) / ptf_std
    return mcr


def CR(mcr, w):
    cr = mcr * w
    return cr


def risk_decomposition(vCov, w):
    """
    : vCoc: pd.DataFrame. Varcov matrix
    : w: pd.Series. Weights.

    From Varcov and weights to risk decomposition metrics:
        -   Marginal risk contribution (MCR)
        -   Risk contribution (CR)
        -   Percentage risk contribution (PCR)
    """

    ptf_std = np.sqrt(w.dot(vCov).dot(w))
    mcr = MCR(vCov, w, ptf_std)
    cr = CR(mcr, w)
    # todo check if cr.sum() == ptf_std
    pcr = cr / ptf_std
    return mcr, cr, pcr


def PCR(w, omega):
    cr = CR(w, omega)
    return cr.div(np.sqrt(w.dot(omega.dot(w))), axis=0)


class RiskBudgeting:

    def __init__(self, varcov, method='naive_risk_parity', risk_budget=None, w=None):
        """
        :param ris
        :param method: str; it can be:

            - 'naive_risk_parity'
            - 'risk_parity'
            - 'naive_risk_budgeting'
            - 'risk_budgeting'
            - 'naive_marginal_risk_contribution'
            - 'marginal_risk_contribution'

            naive means diagonal varcov matrix, i.e. the correlations are set to 0

        :param risk_budget
        :param w


        """

        self.varcov = varcov
        self.method = method
        self.rb = risk_budget
        self.w = w

    @staticmethod
    def risk_parity_objective(w, pars, method='min_max'):

        omega = pars[0]
        pcr = PCR(w, omega)

        if method == 'min_max':
            J = pcr.max() - pcr.min()  # sum of squared error
        elif method == 'mean_squared':
            risk_target = pars[1]
            J = sum(np.square(pcr - risk_target.T))  # sum of squared error
        return J

    @staticmethod
    def total_weight_constraint(x):
        return np.sum(x) - 1.0

    @staticmethod
    def long_only_constraint(x):
        return x

    def optimize(self):
        if self.method == 'naive_risk_parity':
            rb = pd.Series(1 / len(self.varcov.index), index=self.varcov.index)
            std = np.sqrt(pd.Series(np.diag(self.varcov), index=self.varcov.index))
            inv_std = std.rdiv(1)
            w = inv_std / inv_std.sum()
            return w, rb

        if self.method == 'naive_risk_budgeting':
            std = np.sqrt(pd.Series(np.diag(self.varcov), index=self.varcov.index))
            w = self.rb.div(std).div(self.rb.div(std).sum())
            return w, self.rb

        elif self.method == 'naive_marginal_risk_contribution':
            sigma = np.sqrt(pd.Series(np.diag(self.varcov), index=self.varcov.index))
            mrc_raw = self.w.mul(sigma)
            mrc = mrc_raw / mrc_raw.sum()
            return self.w, mrc

        elif self.method == 'marginal_risk_contribution':
            mrc = marginal_percentage_risk_contribution(self.w.dropna(), self.varcov)
            return self.w, mrc

        elif self.method == 'risk_parity':
            # todo finish
            from scipy.optimize import minimize
            w0 = np.array([0.25, 0.25, 0.25, 0.25])
            x_t = np.array([0.25, 0.25, 0.25, 0.25])  # your risk budget percent of total portfolio risk (equal risk)

            # Equality constraint means that the constraint function result
            # is to be zero whereas inequality means that it is to be non-negative.
            cons = ({'type': 'eq', 'fun': self.total_weight_constraint},
                    {'type': 'ineq', 'fun': self.long_only_constraint})

            res = minimize(self.risk_parity_objective, w0,
                           args=[self.ris.pct_change().cov(), x_t],
                           method='SLSQP',
                           constraints=cons,
                           options={'disp': True})

            w_rb = np.asmatrix(res.x)


def risk_budgeting_rolling(w_df, ris, method='naive_risk_parity', risk_budget=None, days=365):
    """
    :param w_df: pandas DataFrame. It is the weights dataframe.
    """
    w_dict = {}
    rb_dict = {}
    for dt, row in w_df.iterrows():
        st, end = dt - pd.to_timedelta(days, unit='D'), dt  # todo make param out of 365
        ris_to_use = ris.loc[:, row.dropna().index]
        varcov = ris_to_use.truncate(before=st, after=end).pct_change().cov()
        dt_rb = risk_budget.loc[dt, :] if risk_budget is not None else None
        risk_budgeting = RiskBudgeting(varcov, method=method, risk_budget=dt_rb, w=row)
        date_w, date_rb = risk_budgeting.optimize()
        w_dict[dt] = date_w
        rb_dict[dt] = date_rb

    W = pd.concat(w_dict, axis=1)
    RB = pd.concat(rb_dict, axis=1)
    return W, RB


class ConvictionPropagation:

    def propagation_base(self):
        return self.rb.mul(self.centre_at * (1 - self.gamma) + self.gamma * self.convictions.fillna(self.fill_with))

    def propagate(self):
        rb_raw = self.propagation_base()
        tot_rb_raw = rb_raw.sum(axis=1)
        if not self.with_cash:
            rb = rb_raw.div(tot_rb_raw, axis=0)
            return rb
        else:
            cash_unconstrained_rb = rb_raw.div(tot_rb_raw.apply(lambda x: max(x, 0.5)), axis=0)
            total_rb_constrained = cash_unconstrained_rb.sum(axis=1).apply(lambda x: max(1 - self.max_cash, x))
            cash_constrained_rb = cash_unconstrained_rb.div(cash_unconstrained_rb.sum(axis=1), axis=0).mul(
                total_rb_constrained, axis=0)

            return cash_constrained_rb

    def __init__(
            self,
            rb,
            gamma,
            convictions,
            centre_at=0.5,
            fill_with=0.5,
            with_cash=False,
            max_cash=0.2
    ):

        """
        :param rb: ,
        :param gamma: ,
        :param convictions: ,
        :param fill_with: float;
        :param centre_at: float
        """

        self.rb = rb
        self.gamma = gamma
        self.convictions = convictions
        self.fill_with = fill_with
        self.centre_at = centre_at
        self.with_cash = with_cash
        self.max_cash = max_cash
