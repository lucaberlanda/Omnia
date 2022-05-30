import inspect
import numpy as np
import pandas as pd

from logger import log
from traceback import print_tb
from log_colors import bcolors as bc


def exception_handler(func):

    def inner_function(*args, **kwargs):

        try:
            returning = func(*args, **kwargs)
            return returning

        except Exception as e:

            params_dict = {i: j for i, j in zip(inspect.getfullargspec(func).args, args)}
            log.error(bc.FAIL + f"{func.__name__}: %s - %s" % (params_dict, e) + bc.ENDC)
            print_tb(e.__traceback__)

    return inner_function


def rebase_at_x(df, at=100):
    if type(df) == pd.Series:
        df = df.dropna()
        df = df / df.iloc[0] * at
    else:
        df = df.apply(lambda x: x / x.dropna().values.tolist()[0] * at)
    return df


def sql_string(ids, is_string=False):
    if len(ids) > 1:
        return str(tuple(ids))
    else:
        if is_string:
            return "('" + str(ids[0]) + "')"
        else:
            return "(" + str(ids[0]) + ")"


def millify(n):
    import math
    millnames = ['', ' Thousand', ' Million', ' Billion', ' Trillion']
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.0f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


def max_drawdown(xs, also_dates=False, ts=False):
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


def top_n_drawdown(ri, n=3):
    dds_dict = {}
    for i in range(1, n + 1):
        ri.plot()

        dd_dict = {}
        df = ri.apply(lambda x: max_drawdown(x, True)).T
        df.columns = ["drawdown", 'start_date', 'end_date']

        dd = df.loc[:, 'drawdown']
        st = df.loc[:, 'start_date']
        end = df.loc[:, 'end_date']

        dd_dict['dd'] = dd.values[0]
        dd_dict['start'] = pd.to_datetime(st.values[0]).strftime('%Y-%m-%d')
        dd_dict['end'] = pd.to_datetime(end.values[0]).strftime('%Y-%m-%d')
        dd_dict['duration_in_days'] = (pd.to_datetime(end.values[0]) - pd.to_datetime(st.values[0])).days

        dds_dict[i] = dd_dict

        ret = ri.pct_change()
        ret = ret[~((ret.index > df.values[0][1]) & (ret.index < df.values[0][2]))]
        ri = (1 + ret).fillna(1).cumprod()

    return dds_dict


def get_z_score(df, f, **kwargs):
    if f is not None:
        z_score = df.sub(getattr(df, f)(**kwargs).mean(), axis=0).div(getattr(df, f)(**kwargs).std(), axis=0)
    else:
        z_score = df.sub(df.mean()).div(df.std())
    return z_score


def normalize_signal(df, how='expanding', **kwargs):
    if type(df) == pd.Series:
        df = df.to_frame()

    z_score = get_z_score(df, f=how, **kwargs)
    from scipy.stats import logistic
    signal = pd.DataFrame(logistic.cdf(z_score), index=df.index, columns=df.columns)
    return signal
