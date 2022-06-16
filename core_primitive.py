import time
import numpy as np
import pandas as pd
import datetime as dt


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))

        return ret

    return wrap


def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))


# TODO this must be deleted when there won't be EIS anymore
def get_following_business_day(date, only_if_not_bd=True, num_days=1):
    if isinstance(date, str):
        date = pd.to_datetime(date)

    if only_if_not_bd:
        following_date = date + pd.tseries.offsets.BDay(num_days) if not is_business_day(date) else date
    else:
        following_date = date + pd.tseries.offsets.BDay(num_days)

    return following_date


def business_day(date, only_if_not_bd=False, shift=1):
    if isinstance(date, str):
        date = pd.to_datetime(date)

    if only_if_not_bd:
        following_date = date + pd.tseries.offsets.BDay(shift) if not is_business_day(date) else date
    else:
        following_date = date + pd.tseries.offsets.BDay(shift)

    return following_date


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + dt.timedelta(days=4)  # this will never fail
    return next_month - dt.timedelta(days=next_month.day)


def print_dict(to_print):
    for x in to_print:
        print(x, ':', to_print[x])


def get_year_to_date(reference_date):
    if type(reference_date) == str:
        reference_date = pd.to_datetime(reference_date)
    ytd_date = weekend_to_friday(pd.to_datetime(str(reference_date.year)) - pd.Timedelta(1, unit='d'))
    return ytd_date


def get_last_friday(reference_day='today', include_friday=False):
    """
    :param reference_day: date to compare to in order to get the last friday
    :param include_friday:
    :return: returns the last friday
    """
    today_day_of_the_week = pd.to_datetime(reference_day).dayofweek

    if not include_friday:
        if today_day_of_the_week >= 5:
            # for saturday and sunday
            days_delta = today_day_of_the_week - 4
        else:
            # for the other days
            days_delta = today_day_of_the_week + 3
    else:
        if today_day_of_the_week >= 4:
            # for saturday and sunday
            days_delta = today_day_of_the_week - 4
        else:
            # for the other days
            days_delta = today_day_of_the_week + 3

    end_date = pd.to_datetime(reference_day).normalize() - pd.Timedelta(days_delta, unit='d')

    return end_date


def import_fee_for_given_peer_group(conn, peer_group_name):
    ri_q = """select fund_extended.instrument_id, max_fee from ranking_results_extended inner join fund_extended on
    ranking_results_extended.instrument_id = fund_extended.instrument_id where
    ranking_results_extended.peer_group_name = '""" \
           + peer_group_name + """'  ORDER BY max_fee"""

    all_ri = pd.read_sql(ri_q, conn, index_col='instrument_id')
    return all_ri


def strategy_binning(strat, size=1, to=2):
    str_bins = bins_generator(size, to)
    binned_strat = pd.cut(strat, bins=str_bins)
    binned_strat = pd.Series([j.mid for j in list(binned_strat)], index=binned_strat.index)
    return binned_strat


def bins_generator(size=1, to=2):
    bins = [-size / 2]
    notch = size / 2
    bins.extend([notch])
    stop = False
    while not stop:
        notch = notch + size
        bins.extend([notch])
        if notch > to:
            stop = True

    return bins


def get_precise_intervals_on_dates(prices, start_date=None, end_date=None):
    """
    trim a/some timeseries exactly at the chosen intervals
    :param prices: pandas series or dataframe.
    :param start_date: str. starting date
    :param end_date: str. ending date
    :return: pandas series/dataframe. trimmed timeseries

    """

    if start_date is None and end_date is None:
        to_return = prices
    elif start_date is None:
        to_return = prices.truncate(after=pd.to_datetime(end_date))
    elif end_date is None:
        to_return = prices.truncate(before=pd.to_datetime(start_date))
    else:
        to_return = prices.truncate(before=pd.to_datetime(start_date), after=pd.to_datetime(end_date))

    if len(to_return) == 0:
        return pd.DataFrame(np.repeat(np.nan, len(prices.columns)), columns=prices.columns)
    else:
        return to_return


def weekend_to_friday(dt):
    if dt.dayofweek == 5:
        dt = dt - pd.Timedelta(1, unit='d')
    if dt.dayofweek == 6:
        dt = dt - pd.Timedelta(2, unit='d')
    return dt


def get_dates_for_reporting(end_date=None, ytd_date=None):
    if type(end_date) == str:
        end_date = pd.to_datetime(end_date)

    if type(ytd_date) == str:
        end_date = pd.to_datetime(ytd_date)

    dates = {'YTD': (ytd_date, end_date),
             '1y': (get_last_friday(end_date - pd.Timedelta(1 * 365, unit='d'), include_friday=True),
                    get_last_friday(end_date, include_friday=True)),
             '3y': (get_last_friday(end_date - pd.Timedelta(3 * 365, unit='d'), include_friday=True),
                    get_last_friday(end_date, include_friday=True))}

    return dates


def compute_returns_for_reporting(prices, dates_to_compute_performance_on, fill_na_with_the_average=True):
    dict_returns_unweighted = {}

    # it takes this kind of dates: [M, Y, s.i. etc.] (usually used for reporting, but also for sigmaTer tool)
    dates_for_reporting_dict = dates_to_compute_performance_on

    # all this process is done in order to compute all the metrics in the correct time span (0-1 years, 1-2y etc..
    # or Daily, Weekly, etc...)
    for period, dates_pair in dates_for_reporting_dict.items():

        if dates_pair[0].dayofweek == 6:
            start_date = dates_pair[0] - pd.DateOffset(2)
        elif dates_pair[0].dayofweek == 5:
            start_date = dates_pair[0] - pd.DateOffset(1)
        else:
            start_date = dates_pair[0]

        end_date = dates_pair[1]
        try:
            data_on_which_to_operate = get_precise_intervals_on_dates(prices, start_date=start_date, end_date=end_date)
            returns_per_period = (data_on_which_to_operate.iloc[-1, :] / data_on_which_to_operate.iloc[0, :]) - 1
            if start_date < prices.index.tolist()[0]:
                returns_per_period = pd.Series(np.NaN, returns_per_period.index)

            # fill NaN returns with the average of the other funds
            if fill_na_with_the_average:
                returns_per_period = returns_per_period.fillna(returns_per_period.mean())

            dict_returns_unweighted[period] = returns_per_period

        except:
            dict_returns_unweighted[period] = np.nan

    returns_for_reporting = pd.DataFrame(dict_returns_unweighted).dropna(how='all')
    return returns_for_reporting


def compute_vol_for_reporting(returns, scaling_coefficient, dates_for_reporting_dict):
    dict_vol_unweighted_for_reporting = {}

    for period, dates_pair in dates_for_reporting_dict.items():
        try:
            data_on_which_to_operate = get_precise_intervals_on_dates(returns, start_date=dates_pair[0],
                                                                      end_date=dates_pair[1])

            if len(data_on_which_to_operate) == 1:
                std_devs_per_period = pd.Series(1, index=returns.columns)
                dict_vol_unweighted_for_reporting[period] = std_devs_per_period
            else:
                std_devs_per_period = data_on_which_to_operate.std() * scaling_coefficient
                dict_vol_unweighted_for_reporting[period] = std_devs_per_period
        except:
            dict_vol_unweighted_for_reporting[period] = pd.Series(np.repeat(np.nan, len(returns.columns)),
                                                                  index=returns.columns)

    vols_for_reporting = pd.DataFrame(dict_vol_unweighted_for_reporting)

    return vols_for_reporting


def get_dates_for_performances(end_date=None, ytd_date=None):
    if end_date is None:
        end_date = pd.to_datetime('today').normalize()

    if type(end_date) == str:
        end_date = pd.to_datetime(end_date)

    end_date = weekend_to_friday(end_date)

    if ytd_date is None:
        ytd_date = get_year_to_date(end_date)
    else:
        if type(ytd_date) == str:
            ytd_date = pd.to_datetime(ytd_date)

    one_d_day = end_date - pd.Timedelta(1, unit='d')
    one_w_day = end_date - pd.Timedelta(7, unit='d')
    one_m_day = end_date - pd.Timedelta(30, unit='d')
    one_y_day = end_date - pd.Timedelta(1 * 365, unit='d')
    three_y_day = end_date - pd.Timedelta(3 * 365, unit='d')

    dates_dict = {'YTD': ytd_date,
                  '1W': one_w_day,
                  '1M': one_m_day,
                  '1Y': one_y_day,
                  '3Y': three_y_day,
                  '1D': one_d_day}

    dates_list = dates_dict.keys()

    for dt in dates_list:
        dates_dict[dt] = weekend_to_friday(dates_dict[dt])

    dates_couples = {}
    for period in dates_dict.keys():
        dates_couples[period] = (dates_dict[period], end_date)

    return dates_couples


def determine_quarter_to_date(target_quarter, year):
    if target_quarter == 1:
        return pd.to_datetime(year)
    elif target_quarter == 2:
        return pd.to_datetime(year) + pd.Timedelta(30 * 3, unit='D')
    elif target_quarter == 3:
        return pd.to_datetime(year) + pd.Timedelta(30 * 6, unit='D')
    else:
        return pd.to_datetime(year) + pd.Timedelta(30 * 9, unit='D')


def get_dates_for_performances_extended(inception_date, end_date=None):
    if end_date is None:
        end_date = pd.to_datetime('today').normalize()

    if type(end_date) == str:
        end_date = pd.to_datetime(end_date)

    if type(inception_date) == str:
        inception_date = pd.to_datetime(inception_date)

    end_date = weekend_to_friday(end_date)
    ytd_date = get_year_to_date(end_date)

    qtd_raw = determine_quarter_to_date(end_date.quarter, str(end_date.year))
    qtd_date_computable = pd.to_datetime(str(qtd_raw - pd.Timedelta(1, unit='D'))[:10])

    dates_dict = {'1D': max(end_date - pd.Timedelta(1, unit='d'), inception_date),
                  'YTD': max(ytd_date, inception_date),
                  'QTD': max(qtd_date_computable, inception_date),
                  '1W': max(end_date - pd.Timedelta(7, unit='d'), inception_date),
                  '2W': max(end_date - pd.Timedelta(14, unit='d'), inception_date),
                  '1M': max(end_date - pd.Timedelta(30, unit='d'), inception_date),
                  '3M': max(end_date - pd.Timedelta(3 * 30, unit='d'), inception_date),
                  '6M': max(end_date - pd.Timedelta(6 * 30, unit='d'), inception_date),
                  '1Y': max(end_date - pd.Timedelta(1 * 365, unit='d'), inception_date),
                  '1.5Y': max(end_date - pd.Timedelta(round(1.5 * 365), unit='d'), inception_date),
                  '2Y': max(end_date - pd.Timedelta(2 * 365, unit='d'), inception_date),
                  '3Y': max(end_date - pd.Timedelta(3 * 365, unit='d'), inception_date),
                  'ITD': inception_date}

    dates_list = dates_dict.keys()

    for dt in dates_list:
        dates_dict[dt] = weekend_to_friday(dates_dict[dt])

    dates_couples = {}
    for period in dates_dict.keys():
        dates_couples[period] = (dates_dict[period], end_date)

    whole_years = range(inception_date.year, end_date.year)
    for i in whole_years:
        end = business_day(pd.to_datetime(str(i) + '-12-31'), only_if_not_bd=True, shift=-1)
        st = max(get_year_to_date(end), inception_date)
        if st >= end:
            continue
        dates_couples[i] = (st, end)

    return dates_couples


def compute_volatility_measures(returns,
                                scaling_coefficient,
                                dates_for_ranking_dict,
                                dates_for_reporting_dict,
                                weights_standard_ranking,
                                what='vol'):
    """
    :param returns:
    :param scaling_coefficient: float. coefficient to scale the volatility
    :param dates_for_ranking_dict: dict. it takes this kind of dates: [(0y,1y), (1y,2y), ..] (usually used for rankings)
    :param dates_for_reporting_dict: dict. it takes this kind of dates: [M, Y, s.i. etc.] (usually used for reporting,
    but also for sigmaTer tool)
    :param what: str;
    :param weights_standard_ranking: weights used to assign importance to more recent vol volatility
    :return: vols_aggregated (weighted and summed), vols_per_period_unweighted (vol for each year), vols_for_reporting
    """

    dict_vol_unweighted = {}
    dict_vol_unweighted_for_reporting = {}

    # it takes this kind of dates: [(0y,1y), (1y,2y), etc] (usually used for rankings)
    link_freq_with_weights = pd.Series(weights_standard_ranking)

    # all this process is done in order to compute all the metrics in the correct time span (0-1 years, 1-2y etc..
    # or Daily, Weekly, etc...)

    # here the VOLATILITY CONCERNING RANKING is computed
    for period, dates_pair in dates_for_ranking_dict.items():
        if dates_pair[0] >= returns.index.tz_localize(None)[0]:
            data_on_which_to_operate = get_precise_intervals_on_dates(returns, start_date=dates_pair[0],
                                                                      end_date=dates_pair[1])

            # TODO: where the volatility is NaN, the volatility should be equal to the average (median?) volatility of
            #   the peer group scaled by a coefficient equal to the ratio of the vol of the NaN fund with the vol
            #   of the other funds. For now, NaN vols are set equal to the average vol over each period.
            std_devs_per_period = data_on_which_to_operate.std() * scaling_coefficient
            std_devs_per_period = std_devs_per_period.fillna(std_devs_per_period.median())  # median of the other funds
            dict_vol_unweighted[period] = std_devs_per_period

        else:
            dict_vol_unweighted[period] = pd.Series(np.nan, index=returns.columns)

    # here the VOLATILITY CONCERNING REPORTING is computed
    for period, dates_pair in dates_for_reporting_dict.items():
        if dates_pair[0] >= returns.index[0]:
            data_on_which_to_operate = get_precise_intervals_on_dates(returns, start_date=dates_pair[0],
                                                                      end_date=dates_pair[1])

            std_devs_per_period = data_on_which_to_operate.std() * scaling_coefficient
            dict_vol_unweighted_for_reporting[period] = std_devs_per_period
        else:
            dict_vol_unweighted_for_reporting[period] = pd.Series(np.nan, index=returns.columns)

    vols_per_period_unweighted = pd.DataFrame(dict_vol_unweighted)  # .dropna(how='all')
    vols_for_reporting = pd.DataFrame(dict_vol_unweighted_for_reporting)  # .dropna(how='all', axis=0)
    link_freq_with_weights.index = vols_per_period_unweighted.columns
    # vols_aggregated = vols_per_period_unweighted.dot(link_freq_with_weights)
    vols_aggregated = (vols_per_period_unweighted * link_freq_with_weights).sum(axis=1)
    vols_per_period_unweighted_columns = []

    if what == 'vol':
        vols_per_period_unweighted_columns = ['vol_0_1y', 'vol_1_2y', 'vol_2_3y', 'vol_3_4y']

    elif what == 'dd':
        vols_per_period_unweighted_columns = ['ddev_0_1y', 'ddev_1_2y', 'ddev_2_3y', 'ddev_3_4y']

    else:
        print('specify vol measure')

    vols_per_period_unweighted.columns = vols_per_period_unweighted_columns[
                                         :len(vols_per_period_unweighted.columns)]

    return vols_aggregated, vols_per_period_unweighted, vols_for_reporting


def compute_consistency(returns):
    results = {}
    for i in returns.columns:
        sliced_df = returns.loc[:, i]
        if returns.loc[:, i].isnull().all():
            results[i] = sliced_df.fillna(0)
        else:
            sliced_df = sliced_df.fillna(sliced_df.median()).fillna(1)
            try:
                results[i] = pd.qcut(sliced_df, 4, labels=[4, 3, 2, 1])
            except:
                results[i] = sliced_df.rank(ascending=False, method='max') / len(sliced_df.index) * 4

    consistency_metric = pd.DataFrame(results).sum(axis=1).apply(lambda x: round(x))

    return consistency_metric


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def weekly_resample(fund_prices):
    fund_prices.index = pd.to_datetime(fund_prices.index)
    fund_prices = fund_prices.resample('W-FRI').last()
    return fund_prices


def coerce_infinite(s):
    main_msk = np.isfinite(s)

    max_s = s[main_msk].max()
    min_s = s[main_msk].min()

    s[~main_msk] = s[~np.isfinite(s)].apply(lambda x: max_s if x > 0 else min_s)

    if s.isnull().all():
        s = s.fillna(1)

    return s


def dummy_cut(original_data, q_labels, top_percentage_slice, name_to_set):
    try:
        target_s = pd.qcut(
            original_data,
            top_percentage_slice,
            labels=q_labels).apply(lambda x: 0 if x > 1 else 1)

    except:
        try:
            target_s = pd.cut(original_data, top_percentage_slice,
                              labels=q_labels).apply(lambda x: 0 if x > 1 else 1)

        except:
            target_s = (original_data.rank(ascending=False, method='max') / len(original_data.index) * 4). \
                apply(lambda x: 0 if x > 1 else 1)

    target_s.name = name_to_set
    return target_s


def apply_israelsen(s1, s2):
    idx_to_mul = s1[s1 < 0].index
    idx_to_div = s1.index[~ s1.index.isin(idx_to_mul)]

    s3 = []

    if len(idx_to_mul) > 0:
        s3.append(s1[idx_to_mul] * s2[idx_to_mul])

    if len(idx_to_div) > 0:
        s3.append(s1[idx_to_div] / s2[idx_to_div])

    return pd.concat(s3, axis=0)


def closest_date(dates_to_compare, ref_date):
    date = \
        pd.Series(abs((dates_to_compare - ref_date).days), dates_to_compare).sort_values(ascending=True).index.tolist()[
            0]
    return date


def round_enhanced(value_to_round, decimals=2):
    return np.rint(np.nextafter(value_to_round * 10 ** decimals, np.inf)) / 10 ** decimals


def yesterday():
    return pd.to_datetime('today').normalize() - pd.to_timedelta(1, unit='D')


def last_two_working_days():
    from pandas.tseries.offsets import BDay
    yestdy = business_day(pd.to_datetime('today').normalize(), shift=-1)
    two_days_ago = yestdy - BDay(1)
    return two_days_ago, yestdy


def ytd(st='today'):
    if st == 'today':
        dt = pd.to_datetime('today')
    elif st == 'friday':
        dt = get_last_friday()
    else:
        raise KeyError('st should be today or friday!')

    ytd_dt = pd.to_datetime(str(dt.year)) - pd.Timedelta(1, unit='d')
    return ytd_dt


def annualized_ret(s, annualized=True):
    # todo change function name in "compute_return" or something like that
    years = (pd.to_datetime(s.index[-1]) - pd.to_datetime(s.index[0])).days / 365
    ret = s.iloc[-1] / s.iloc[0] - 1
    if annualized:
        ann_ret = (ret + 1) ** (1 / years) - 1
        return ann_ret
    else:
        return ret


def annualized_ret_df(df):
    if type(df) == pd.Series:
        ann_ret = annualized_ret(df)

    else:
        ann_ret_dict = {}
        for col in df.columns:
            ann_ret = annualized_ret(df.loc[:, col].dropna())
            ann_ret_dict[col] = ann_ret

        ann_ret = pd.Series(ann_ret_dict)

    return ann_ret


def annualized_vol(df):
    freq = pd.infer_freq(df.index)

    if freq == 'M':
        scaling_constant = 12
    elif freq == 'D':
        scaling_constant = 252
    elif freq == 'W':
        scaling_constant = 52
    else:
        # todo: with missing weekends, pandas returns None to this: freq = pd.infer_freq(df.index)
        scaling_constant = 252

    vol_ann = df.pct_change().std() * np.sqrt(scaling_constant)
    return vol_ann


def sharpe_ratio(df, rolling=False, rolling_wdw=60, expanding=None, min_p=180):
    """

    :param df:
    :param rolling:
    :param rolling_wdw:
    :param expanding:
    :param min_p:
    :return:
    """

    if rolling:
        ret_roll = df.rolling(window=rolling_wdw).apply(annualized_ret)
        vol_roll = df.rolling(window=rolling_wdw).apply(annualized_vol)
        sr = ret_roll / vol_roll

    elif expanding:
        ret_exp = df.expanding(min_periods=min_p).apply(annualized_ret)
        vol_exp = df.expanding(min_periods=min_p).apply(annualized_vol)
        sr = ret_exp / vol_exp

    else:
        rets_ann = annualized_ret(df)
        vol_ann = annualized_vol(df)
        sr = rets_ann / vol_ann

    return sr


def israelsen_sharpe_ratio(df, rolling=False, rolling_wdw=60, min_p=180):
    if rolling:
        ret_roll = df.rolling(window=rolling_wdw, min_periods=min_p).apply(annualized_ret)
        vol_roll = df.rolling(window=rolling_wdw, min_periods=min_p).apply(annualized_vol)
        sr = ret_roll.apply(lambda cols: apply_israelsen(cols, vol_roll.loc[:, cols.name]))
    else:
        rets_ann = annualized_ret(df)
        vol_ann = annualized_vol(df)
        sr = apply_israelsen(rets_ann, vol_ann)
    return sr


def rebase_at_x(df, at=100):
    if type(df) == pd.Series:
        df = df.dropna()
        df = df / df.iloc[0] * at
    else:
        df = df.apply(lambda x: x / x.dropna().values.tolist()[0] * at)
    return df


def rebase_at_xs(ri, at):
    """
    :param ri: pd.DataFrame;
    :param at: pd.DataFrame; df that has rebalancing dates as columns and instrument ids as index
    """
    multiple_dts = at.columns
    df_dict = {}
    ri_reb = ri.copy()
    ri_reb = ri_reb.reindex(at.index.get_level_values(0).tolist(), axis=1)
    ri_reb.loc[:, ri_reb.isna().all()] = 100

    for i in np.arange(len(multiple_dts)):
        at_dt = at.loc[:, multiple_dts[i]]
        if len(at.index.names) == 2 and 'daily' in at_dt.index.get_level_values(1).tolist():
            to_rescale = at_dt.xs('daily', level=1)
            w_to_rescale = to_rescale.sum()

        if i == 0:
            df = ri_reb.loc[:multiple_dts[i + 1], :].dropna(how='all', axis=1)
        elif i + 1 == len(multiple_dts):
            df = ri_reb.loc[multiple_dts[i]:, :].dropna(how='all', axis=1)
        else:
            df = ri_reb.loc[multiple_dts[i]:multiple_dts[i + 1], :].dropna(how='all', axis=1)

        if len(at.index.names) < 2:
            df_dict[i] = df.shift().iloc[1:, :].apply(
                lambda x: x / x.dropna().values.tolist()[0] * at.loc[x.name, multiple_dts[i]])
        else:
            df_before_resc = df.apply(lambda x: x / x.dropna().values.tolist()[0]
                                                * at.reset_index(1, True).loc[x.name, multiple_dts[i]])

            df_before_resc.loc[:, to_rescale.index] = to_rescale.to_frame().T.reindex(ri_reb.index).ffill().loc[
                                                      df_before_resc.index, :].mul(df_before_resc.loc[:,
                                                                                   to_rescale.index].sum(1),
                                                                                   0) / w_to_rescale
            df_dict[i] = df_before_resc

    df_to_go = pd.concat(df_dict)
    df_to_go.index = df_to_go.index.droplevel()
    return df_to_go


def get_root_and_store_dir():
    import os
    from os.path import dirname, abspath
    ROOT_DIR = dirname(abspath(__file__))

    if 'workspace' in ROOT_DIR or 'euclidea' in ROOT_DIR:
        STORE_DIR = os.path.join('/Storage', 'SpiritData')

    else:
        root_split = ROOT_DIR.split('\\')
        proj_name_num = root_split.index('PycharmProjects') + 2
        ROOT_DIR = '\\'.join(root_split[:proj_name_num])
        STORE_DIR = os.path.join(ROOT_DIR, 'Storage')

    return ROOT_DIR, STORE_DIR


def weighted_median(df, value_col='value', weight_col='weight', drop_nan=False):
    item_df = df.loc[:, [value_col, weight_col]]
    if drop_nan:
        item_df = item_df[item_df[value_col].notna()]
    item_df['cum_weight'] = item_df.loc[:, weight_col].cumsum()
    cutoff = item_df.loc[:, weight_col].sum() / 2
    item = item_df[item_df.cum_weight > cutoff].loc[:, value_col].values[0]
    return item


def df_to_series(df):
    if not isinstance(df, pd.Series) and not isinstance(df, pd.DataFrame):
        return df
    elif isinstance(df, pd.Series):
        return df
    elif df.shape[1] == 1:
        return df.iloc[:, 0]
    return df


def get_npv(flows=2500, periods=12 * 30, freq='M', yields=None):
    import numbers

    if isinstance(yields, float):
        if freq == 'Y':
            pass
        elif freq == 'H':
            interest_rate = yields / 2
        elif freq == 'Q':
            interest_rate = yields / 4
        elif freq == 'M':
            interest_rate = (1 + yields) ** (1 / 12) - 1
            print(interest_rate)
        # start from npv = 0 and sum each period
        npv = 0.0
        for i in range(periods):
            npv += flows / (1 + yields) ** (i + 1)

        return npv

    # if interest rate is variable -> series
    if isinstance(yields, pd.Series):
        mapping_dict = {}
        for i in yields.index:
            freq = i[-1]
            if freq == 'M':
                mapping_dict[i] = int(i[:-1]) * 1
            if freq == 'Y':
                mapping_dict[i] = int(i[:-1]) * 12

        yields = yields.rename(index=mapping_dict) / 100
        yields_interpolated = yields.reindex(range(1, periods + 1)).interpolate().reset_index().bfill()
        yields_interpolated.columns = ['period', 'yield']
        discount_factor = (yields_interpolated.loc[:, 'yield'] + 1) ** (yields_interpolated.loc[:, 'period'] / 12)
        discount_factor.name = 'discount'

        if isinstance(flows, numbers.Number):
            flows_s = pd.Series(flows, index=yields_interpolated.index, name='flows')
        else:
            flows_s = flows

        full_df = pd.concat([yields_interpolated, discount_factor, flows_s], axis=1)
        discounted_cfs = full_df.flows.div(full_df.discount)
        npv = discounted_cfs.sum()
        return npv


def herfindahl_index(l):
    # this is a differentiation index, defined as the sum of 1 - w**2 of all securities in a ptf
    return 1 - sum(list(map(lambda x: x ** 2, l)))


def comprehensive_concentration_index(l):
    w_max = max(l)
    return 1 - w_max - sum(list(map(lambda x: (x ** 2) * (1 + (1 - x)), l)))


def entropy_index(l):
    l2 = filter(lambda x: x > 0, l)
    return - sum(list(map(lambda x: x * np.log(x), l2)))


def common_reindex(x, y, axis=0, sort=False):
    if axis == 0:
        common_index = list(set(x.index) & set(y.index))
        if sort:
            common_index.sort()
        return x.loc[common_index], y.loc[common_index]
    else:
        common_col = list(set(x.columns) & set(y.columns))
        if sort:
            common_col.sort()
        return x.loc[:, common_col], y.loc[:, common_col]


def adj_tev(te, ts, bench_ts, start, end):
    if te >= 0:
        return (ts.loc[start:end, :].iloc[1:] - bench_ts.loc[start:end, :].iloc[1:]).std() * np.sqrt(252)
    else:
        return 1 / ((ts.loc[start:end, :].iloc[1:] - bench_ts.loc[start:end, :].iloc[1:]).std() * np.sqrt(252))


def tracking_error(ts, bmk, st='2016-12-31'):
    df = pd.concat([ts, bmk], axis=1).truncate(before=st)
    df_ret = df.pct_change().dropna()
    tr_err = df_ret.iloc[:, 0] - df_ret.iloc[:, 1]
    tr_err.name = 'tracking_error'
    return tr_err


def tev(ts, bmk, freq='daily', rolling=True, window=252):
    # todo different frequencies
    if freq == 'daily':
        n = 252
    else:
        n = 252

    if not rolling:
        return tracking_error(ts, bmk).std() * np.sqrt(n)
    else:
        return (tracking_error(ts, bmk) * np.sqrt(n)).rolling(window=window).std()


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
