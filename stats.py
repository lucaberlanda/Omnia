import numpy as np
import pandas as pd
import primitive as prim
import Omnia.risk as orm
import Omnia.core_primitive as core
from Omnia.tail_risk import TailRisk


class Stats:

    def __init__(self, ts, bench=None, end=pd.to_datetime('today')):

        self.ts = ts.to_frame() if isinstance(ts, pd.Series) else ts
        self.ts = self.ts.truncate(after=end)
        self.rets = self.ts.pct_change(fill_method=None)

        if bench is not None:
            self.bench = core.df_to_series(bench).truncate(after=end)
            self.bench_rets = self.bench.pct_change(fill_method=None)
        else:
            self.bench = bench
            self.bench_rets = None

        self.end = end

        self.vols = None
        self.performances = None
        self.dds = None
        self.relative_performances = None
        self.te_vola = None
        self.info_ratio = None
        self.adj_info_ratio = None

    # RETS MEASURES ================

    def mean_positive_ret(self):
        return self.rets.apply(lambda x: x[x > 0].mean())

    def mean_negative_ret(self):
        return self.rets.apply(lambda x: x[x < 0].mean())

    def proportion_positive_ret(self):
        num_pos_rets = self.rets.apply(lambda x: x[x > 0].dropna().count())
        tot = self.rets.dropna().count()
        return (num_pos_rets / tot).rename('Proportion Positive Returns')

    # RISK MEASURES ================

    def volatility(self, rolling=False, window=60, min_p=60):
        if rolling:
            vol = self.rets.rolling(window=window, min_periods=min_p).std() * np.sqrt(252)
        else:
            vol = core.annualized_vol(self.ts)
        return vol

    def downside_risk(self, threshold=0, rolling=False, window=60, min_p=60):
        negative_rets = self.rets.applymap(lambda x: min([threshold, x]) if not np.isnan(x) else x)
        if rolling:
            dr = np.sqrt((negative_rets ** 2).rolling(window=window, min_periods=min_p).mean() * 252)
        else:
            dr = np.sqrt((negative_rets ** 2).mean() * 252)
        return dr

    def skw(self):
        return self.rets.skew()

    def krt(self):
        return self.rets.kurtosis()

    def tail_risk(self, m=10):
        # TODO m percentage
        return self.rets.apply(lambda x: TailRisk.compute_hill_estimator(x, m=m))

    def max_dd(self, rolling=False, window=60, min_p=60):
        if rolling:
            dd = self.ts.rolling(window=window, min_periods=min_p).apply(lambda x: orm.drawdown(x))
            return dd
        else:
            all_data = self.ts.apply(lambda x: orm.drawdown(x, True)).T
            all_data.columns = ['drawdown', 'start_date', 'end_date']
            dd = all_data.loc[:, 'drawdown']
            st = all_data.loc[:, 'start_date']
            end = all_data.loc[:, 'end_date']
            return dd, st, end

    def time_to_recovery(self):
        _, start_dd, end_dd = self.max_dd()
        start_ttr = end_dd.apply(lambda x: core.get_following_business_day(x, False))
        recovery_values = self.ts.apply(lambda x: x.loc[start_dd.loc[x.name]])
        ttr = self.ts.apply(lambda x: orm.time_to_recovery(x, start_ttr.loc[x.name], recovery_values.loc[x.name]))
        return ttr

    # RATIOS ================

    def sr(self, apply_israelsen=True, rolling=False, window=60, min_p=60):
        if apply_israelsen:
            return core.israelsen_sharpe_ratio(self.ts, rolling, window, min_p)
        else:
            return core.sharpe_ratio(self.ts, rolling, window, None, min_p)

    def calmar_ratio(self, apply_israelsen=True, rolling=False, window=60, min_p=60):
        if rolling:
            rets = self.rets().rolling(window=window, min_periods=min_p).mean()
            dd = self.max_dd(rolling=True, window=window, min_p=min_p)
        else:
            rets = core.annualized_ret(self.ts)
            dd, _, _ = self.max_dd()

        if apply_israelsen:
            if isinstance(rets, pd.DataFrame):
                return rets.apply(lambda cols: core.apply_israelsen(cols, dd.loc[:, cols.name]))
            else:
                return rets / dd
        else:
            return rets / dd

    def sortino_ratio(self, threshold=0, rolling=False, window=60, min_p=60):
        if rolling:
            rets = self.rets().rolling(window=window, min_periods=min_p).mean()
        else:
            rets = core.annualized_ret(self.ts)
        dr = self.downside_risk(threshold=threshold, rolling=rolling, window=window, min_p=min_p)
        return (rets - threshold) / dr

    def max_dd_over_vol(self):
        dd, _, _ = self.max_dd()
        vols = core.annualized_vol(self.ts)
        return dd / vols

    # Benchmark Relative Measures

    def tracking_error(self):
        return (self.rets.sub(self.bench_rets, axis=0)).dropna().mean() * 252

    def tev(self):
        return (self.rets.sub(self.bench_rets, axis=0)).dropna().std() * np.sqrt(252)

    def proportion_positive_ter(self):
        ter = self.rets.sub(self.bench_rets, axis=0).dropna()
        num_pos_ter = ter.apply(lambda x: x[x > 0].dropna().count())
        tot = ter.count()
        return (num_pos_ter / tot).rename('Proportion Positive TER')

    def information_ratio(self):
        te = self.tracking_error()
        tev = self.tev()
        return te / tev

    def marginal_risk_contribution(self, weights, period_back=365):
        first_day = pd.to_datetime('today') - pd.Timedelta(period_back, unit='D')
        truncated_ts_rets = self.rets.truncate(before=first_day)
        var_cov_matrix = truncated_ts_rets.cov().loc[weights.index, weights.index]
        return orm.marginal_percentage_risk_contribution(w=weights, omega=var_cov_matrix).rename(
            'marginal_risk_contribution')

    def summary_stats(self):
        stats_dict = dict()
        stats_dict['Return'] = core.annualized_ret(self.ts, annualized=False)
        stats_dict['Annualized Return'] = core.annualized_ret(self.ts)
        stats_dict['Annualized Volatility'] = self.volatility()
        stats_dict['Mean Positive Return'] = self.mean_positive_ret()
        stats_dict['Mean Negative Return'] = self.mean_negative_ret()
        stats_dict['Proportion Positive Returns'] = self.proportion_positive_ret()

        stats_dict['Annualized Sharpe Ratio'] = self.sr()
        stats_dict['Annualized Sortino Ratio'] = self.sortino_ratio()
        stats_dict['Annualized Calmar Ratio'] = self.calmar_ratio()

        dd_data = self.max_dd()
        stats_dict['Maximum Drawdown'] = dd_data[0]
        stats_dict['Maximum Drawdown Start Date'] = dd_data[1]
        stats_dict['Maximum Drawdown End Date'] = dd_data[2]

        stats_dict['Skewness'] = self.skw()
        stats_dict['Kurtosis'] = self.krt()

        stats_dict['Annualized Tracking Error'] = self.tracking_error()
        stats_dict['Annualized TEV'] = self.tev()
        stats_dict['Hit Ratio'] = self.proportion_positive_ter()
        stats_dict['Annualized Information Ratio'] = self.information_ratio()

        return pd.concat(stats_dict).unstack(level=0).T

    # REPORTING FUNCTIONS ================

    def get_performances(self, dts=None, annualized=False):
        performances = self.stats_over_periods(dts, 'performances', annualized=annualized)
        self.performances = performances

    def get_vols(self, dts=None, skipna=False, annualized=True):
        vols = self.stats_over_periods(dts, 'vol', annualized=annualized, skipna=skipna)
        self.vols = vols

    def get_dds(self, dts=None):
        dds = self.stats_over_periods(dts, 'dd', annualized=False)
        self.dds = dds

    def get_relative_performances(self, dts=None, annualized=False):
        relative_performances = self.stats_over_periods(dts, 'relative_performances', annualized=annualized)
        self.relative_performances = relative_performances

    def get_tev(self, dts=None, annualized=False):
        terr_vola = self.stats_over_periods(dts, 'tev', annualized=annualized)
        self.te_vola = terr_vola

    def get_information_ratio(self, dts=None, annualized=False):
        information_ratios = self.stats_over_periods(dts, 'information_ratio', annualized=annualized)
        self.info_ratio = information_ratios

    def get_adj_info_ratio(self, dts=None, annualized=False):
        adj_info_ratios = self.stats_over_periods(dts, 'adj_info_ratio', annualized=annualized)
        self.adj_info_ratio = adj_info_ratios

    def stats_over_periods(self, dts, what, annualized, skipna=False, extended_report=True):

        if self.end.normalize() == pd.to_datetime('today').normalize():
            yesterday = (self.end - pd.to_timedelta(1, unit='D')).normalize()
        else:
            yesterday = self.end

        if dts is None:
            if extended_report:
                inc = self.ts.index[0]
                dts = core.get_dates_for_performances_extended(inc, yesterday)
            else:
                dts = core.get_dates_for_performances(yesterday)

        stats_dict = {}
        for period in dts.keys():
            end = dts[period][1]
            start = dts[period][0]

            ts_cleaned = self.ts.truncate(before=start, after=end)

            dates = ts_cleaned.index.tolist()
            all_dates = sorted(list(set(dates + [end] + [start])))
            ts_cleaned = ts_cleaned.reindex(all_dates).ffill().bfill()

            ts_cleaned_st = ts_cleaned.loc[start, :]
            ts_cleaned_end = ts_cleaned.loc[end, :]

            if what == 'performances':

                if annualized:
                    years = (end - start).days / 365
                    stats = (ts_cleaned_end / ts_cleaned_st) ** (1 / years) - 1
                else:
                    stats = ts_cleaned_end / ts_cleaned_st - 1

            elif what == 'vol':
                if annualized:
                    stats = self.rets.loc[start:end, :].iloc[1:].std(skipna=skipna) * np.sqrt(252)
                else:
                    stats = self.rets.loc[start:end, :].iloc[1:].std(skipna=skipna)

            elif what == 'dd':
                stats = self.ts.loc[start:end, :].apply(lambda x: orm.drawdown(x, False)).T

            elif what == 'relative_performances':
                if annualized:
                    years = (end - start).days / 365
                    reindexed_series = self.bench.reindex(ts_cleaned.index).ffill()
                    bench_ret = reindexed_series.loc[end, :] / reindexed_series.loc[start, :] ** (1 / years) - 1
                    ptf_ret = (ts_cleaned_end / ts_cleaned_st) ** (1 / years) - 1
                    stats = ptf_ret - bench_ret
                else:
                    reindexed_series = self.bench.reindex(ts_cleaned.index).ffill()
                    bench_ret = reindexed_series.loc[end] / reindexed_series.loc[start] - 1
                    ptf_ret = ts_cleaned_end / ts_cleaned_st - 1
                    stats = ptf_ret - bench_ret

            elif what == 'tev' and len(self.rets.columns) == 1:
                reindexed_rets = self.bench_rets.reindex(self.rets.index).ffill()
                terr_vol = (core.df_to_series(self.rets.loc[start:end, :].iloc[1:]) - reindexed_rets.loc[
                                                                                      start:end].iloc[
                                                                                      1:]).std() * np.sqrt(252)
                stats = pd.Series(terr_vol, index=self.rets.columns)

            elif what == 'information_ratio' and len(self.rets.columns) == 1:
                reindexed_series = self.bench.reindex(ts_cleaned.index).ffill()
                reindexed_rets = core.df_to_series(self.bench_rets.reindex(self.rets.index).fillna(0))

                bench_ret = reindexed_series.loc[end] / reindexed_series.loc[start] - 1
                ptf_ret = (ts_cleaned_end / ts_cleaned_st - 1)[0]
                te = ptf_ret - bench_ret
                tev = (core.df_to_series(self.rets.loc[start:end, :].iloc[1:]) - reindexed_rets.loc[start:end].iloc[
                                                                                 1:]).std() * np.sqrt(252)
                stats = pd.Series(te / tev, index=self.rets.columns)

            elif what == 'adj_info_ratio' and len(self.rets.columns) == 1:
                reindexed_series = self.bench.reindex(ts_cleaned.index).ffill()
                reindexed_rets = core.df_to_series(self.bench_rets.reindex(self.rets.index).fillna(0))
                reindexed_rets = reindexed_rets.rename(self.rets.columns[0])

                bench_ret = reindexed_series.loc[end] / reindexed_series.loc[start] - 1
                ptf_ret = ts_cleaned_end / ts_cleaned_st - 1
                te = ptf_ret - bench_ret
                tev = pd.Series([core.adj_tev(row, self.rets, reindexed_rets.to_frame(), start, end) for row in te])[0]
                stats = te / tev

            else:
                return
                # raise KeyError('Stats have to be computed for performances, vols and dds')

            stats_dict[period] = stats

        stats_df = pd.DataFrame(stats_dict)
        stats_df.columns = pd.MultiIndex.from_product([[what], stats_df.columns])
        return stats_df

    def report(self, dts=None, verbose=True):

        self.get_performances(dts=dts)
        self.get_vols(dts=dts, skipna=True)
        self.get_dds(dts=dts)
        self.get_relative_performances(dts=dts)
        self.get_tev(dts=dts)
        self.get_information_ratio(dts=dts)
        self.get_adj_info_ratio(dts=dts)

        if verbose:
            print('--- Performances ---\n', self.performances, '\n')
            print('--- Volatilities ---\n', self.vols, '\n')
            print('--- Maximum Drawdown ---\n', self.dds, '\n')
        else:
            pass
