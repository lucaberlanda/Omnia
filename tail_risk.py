import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('white')

"""
Here we try to find an estimator for the tails in financial time series

MAD / standard deviation: We can use the MAD/SD ratio as a gauge of fat-tailedness. 
The closer the number is to zero, the fatter the tails. The closer the number is to 1 (it can never exceed 1!), 
the thinner the tails.
"""


class TailRisk:
    def __init__(self, data_to_download={'distr': 't_student', 'param': 2}, draws=5000):

        self.data_to_download = data_to_download

        if self.data_to_download['distr'] == 't_student':
            self.distribution = pd.Series(np.random.standard_t(data_to_download['param'], draws))

        elif self.data_to_download['distr'] == 'gaussian':
            self.distribution = pd.Series(np.random.normal(size=draws))

        elif self.data_to_download['distr'] == 'ids':
            print('This has to be fixed')

    @staticmethod
    def plot_histogram(x, degrees=None, sample=None, x_label='MAD on SD', y_label='Frequency'):

        x_label = x_label
        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)
        ax.text(2, 0.4, 't-student, ' + str(degrees) + ' degrees of freedom, sample size =' + str(sample),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10}, fontsize=30)

        ax.hist(x, bins=50, normed=True, color='yellowgreen')
        ax.set_xlabel(x_label, size=30)
        ax.set_ylabel(y_label, size=25)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.show()

    @staticmethod
    def plot_scatter(x, x_label='Unknown', y_label='Degrees of Freedom', log_scale=False):
        """

        :param y_label:
        :param x_label:
        :param x:
        :param log_scale:
        :return: plot a series with its index
        """
        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

        if log_scale:
            ax.set_yscale('log')

        ax.scatter(x.index, x, facecolor="none", edgecolor="green", linewidth='2', s=40)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlabel(y_label, size=20)
        ax.set_ylabel(x_label, size=20)
        plt.show()

    @staticmethod
    def compute_mad_on_std(s, reverse=False):

        MAD = abs(s - s.mean()).mean()
        std_dev = s.std()

        if reverse:
            ratio = std_dev / MAD
        else:
            ratio = MAD / std_dev

        return ratio

    @staticmethod
    def expanding_metric(s, metric='MAD_on_SD', reverse=False):

        expanding_series = {}

        if metric == 'MAD_on_SD':
            expanding_series = s.expanding().apply(lambda x: TailRisk.compute_mad_on_std(x, reverse=reverse))

        elif metric == 'Hills':
            expanding_series = s.expanding(min_periods=30).apply(lambda x: TailRisk.compute_hill_estimator(x))

        elif metric == 'Kurtosis':
            expanding_series = s.expanding(min_periods=30).apply(lambda x: pd.Series(x).kurtosis())

        else:
            print('Which estimator?')
            quit()

        return expanding_series

    @staticmethod
    def rolling_metric(s, metric='MAD_on_SD', reverse=False):

        rolling_series = {}

        if metric == 'MAD_on_SD':
            rolling_series = s.rolling(window=60).apply(lambda x: TailRisk.compute_mad_on_std(x, reverse=reverse))

        elif metric == 'Hills':
            rolling_series = s.rolling(window=60).apply(lambda x: TailRisk.compute_hill_estimator(x))

        elif metric == 'Kurtosis':
            rolling_series = s.rolling(window=60).kurtosis()


        else:
            print('Which estimator?')
            quit()

        return rolling_series

    @staticmethod
    def compute_hill_estimator(s, m=10):
        """

        :param s: pd.Series, ordered descending
        :param m:
        :return:
        """

        if m == 0:
            hill_estimator = None

        elif m > len(s.index) or m >= len(s.unique()) or (np.log(s[:m] / s[m]).sum()) == 0:
            hill_estimator = None

        else:
            hill_estimator = 1 / (np.log(s[:m] / s[m]).sum() / m)  # not m + 1 cuz in the denominator, m considers the 0

        return hill_estimator

    @staticmethod
    def log_log_plot_with_threshold(s, threshold=0):

        if type(s) == pd.Series:
            s = pd.DataFrame(s)

        id = s.columns.values.tolist()[0]
        s = s.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        filtered_data = s[abs(s) > threshold]

        sorted_returns = abs(filtered_data).dropna().sort(columns=filtered_data.columns[0]).reset_index().drop(
            'index', 1).reset_index()

        sorted_returns['cdf'] = 1 - (sorted_returns.loc[:, 'index'] / len(sorted_returns.index))

        return sorted_returns

    @staticmethod
    def ME_plot(s, starting_threshold=0):

        if type(s) == pd.Series:
            s = pd.DataFrame(s)

        s = s.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        filtered_data = abs(s[abs(s.iloc[:, 0]) > starting_threshold])

        mean_exceedances_dict = {}
        u = filtered_data.values.flatten().tolist()

        for i in u:
            exceedances = s[s.iloc[:, 0] > i] - i
            mean_exceedances = exceedances.mean().values[0]
            mean_exceedances_dict[i] = mean_exceedances

        to_plot = pd.Series(mean_exceedances_dict).to_frame().reset_index()
        to_plot.columns = ['threshold', 'mean_exceedances']

        return to_plot


def compute_stuff_part_II(ts, name, ax1, ax2, ax3, color):
    cleaned_series = abs(ts).sort_values(ascending=False).dropna()
    cleaned_series.index = range(len(cleaned_series.index))

    hills_estimator_func_of_k_dict = {}

    for j in range(1, round(len(cleaned_series.index) * 0.1)):  # range(1, len(data.cleaned_series.index)):
        hill = TailRisk.compute_hill_estimator(cleaned_series, m=j)
        hills_estimator_func_of_k_dict[j] = hill

    hills_estimator_func_of_k = pd.Series(hills_estimator_func_of_k_dict)

    ax1.plot(hills_estimator_func_of_k, 'bo', label=name, alpha=0.5, color=color)
    ax1.plot(hills_estimator_func_of_k, color)
    # ax.title('Hill estimator')

    mean_exc = TailRisk.ME_plot(cleaned_series).dropna()
    ax2.scatter(x=mean_exc.loc[:, 'threshold'], y=mean_exc.loc[:, 'mean_exceedances'], alpha=0.5, s=70,
                c=color, label=name)  # .set_title('ME plot')

    sorted_returns = TailRisk.log_log_plot_with_threshold(cleaned_series)
    sorted_returns.columns = ['index', 'ret', name]
    ax3.loglog(sorted_returns.loc[:, 'ret'], sorted_returns.loc[:, name], 'o', alpha=0.5, c=color)

    ax1.legend(loc='upper right')
    ax1.set_title('Hills Plot', size=15)
    ax1.set_xlabel('k order statistic', fontsize=12)
    ax1.set_ylabel('Hill estimator', fontsize=12)

    ax2.legend(loc='upper right')
    ax2.set_title('Mean Exceedances Plot', size=15)
    ax2.set_xlabel('threshold', fontsize=12)
    ax2.set_ylabel('mean_exceedance', fontsize=12)

    ax3.legend(loc='upper right')
    ax3.set_title('Log Log Plot', size=15)
    ax3.set_xlabel('log of returns', fontsize=12)
    ax3.set_ylabel('Pr (|return| > x)', fontsize=12)


def compute_stuff(distribution, ids_or_param, colors):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    data_to_download = {'distr': distribution, 'param': ids_or_param}
    data = TailRisk(data_to_download)

    if type(data.distribution) == pd.Series:
        data.distribution = pd.DataFrame(data.distribution)
        data.distribution.columns = [data_to_download['distr']]

    for k in data.distribution.columns:
        print(k)
        compute_stuff_part_II(data.distribution.loc[:, k],
                              str(k),
                              ax1,
                              ax2,
                              ax3,
                              colors[data.distribution.columns.tolist().index(k)])

    plt.show()
