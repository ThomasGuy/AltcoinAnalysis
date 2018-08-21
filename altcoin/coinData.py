import logging
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from .getData import getData

log = logging.getLogger(__name__)


class CoinData:
    """
    Generate data frame, set big, small and long ema's.
    Set daily and cumlative returns.
    Check volatility and calculate risk
    """
    def __init__(self, coin, step, start=None, finish=None, sma=10, bma=27, lma=74):
        self.coin = coin
        self.step = step
        self.sma = sma
        self.bma = bma
        self.lma = lma
        self.dataf = self._get_DFTable(start, finish)
        self.cross = self._crossover(self.dataf)

    def _get_DFTable(self, start, finish):
        df = getData(self.coin, self.step).loc[start:finish]
        df.drop_duplicates()
        df = df.groupby('MTS')['open', 'close', 'high', 'low', 'volume'].mean()
        df['sewma'] = df['close'].ewm(span=self.sma).mean()
        df['bewma'] = df['close'].ewm(span=self.bma).mean()
        df['lewma'] = df['close'].ewm(span=self.lma).mean()
        df['return'] = df['close'].pct_change(1)
        df['cumlative'] = (1 + df['return']).cumprod()
        return df

    def _crossover(self, dataset):
        """From DataFrame 'dataset'. Return a DataFrame of all the crossing points
         of the small and medium moving averages"""
        record = []
        # Don't use 1st db record as it may have equal SEWMA = BEWMA
        Higher = dataset.iloc[2]['sewma'] > dataset.iloc[2]['bewma']
        # initialize record[] ensures record is never empty
        if Higher:
            record.append([dataset.index[1], dataset['close'].iloc[1], 'Buy'])
        else:
            record.append([dataset.index[1], dataset['close'].iloc[1], 'Sell'])

        # Catch Numpy warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for date, row in dataset.iterrows():
                if Higher:
                    # Sell condition
                    if row['sewma'] / row['bewma'] < 0.9965:
                        record.append([date, row['close'], 'Sell'])
                        Higher = not Higher
                else:
                    # Buy condition
                    if row['sewma'] / row['bewma'] > 1.0035:
                        record.append([date, row['close'], 'Buy'])
                        Higher = not Higher

        cross = pd.DataFrame(record, columns=('MTS close Transaction').split())
        cross.set_index('MTS', drop=True, inplace=True)
        return cross

    def plotDataset(self, text='Title'):
        fig = plt.figure(figsize=(16, 9))
        axes = fig.add_axes([0, 0, 1, 1])
        axes.plot(self.dataf.index, self.dataf['sewma'], label='sma={}'.format(self.sma), color='blue')
        axes.plot(self.dataf.index, self.dataf['bewma'], label='bma={}'.format(self.bma), color='red')
        axes.plot(self.dataf.index, self.dataf['lewma'], label='lma={}'.format(
                  self.bma), color='orange', alpha=.5)
        axes.plot(self.dataf.index, self.dataf['close'], label='close', color='green', alpha=.5)
        axes.plot(self.dataf.index, self.dataf['high'], label='high', color='pink', alpha=.5)

        # Plot the sewma/bewma crossing buy & sell points
        sold = pd.DataFrame(self.cross[self.cross['Transaction'] == 'Sell']['close'])
        axes.scatter(sold.index, sold['close'], color='r', label='Sell', lw=3)
        bought = pd.DataFrame(self.cross[self.cross['Transaction'] == 'Buy']['close'])
        axes.scatter(bought.index, bought['close'], color='g', label='Buy', lw=3)

        axes.set_ylabel('closing pric')
        axes.set_xlabel('Date')
        axes.grid(color='b', alpha=0.5, linestyle='--', linewidth=0.5)
        axes.grid(True)
        plt.legend()
        plt.show()
