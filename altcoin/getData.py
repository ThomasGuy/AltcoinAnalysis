import pandas as pd
import pkg_resources
from pandas.tseries.offsets import Hour


times = {
    '3h': [1, 4, 7, 10, 13, 16, 19, 22],
    '6h': [1, 7, 13, 19],
    '12h': [7, 19],
    '1D': [7]
}


def getData(sym, step, *args):
    filepath = _masterDataPath(sym, step)
    return pd.read_csv(filepath, index_col='MTS', parse_dates=True)
    # return df.rename({'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, axis=1)


def rawDataResampled(sym, step, resample_freq='6H'):
    """This is for use Altcoin .csv data"""
    df = getData(sym, step)
    base = df.index[-1].hour + df.index[-1].minute / 60.0
    return df.resample(rule=resample_freq, closed='right', label='right', base=base).agg(
        {'Open': 'first', 'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'mean'})


def _setMasterData(sym, step, dataf):
    filepath = _masterDataPath(sym, step)
    dataf.to_csv(filepath)


def getOldData(sym, step):
    path = 'Data\\Master_old\\{}\\{}_Candles_{}.csv'.format(step, sym, step)
    filepath = pkg_resources.resource_filename(__name__, path)
    return pd.read_csv(filepath, index_col='MTS', parse_dates=True)


def _masterDataPath(sym, step):
    path = 'Data\\MasterData\\{}\\{}_Candles_{}.csv'.format(step, sym, step)
    return pkg_resources.resource_filename(__name__, path)


def retrieve_oldMaster(coins, interval):
    for delta in interval:
        for coin in coins:
            sym = coin[:3]
            backup = getOldData(sym, delta)
            _setMasterData(sym, delta, backup)
