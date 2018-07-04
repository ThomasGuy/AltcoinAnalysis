import pandas as pd
import pkg_resources
from pandas.tseries.offsets import Hour


times ={'3h': [1, 4, 7, 10, 13, 16, 19, 22],
        '6h': [1, 7, 13, 19],
        '12h': [7, 19],
        '1D': [7]}


def getData(sym, step):
    filepath = _masterDataPath(sym, step)
    return pd.read_csv(filepath, index_col='MTS', parse_dates=True)


def _setMasterData(sym, step, dataf):
    filepath = _masterDataPath(sym ,step)
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


# This has been sucessful for 3h and 6h
def fixTimeStamp(coins):

    delta = '6h'
    for coin in coins:
        sym = coin[:3]
        dataf = getData(sym, delta)
        offset_arr = []
        hours = times[delta]
        for date, row in dataf.iterrows():
            if date.hour not in hours:
                dataf.drop(date, inplace=True)
                date = date + Hour()
                offset_arr.append([date,row['open'],row['close'],row['high'],row['low'],row['volume']])
        
        replacment = pd.DataFrame(offset_arr, columns=('MTS open close high low volume').split())
        replacment.set_index('MTS', inplace=True)
        dataf = pd.concat([dataf, replacment])
        dataf.drop_duplicates()
        grouped = dataf.groupby('MTS')['open', 'close', 'high', 'low', 'volume'].mean()
        _setMasterData(sym, delta, grouped)


def _test(data, interval):
    ts_arr = []
    hours = times[interval] # hours need to be chosen for the interval being tested
    for date, row in data.iterrows():
        if date.hour not in hours:
            ts_arr.append(date)
    return ts_arr


# to fix 12h we need to substitute in from 6h
def fix12hTimeStamp(coins):
    for coin in coins:
        replace = []
        sym = coin[:3]
        data = getData(sym, '12h')
        sixhourData = getData(sym, '6h')
        arr = _test(data, '12h')
        for date in arr:
            data.drop(date, inplace=True)
            if date.hour in [1, 13]:
                date = date + 6*Hour()
                row = sixhourData.loc[date]
                replace.append([date, row['open'], row['close'],
                                row['high'], row['low'], row['volume']])
        DF = pd.DataFrame(replace, columns=(
            'MTS open close high low volume').split())
        DF.set_index('MTS', inplace=True)
        finalDF = pd.concat([data, DF])
        _setMasterData(sym, '12h', finalDF)


# to fix 1D entries we need to do something simular
def fix_1D_TimeStamp(coins):
    for coin in coins:
        replace = []
        sym = coin[:3]
        data = getData(sym, '1D')
        arr = _test(data, '1D')
        for date in arr:
            row = data.loc[date]
            data.drop(date, axis=0, inplace=True)
            date = date + ((7 - date.hour)*Hour())
            replace.append([date, row['open'], row['close'],row['high'], row['low'], row['volume']])
        DF = pd.DataFrame(replace, columns=('MTS open close high low volume').split())
        DF.set_index('MTS', inplace=True)
        finalDF = pd.concat([data, DF])
        finalDF.drop_duplicates()
        grouped = finalDF.groupby('MTS')['open', 'close', 'high', 'low', 'volume'].mean()
        _setMasterData(sym, '1D', grouped)

