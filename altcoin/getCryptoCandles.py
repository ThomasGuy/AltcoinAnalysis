import json
import requests
import pandas as pd
import pkg_resources


endpoint = 'https://min-api.cryptocompare.com/data/histoday'
coinlist = 'https://min-api.cryptocompare.com/data/all/coinlist'


top_thirty1 = ['BTC', 'ETH', 'XRP', 'BCH', 'EOS', 'ADA', 'LTC', 'XLM', 'TRX',
               'NEO']
top_thirty2 = ['MIOTA', 'XMR', 'DASH', 'XEM', 'VEN', 'USDT', 'ETC', 'QTUM',
               'OMG', 'ICX']
top_thirty3 = ['BNB', 'LSK', 'BTG', 'AE', 'NANO', 'ZEC', 'BTM', 'STEEN', 'XVG',
               'BCN', 'ICX']


def _getData(sym):
    res = requests.get(endpoint + '?fsym={}&tsym=USD&limit=2000'.format(sym))
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    return hist


def newCoins(coins):
    for coin in coins:
        data = _getData(coin)
        path = 'cryptoData\\1D\\{}_Candles.csv'.format(coin)
        data.to_csv(pkg_resources.resource_filename(__name__, path))
