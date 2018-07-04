# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:37:23 2018

@author: Sporty
"""

import pkg_resources, time, logging
import pandas as pd
import numpy as np
import requests, json, time
from requests import Session
from pandas import DataFrame
from datetime import datetime
from datetime import timedelta
from .getData import getData

log = logging.getLogger(__name__)


bitfinex = 'https://api.bitfinex.com/v2/candles/trade:'

sym_usd = ['AIDUSD','AVTUSD','BATUSD','BCHUSD','BTCUSD','BTGUSD','DATUSD',
           'DSHUSD','EDOUSD','ELFUSD','EOSUSD','ETCUSD','ETHUSD','ETPUSD',
           'FUNUSD','GNTUSD','IOTUSD','LTCUSD','MNAUSD','NEOUSD','OMGUSD',
           'QSHUSD','QTMUSD','RCNUSD','REPUSD','RLCUSD','RRTUSD','SANUSD',
           'SNGUSD','SNTUSD','SPKUSD','TNBUSD','TRXUSD','XMRUSD','XRPUSD',
           'YYWUSD','ZECUSD','ZRXUSD']

top_twenty5 = ['BCHUSD','BTCUSD','BTGUSD','DSHUSD','EOSUSD','ETCUSD','ETHUSD',
               'IOTUSD','LTCUSD','NEOUSD','OMGUSD','QSHUSD','QTMUSD',
               'TRXUSD','XMRUSD', 'XRPUSD','ZECUSD','GNTUSD','SANUSD','SPKUSD']

takeAway = ['AVTUSD','FUNUSD','RCNUSD','RLCUSD']

all_coins = ['BCHUSD', 'BTCUSD', 'BTGUSD', 'DSHUSD', 'EOSUSD', 'ETCUSD', 'ETHUSD',
             'IOTUSD', 'LTCUSD', 'NEOUSD', 'OMGUSD', 'QSHUSD', 'QTMUSD',
             'TRXUSD', 'XMRUSD', 'XRPUSD', 'ZECUSD', 'GNTUSD', 'SANUSD', 'SPKUSD',
             'AVTUSD', 'FUNUSD', 'RCNUSD', 'RLCUSD']


intervals = ['1h', '3h', '6h', '12h', '1D']
short_int = ['1h', '3h']
mid_int = ['6h', '12h']
long_int = ['1D']
intervalsInSecounds = {
    '1h': 3600,
    '3h': 10800,
    '6h': 21600,
    '12h': 43200,
    '1D': 86400
}


# get request response from exchange api
class Return_API_response:
    """Get data from BitFinex API. Bitfinex rate limit policy can vary in a
     range of 10 to 90 requests per minute. So if we get a 429 response wait
     for a minute"""
    def __init__(self):
        self.sesh = requests.Session()

    def api_response(self, url):
        try:
            res = self.sesh.get(url)
            while res.status_code == 429:
                log.info(f'{res.status_code} {url[40:]}')
                # wait for Bitfinex
                time.sleep(62)
                res = self.sesh.get(url)

            data = res.json()
            res.raise_for_status()
        except requests.exceptions.HTTPError:
            log.error("Requsts error ", exc_info=True)
            # Do something here to fix error
            return False

        if data:
            return data
        log.error(f'Failed to get data from BitFinex: {url[40:]}')
        return False

    def close_session(self):
        self.sesh.close()


# Collect candle Data from Bitfinex and store it in the Dict. candles
def _getCandlesBFX_2(coins, delta, limit):
    candles = {}
    resp = Return_API_response()
    for coin in coins:
        data = resp.api_response(bitfinex + '{0}:t{1}/hist?limit={2}'.
                                format(delta, coin, limit))
        if data:
            df = pd.DataFrame(data=data, columns=(
                'MTS open close high low volume').split())
            df.set_index('MTS', drop=True, inplace=True)
            df.index = pd.to_datetime(df.index, unit='ms')
            df.name = coin[:3]
            df.index.name = 'MTS'
            candles[coin[:3]] = df
    resp.close_session()
    return candles


# Get the latest batch of candles per time interval check limit to insure we
# have data time overlap
def _latest(coins, interval, limit):
    for delta in interval:
        candles = _getCandlesBFX_2(coins, delta, limit)
        for candle in candles:
            path = 'Data\\{0}\\{1}_Candles_{2}.csv'.format(delta, candles[candle].name[:3], delta)
            filepath = pkg_resources.resource_filename(__name__, path)
            # store data descending earliest at top/begining
            candles[candle][::-1].to_csv(filepath)


# update Master Data with latest data
def _update(coins, interval, sort=False):
    for delta in interval:
        for sym in coins:
            path = '{0}\\{1}_Candles_{2}.csv'.format(delta, sym[:3], delta)
            newDataPath = 'Data\\{}'.format(path)
            masterPath = 'Data\\MasterData\\{}'.format(path)
            backupPath = 'Data\\Master_old\\{}'.format(path)
            old = getData(sym[:3], delta)
            new = pd.read_csv(pkg_resources.resource_filename(__name__, newDataPath),
                              index_col='MTS', parse_dates=True)
            update = pd.concat([old, new]).drop_duplicates()
            master = update.groupby('MTS')['open', 'close', 'high', 'low', 'volume'].mean()
            if sort:
                master.sort_index()
            old.to_csv(pkg_resources.resource_filename(__name__, backupPath))
            master.to_csv(pkg_resources.resource_filename(__name__, masterPath))


# Only need to use this once to bring some new coins from Bitfinex
def _newCoins(coins=takeAway, interval=intervals, limit=1000):
    _latest(coins, interval, limit)
    for delta in interval:
        for coin in coins:
            path = '{0}\\{1}_Candles_{2}.csv'.format(delta, coin[:3], delta)
            newDataPath = 'Data\\{}'.format(path)
            masterPath = 'Data\\MasterData\\{}'.format(path)
            oldPath = 'Data\\Master_old\\{}'.format(path)
            master = pd.read_csv(pkg_resources.resource_filename(__name__, newDataPath),
                              index_col='MTS', parse_dates=True)
            master.to_csv(pkg_resources.resource_filename(__name__, masterPath))
            master.to_csv(pkg_resources.resource_filename(__name__, oldPath))


# Update coin MasterData
def nextUpdate(coins, step, limit):
    _latest(coins, [step], limit)
    _update(coins, [step])


# get candle data for a coin
def getOneBITFINEX(sym, step):
    candles = _getCandlesBFX_2([sym], step, 1000)
    for candle in candles:
        path = 'Data\\{0}\\{1}_Candles_{2}.csv'.format(step, sym[:3], step)
        filepath = pkg_resources.resource_filename(__name__, path)
        candles[candle][::-1].to_csv(filepath)

    return candles[candle][::-1]


from .getData import getData


def updateAll(coins):
    for delta in intervals:
        coin = 'BTCUSD'
        path = 'Data\\MasterData\\{0}\\{1}_Candles_{2}.csv'.format(delta, coin[:3], delta)
        test = pd.read_csv(pkg_resources.resource_filename(__name__, path),
                     index_col='MTS', parse_dates=True)
        limit = int((datetime.utcnow() - test.index[-1]).total_seconds()/intervalsInSecounds[delta])
        print(f'limit for {delta} = {limit}')
        if limit > 0 and limit <= 1000:
            nextUpdate(coins, delta, limit)
            time.sleep(65)
