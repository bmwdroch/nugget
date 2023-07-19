import datetime as dt
import time

import binance as bn
import numpy as np


class BinanceAPI():
    NAME = 'Binance linear futures'
    KLINE_INTERVALS = {
        '1m': '1m', 1: '1m',
        '3m': '3m', 3: '3m',
        '5m': '5m', 5: '5m',
        '15m': '15m', 15: '15m',
        '30m': '30m', 30: '30m',
        '1h': '1h', 60: '1h',
        '2h': '2h', 120: '2h',
        '4h': '4h', 240: '4h',
        '6h': '6h', 360: '6h',
        '12h': '12h', 720: '12h',
        '1d': '1d', 'D': '1d',
        '1w': '1w', 'W': '1w',
        '1M': '1M', 'M': '1M'   
    }

    def __init__(self, api_key, secret_key):
        self.client = bn.Client()

    def get_data(self, symbol, interval, start, end):
        self.symbol = symbol
        self.interval = interval
        self.start = dt.datetime.strptime(
            start, '%Y/%m/%d %H:%M %z').strftime('%Y-%m-%d %H:%M')
        self.end = dt.datetime.strptime(
            end, '%Y/%m/%d %H:%M %z').strftime('%Y-%m-%d %H:%M')

        interval = BinanceAPI.KLINE_INTERVALS[interval]
        start = int(
            dt.datetime.strptime(start, '%Y/%m/%d %H:%M %z').timestamp()
        ) * 1000
        end = int(
            dt.datetime.strptime(end, '%Y/%m/%d %H:%M %z').timestamp()
        ) * 1000
        self.price_data = np.array(
            self.client.get_historical_klines(
                symbol, interval, start, end,
                klines_type=bn.enums.HistoricalKlinesType(2)
            )
        )[:-1, :5].astype(float)
        self.price_precision = self.get_price_precision(symbol)
        self.qty_precision = self.get_qty_precision(symbol)
    
    def get_price_precision(self, symbol):
        symbols_info = self.client.futures_exchange_info()['symbols']
        symbol_info = next(
            filter(lambda x: x['symbol'] == symbol, symbols_info))
        return float(symbol_info['filters'][0]['tickSize'])

    def get_qty_precision(self, symbol):
        symbols_info = self.client.futures_exchange_info()['symbols']
        symbol_info = next(
            filter(lambda x: x['symbol'] == symbol, symbols_info))
        return float(symbol_info['filters'][1]['minQty'])