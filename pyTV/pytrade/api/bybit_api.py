import datetime as dt
import time

from pybit.unified_trading import HTTP
import numpy as np


class BybitAPI():
    NAME = 'Bybit linear futures'
    KLINE_INTERVALS = {
        1: 1, '1m': 1,
        3: 3, '3m': 3,
        5: 5, '5m': 5,
        15: 15, '15m': 15,
        30: 30, '30m': 30,
        60: 60, '1h': 60,
        120: 120, '2h': 120,
        240: 240, '4h': 240,
        360: 360, '6h': 360,
        720: 720, '12h': 720,
        'D': 'D', '1d': 'D',
        'W': 'W', '1w': 'W',
        'M': 'M', '1M': 'M'
    }

    def __init__(self, api_key, secret_key):
        self.client = HTTP()

    def get_data(self, symbol, interval, start, end):
        self.symbol = symbol
        self.interval = interval
        self.start = dt.datetime.strptime(
            start, '%Y/%m/%d %H:%M %z').strftime('%Y-%m-%d %H:%M')
        self.end = dt.datetime.strptime(
            end, '%Y/%m/%d %H:%M %z').strftime('%Y-%m-%d %H:%M')

        interval = BybitAPI.KLINE_INTERVALS[interval]
        start = int(
            dt.datetime.strptime(start, '%Y/%m/%d %H:%M %z').timestamp()
        ) * 1000
        end = int(
            dt.datetime.strptime(end, '%Y/%m/%d %H:%M %z').timestamp()
        ) * 1000
        price_data = np.array(
            self.client.get_kline(
                category='linear',
                symbol=symbol,
                interval=interval,
                start=start
            )['result']['list']
        )[:0:-1, :5].astype(float)

        while True:
            time.sleep(0.01)
            start = int(price_data[-1, 0])

            if start >= end:
                price_data = price_data[
                    : np.where(price_data[:, 0] == end)[0][0]
                ]
                break
            else:
                last_price_data = np.array(
                    self.client.get_kline(
                        category='linear',
                        symbol=symbol,
                        interval=interval,
                        start=start
                    )['result']['list']
                )[-2:0:-1, :5].astype(float)

                if last_price_data.shape[0] > 0:
                    price_data = np.vstack((price_data, last_price_data))
                else:
                    break

        self.price_data = price_data
        self.price_precision = self.get_price_precision(symbol)
        self.qty_precision = self.get_qty_precision(symbol)
    
    def get_price_precision(self, symbol):
        symbol_info = self.client.get_instruments_info(
            category="linear", symbol=symbol
        )['result']['list'][0]
        return float(symbol_info['priceFilter']['tickSize'])

    def get_qty_precision(self, symbol):
        symbol_info = self.client.get_instruments_info(
            category="linear", symbol=symbol
        )['result']['list'][0]
        return float(symbol_info['lotSizeFilter']['qtyStep'])