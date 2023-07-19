import time

from pytrade import BinanceAPI, BybitAPI, GA, Tester


class Manager:
    @staticmethod
    def manage(api_key, secret_key, start, end, symbol, 
               interval, execution, exchange, strategy):
        start_time = time.time()
        print('The program has started.')

        if exchange == 'binance':
            print('Requesting data from the Binance.')
            Manager.client = BinanceAPI(api_key, secret_key)
        elif exchange == 'bybit':
            print('Requesting data from the Bybit.')
            Manager.client = BybitAPI(api_key, secret_key)

        Manager.client.get_data(symbol, interval, start, end)
        print('Data received.')

        if execution == 'optimization':
           print('Optimization in progress.')
           GA(Manager.client, strategy).start()
           print('Optimization completed.')
        elif execution == 'backtesting':
            print('Backtest in progress.')
            Tester(Manager.client, strategy).start()
            print('Backtest completed.')
        elif execution == 'automation':
            print('Automation in progress.')

        end_time = time.time()
        print('Program execution time:', (end_time - start_time) * 1000, 'ms.')