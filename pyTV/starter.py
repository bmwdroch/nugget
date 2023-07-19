from pytrade.manager import Manager
from strategies import NuggetV3


# inputs
api_key = ''
secret_key = ''
start = '2022/01/01 00:00 +0000'
end = '2025/01/01 00:00 +0000'
symbol = 'BTCUSDT'
interval = '1h'
execution = 2
exchange = 1
strategy = 3

executions = {
    1: 'optimization',
    2: 'backtesting'
}
exchangies = {
    1: 'binance',
    2: 'bybit'
}
strategies = {
    3: {'name': 'nugget_v3', 'class': NuggetV3}
}


def main():
    Manager.manage(api_key, secret_key, start, end, symbol, interval,
                   executions[execution], exchangies[exchange],
                   strategies[strategy])
 

if __name__ == '__main__':
    main()