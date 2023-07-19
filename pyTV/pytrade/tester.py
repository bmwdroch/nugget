import os

import pandas as pd


class Tester():
    def __init__(self, exchange, strategy):
        self.exchange = exchange
        self.strategy = strategy
        self.header = '{0} - {1} - {2} - {3}'.format(
            self.exchange.NAME, self.exchange.symbol, self.exchange.interval,
            self.strategy['name'].capitalize().replace('_', ' ')
        )
        self.file = os.path.abspath(
            'strategies/{}/statistics_1.html'.format(self.strategy['name'])
        )
        while True:
            try:
                open(self.file, 'r')
            except:
                break
            else:
                file = self.file.rstrip('.html')
                number =  int(file[file.rfind('statistics_') + 11:]) + 1
                self.file = (file[:file.rfind('statistics_') + 11] +
                             str(number) + '.html')

    def log_preprocessing(self, log):
        deal_type_keywords = {
            0: 'long',
            1: 'short'
        }
        entry_signal_keywords = {
            0: 'Long',
            1: 'Short'
        }
        exit_signal_keywords = {
            0: 'Liquidation',
            1: 'Stop-loss',
            2: 'Take-profit #1',
            3: 'Take-profit #2',
            4: 'Take-profit #3',
            5: 'Take-profit #4',
            6: 'Take-profit #5',
            7: 'Take-profit #6',
            8: 'Take-profit #7',
            9: 'Take-profit #8',
            10: 'Take-profit #9',
            11: 'Take-profit #10'
        }
        log = pd.DataFrame(
            log.reshape(log.shape[0] // 11, 11),
            columns=[
                'deal_type', 'entry_signal', 'exit_signal',
                'entry_date', 'exit_date', 'entry_price',
                'exit_price', 'position_size', 'pnl',
                'pnl_per', 'commission'
            ]
        )

        for key, values in deal_type_keywords.items():
            log['deal_type'] = log['deal_type'].replace(key, values)

        for key, values in entry_signal_keywords.items():
            log['entry_signal'] = log['entry_signal'].replace(key, values)

        for key, values in exit_signal_keywords.items():
            log['exit_signal'] = log['exit_signal'].replace(key, values)

        log['entry_date'] = pd.to_datetime(
            pd.to_numeric(log['entry_date']),
            unit='ms'
        ).dt.strftime('%Y-%m-%d %H:%M')
        log['exit_date'] = pd.to_datetime(
            pd.to_numeric(log['exit_date']),
            unit='ms'
        ).dt.strftime('%Y-%m-%d %H:%M')
        return log

    def calculate_performance(self, log, initial_capital):
        equity = pd.concat(
            [pd.Series(initial_capital), log.pnl],
            ignore_index=True  
        ).cumsum()
        equity.index += 1

        try:
            gross_profit = log.loc[:, ['deal_type','pnl']]. \
                query('pnl > 0').groupby('deal_type'). \
                    agg('sum', numeric_only=True)
        except:
            pass
        try:
            all_gross_profit = round(
                gross_profit.pnl.sum(), 2
            )
            all_gross_profit_per = round(
                all_gross_profit / initial_capital * 100, 2
            )
        except:
             all_gross_profit = 0.0
             all_gross_profit_per = 0.0
        try:
            long_gross_profit = round(
                gross_profit.at['long', 'pnl'], 2
            )
            long_gross_profit_per = round(
                long_gross_profit / initial_capital * 100, 2
            )
        except:
            long_gross_profit = 0.0
            long_gross_profit_per = 0.0
        try:
            short_gross_profit = round(
                gross_profit.at['short', 'pnl'], 2
            )
            short_gross_profit_per = round(
                short_gross_profit / initial_capital * 100, 2
            )
        except:
            short_gross_profit = 0.0
            short_gross_profit_per = 0.0

        try:
            gross_loss = log.loc[:, ['deal_type','pnl']]. \
                query('pnl <= 0').groupby('deal_type'). \
                    agg('sum', numeric_only=True)
        except:
            pass
        try:
            all_gross_loss = round(
                abs(gross_loss.pnl.sum()), 2
            )
            all_gross_loss_per = round(
                all_gross_loss / initial_capital * 100, 2
            )
        except:
            all_gross_loss = 0.0
            all_gross_loss_per = 0.0
        try:
            long_gross_loss = round(
                abs(gross_loss.at['long', 'pnl']), 2
            )
            long_gross_loss_per = round(
                long_gross_loss / initial_capital * 100, 2
            )
        except:
            long_gross_loss = 0.0
            long_gross_loss_per = 0.0
        try:
            short_gross_loss = round(
                abs(gross_loss.at['short', 'pnl']), 2
            )
            short_gross_loss_per = round(
                short_gross_loss / initial_capital * 100, 2
            )
        except:
            short_gross_loss = 0.0
            short_gross_loss_per = 0.0

        try:
            all_net_profit = round(
                all_gross_profit - all_gross_loss, 2
            )
            all_net_profit_per = round(
                all_net_profit / initial_capital * 100, 2
            )
        except:
            all_net_profit = 0.0
            all_net_profit_per = 0.0
        try:
            long_net_profit = round(
                long_gross_profit - long_gross_loss, 2
            )
            long_net_profit_per = round(
                long_net_profit / initial_capital * 100, 2
            )
        except:
            long_net_profit = 0.0
            long_net_profit_per = 0.0
        try:
            short_net_profit = round(
                short_gross_profit - short_gross_loss, 2
            )
            short_net_profit_per = round(
                short_net_profit / initial_capital * 100, 2
            )
        except:
            short_net_profit = 0.0
            short_net_profit_per = 0.0

        try:
            drawdowns = []
            drawdowns_per = []
            max_equity = equity.iloc[0]

            for i in range(1, equity.shape[0]):
                if equity.iloc[i] > max_equity:
                    max_equity = equity.iloc[i]

                if equity.iloc[i] < equity.iloc[i - 1]:
                    min_equity = equity.iloc[i]
                    drawdown = min_equity - max_equity
                    drawdown_per = (min_equity / max_equity - 1) * 100
                    drawdowns.append(drawdown)
                    drawdowns_per.append(drawdown_per)
            all_max_drawdown = round(abs(min(drawdowns)), 2)
            all_max_drawdown_per = round(abs(min(drawdowns_per)), 2)
        except:
            all_max_drawdown = 0.0
            all_max_drawdown_per = 0.0

        try:
            all_skew = round(log.pnl_per.skew(), 3)
        except:
            all_skew = None

        try:
            if all_gross_loss != 0:
                all_profit_factor = round(
                    all_gross_profit / all_gross_loss, 3
                )
            else:
                all_profit_factor = None
        except:
            all_profit_factor = None
        try:
            if long_gross_loss != 0:
                long_profit_factor =  round(
                    long_gross_profit / long_gross_loss, 3
                )
            else:
                long_profit_factor = None
        except:
            long_profit_factor = None
        try:
            if short_gross_loss != 0:
                short_profit_factor = round(
                    short_gross_profit / short_gross_loss, 3
                )
            else:
                short_profit_factor = None
        except:
            short_profit_factor = None

        try:
            commission_paid = log.loc[:, ['deal_type','commission']]. \
                groupby('deal_type').agg('sum', numeric_only=True)
        except:
            pass
        try:
            all_commission_paid = round(
                commission_paid.commission.sum(), 2
            )
        except:
            all_commission_paid = 0.0
        try:
            long_commission_paid = round(
                commission_paid.at['long', 'commission'], 2
            )
        except:
            long_commission_paid = 0.0
        try:
            short_commission_paid = round(
                commission_paid.at['short', 'commission'], 2
            )
        except:
            short_commission_paid = 0.0

        all_total_closed_trades = int(log.shape[0])
        long_total_closed_trades = int(
            log.query('deal_type == "long"').shape[0]
        )
        short_total_closed_trades = int(
            log.query('deal_type == "short"').shape[0]
        )

        all_number_winning_trades = int(log.query('pnl > 0').shape[0])
        long_number_winning_trades = int(
            log.query('deal_type == "long" and pnl > 0').shape[0]
        )
        short_number_winning_trades = int(
            log.query('deal_type == "short" and pnl > 0').shape[0]
        )

        all_number_losing_trades = int(log.query('pnl <= 0').shape[0])
        long_number_losing_trades = int(
            log.query('deal_type == "long" and pnl <= 0').shape[0]
        )
        short_number_losing_trades = int(
            log.query('deal_type == "short" and pnl <= 0').shape[0]
        )

        try:
            all_percent_profitable = round(
                all_number_winning_trades / 
                    all_total_closed_trades * 100,
                2
            )
        except:
            all_percent_profitable = None
        try:
            long_percent_profitable = round(
                long_number_winning_trades / 
                    long_total_closed_trades * 100,
                2
            )
        except:
            long_percent_profitable = None
        try:
            short_percent_profitable = round(
                short_number_winning_trades / 
                    short_total_closed_trades * 100,
                2
            )
        except:
            short_percent_profitable = None

        try:
            all_avg_trade = round(log.pnl.mean(), 2)
            all_avg_trade_per = round(log.pnl_per.mean(), 2)
        except:
            all_avg_trade = None
        try:
            long_avg_trade = round(
                log.query('deal_type == "long"').pnl.mean(), 2
            )
            long_avg_trade_per = round(
                log.query('deal_type == "long"').pnl_per.mean(), 2
            )
        except:
            long_avg_trade = None
        try:
            short_avg_trade = round(
                log.query('deal_type == "short"').pnl.mean(), 2
            )
            short_avg_trade_per = round(
                log.query('deal_type == "short"').pnl_per.mean(), 2
            )
        except:
            short_avg_trade = None

        try:
            all_avg_winning_trade = round(
                log.query('pnl > 0').pnl.mean(), 2
            )
            all_avg_winning_trade_per = round(
                log.query('pnl_per > 0').pnl_per.mean(), 2
            )
        except:
            all_avg_winning_trade = None
        try:
            long_avg_winning_trade = round(
                log.query('deal_type == "long" and pnl > 0'). \
                    pnl.mean(), 2
            )
            long_avg_winning_trade_per = round(
                log.query('deal_type == "long" and pnl_per > 0'). \
                    pnl_per.mean(),
                2
            )
        except:
            long_avg_winning_trade = None
        try:
            short_avg_winning_trade = round(
                log.query('deal_type == "short" and pnl > 0'). \
                    pnl.mean(), 2
            )
            short_avg_winning_trade_per = round(
                log.query('deal_type == "short" and pnl_per > 0'). \
                    pnl_per.mean(),
                2
            )
        except:
            short_avg_winning_trade = None

        try:
            all_avg_losing_trade = round(
                abs(log.query('pnl <= 0').pnl.mean()), 2
            )
            all_avg_losing_trade_per = round(
                abs(log.query('pnl_per <= 0').pnl_per.mean()), 2
            )
        except:
            all_avg_losing_trade = None
        try:
            long_avg_losing_trade = round(
                abs(log.query('deal_type == "long" and pnl <= 0'). \
                    pnl.mean()),
                2
            )
            long_avg_losing_trade_per = round(
                abs(log.query(
                    'deal_type == "long" and pnl_per <= 0'
                ).pnl_per.mean()),
                2
            )
        except:
            long_avg_losing_trade = None
        try:
            short_avg_losing_trade = round(
                abs(log.query('deal_type == "short" and pnl <= 0'). \
                    pnl.mean()),
                2
            )
            short_avg_losing_trade_per = round(
                abs(log.query(
                    'deal_type == "short" and pnl_per <= 0'
                ).pnl_per.mean()),
                2
            )
        except:
            short_avg_losing_trade = None

        try:
            all_ratio_avg_win_loss = round(
                all_avg_winning_trade / all_avg_losing_trade, 3
            )
        except:
            all_ratio_avg_win_loss = None
        try:
            long_ratio_avg_win_loss = round(
                long_avg_winning_trade / long_avg_losing_trade, 3
            )
        except:
            long_ratio_avg_win_loss = None
        try:
            short_ratio_avg_win_loss = round(
                short_avg_winning_trade / short_avg_losing_trade, 3
            )
        except:
            short_ratio_avg_win_loss = None

        try:
            all_sortino_ratio = round(
                all_avg_trade_per / ((log.query('pnl_per <= 0'). \
                    pnl_per ** 2).mean() ** 0.5),
                3
            )
        except:
            all_sortino_ratio = None

        try:
            all_largest_winning_trade = round(
                log.query('pnl > 0').max().loc['pnl'], 2
            )
            all_largest_winning_trade_per = round(
                log.query('pnl_per > 0').max().loc['pnl_per'], 2
            )
        except:
            all_largest_winning_trade = None
        try:
            long_largest_winning_trade = round(
                log.query('deal_type == "long" and pnl > 0'). \
                    max().loc['pnl'], 
                2
            )
            long_largest_winning_trade_per = round(
                log.query('deal_type == "long" and pnl_per > 0'). \
                    max().loc['pnl_per'], 
                2
            )
        except:
            long_largest_winning_trade = None
        try:
            short_largest_winning_trade = round(
                log.query('deal_type == "short" and pnl > 0'). \
                    max().loc['pnl'], 
                2
            )
            short_largest_winning_trade_per = round(
                log.query('deal_type == "short" and pnl_per > 0'). \
                    max().loc['pnl_per'], 
                2
            )
        except:
            short_largest_winning_trade = None

        try:
            all_largest_losing_trade = round(
                abs(log.query('pnl <= 0').min().loc['pnl']), 2
            )
            all_largest_losing_trade_per = round(
                abs(log.query('pnl_per <= 0').min().loc['pnl_per']), 2
            )
        except:
            all_largest_losing_trade = None
        try:
            long_largest_losing_trade = round(
                abs(log.query(
                    'deal_type == "long" and pnl <= 0'
                ).min().loc['pnl']),
                2
            )
            long_largest_losing_trade_per = round(
                abs(log.query(
                    'deal_type == "long" and pnl_per <= 0'
                ).min().loc['pnl_per']),
                2
            )
        except:
            long_largest_losing_trade = None
        try:
            short_largest_losing_trade = round(
                abs(log.query(
                    'deal_type == "short" and pnl <= 0'
                ).min().loc['pnl']),
                2
            )
            short_largest_losing_trade_per = round(
                abs(log.query(
                    'deal_type == "short" and pnl_per <= 0'
                ).min().loc['pnl_per']),
                2
            )
        except:
            short_largest_losing_trade = None

        metrics = pd.DataFrame(
            {
                'Title': [
                    'Net profit, USDT',
                    'Net profit, %',
                    'Gross profit, USDT',
                    'Gross profit, %',
                    'Gross loss, USDT',
                    'Gross loss, %',
                    'Max drawdown, USDT',
                    'Max drawdown, %',
                    'Sortino ratio',
                    'Skew',
                    'Profit factor',
                    'Commission paid, USDT',
                    'Total closed trades',
                    'Number winning trades',
                    'Number losing trades',
                    'Percent profitable',
                    'Avg trade, USDT',
                    'Avg trade, %',
                    'Avg winning trade, USDT',
                    'Avg winning trade, %',
                    'Avg losing trade, USDT',
                    'Avg losing trade, %',
                    'Ratio avg win / avg loss',
                    'Largest winning trade, USDT',
                    'Largest winning trade, %',
                    'Largest losing trade, USDT',
                    'Largest losing trade, %',
            ],
                'All': [
                    all_net_profit,
                    all_net_profit_per,
                    all_gross_profit,
                    all_gross_profit_per,
                    all_gross_loss,
                    all_gross_loss_per,
                    all_max_drawdown,
                    all_max_drawdown_per,
                    all_sortino_ratio,
                    all_skew,
                    all_profit_factor,
                    all_commission_paid,
                    str(all_total_closed_trades),
                    str(all_number_winning_trades),
                    str(all_number_losing_trades),
                    all_percent_profitable,
                    all_avg_trade,
                    all_avg_trade_per,
                    all_avg_winning_trade,
                    all_avg_winning_trade_per,
                    all_avg_losing_trade,
                    all_avg_losing_trade_per,
                    all_ratio_avg_win_loss,
                    all_largest_winning_trade,
                    all_largest_winning_trade_per,
                    all_largest_losing_trade,
                    all_largest_losing_trade_per
                ],
                'Long': [
                    long_net_profit,
                    long_net_profit_per,
                    long_gross_profit,
                    long_gross_profit_per,
                    long_gross_loss,
                    long_gross_loss_per,
                    '',
                    '',
                    '',
                    '',
                    long_profit_factor,
                    long_commission_paid,
                    str(long_total_closed_trades),
                    str(long_number_winning_trades),
                    str(long_number_losing_trades),
                    long_percent_profitable,
                    long_avg_trade,
                    long_avg_trade_per,
                    long_avg_winning_trade,
                    long_avg_winning_trade_per,
                    long_avg_losing_trade,
                    long_avg_losing_trade_per,
                    long_ratio_avg_win_loss,
                    long_largest_winning_trade,
                    long_largest_winning_trade_per,
                    long_largest_losing_trade,
                    long_largest_losing_trade_per
                ],
                'Short': [
                    short_net_profit,
                    short_net_profit_per,
                    short_gross_profit,
                    short_gross_profit_per,
                    short_gross_loss,
                    short_gross_loss_per,
                    '',
                    '',
                    '',
                    '',
                    short_profit_factor,
                    short_commission_paid,
                    str(short_total_closed_trades),
                    str(short_number_winning_trades),
                    str(short_number_losing_trades),
                    short_percent_profitable,
                    short_avg_trade,
                    short_avg_trade_per,
                    short_avg_winning_trade,
                    short_avg_winning_trade_per,
                    short_avg_losing_trade,
                    short_avg_losing_trade_per,
                    short_ratio_avg_win_loss,
                    short_largest_winning_trade,
                    short_largest_winning_trade_per,
                    short_largest_losing_trade,
                    short_largest_losing_trade_per
                ]
            }
        )
        metrics = metrics.fillna('').astype(str)
        return metrics
        

    def create_tables(self, log, metrics):
        with open(self.file, 'w') as f:
            print('<!DOCTYPE html>', file=f)
            print('<html lang="en">', file=f)
            print('  <head>', file=f)
            print('    <meta charset="UTF-8" />', file=f)
            print('    <meta http-equiv="X-UA-Compatible" content="IE=edge" />', file=f)
            print('    <title>Statistics</title>', file=f)
            print('    <link rel="stylesheet" href="../../pytrade/bootstrap/bootstrap.css" />', file=f)
            print('  </head>', file=f)
            print('  <body>', file=f)
            print('    <h4 class="text-center my-2">', file=f)
            print('      ' + self.header, file=f)
            print('    </h4>', file=f)
            print('    <div class="accordion accordion-flush">', file=f)
            print('      <div class="accordion-item">', file=f)
            print('        <h2 class="accordion-header">', file=f)
            print('          <button', file=f)
            print('            class="accordion-button collapsed"', file=f)
            print('            type="button"', file=f)
            print('            data-bs-toggle="collapse"', file=f)
            print('            data-bs-target="#flush-collapseOne"', file=f)
            print('            aria-expanded="false"', file=f)
            print('            aria-controls="flush-collapseOne"', file=f)
            print('          >', file=f)
            print('            List of trades', file=f)
            print('          </button>', file=f)
            print('        </h2>', file=f)
            print('        <div id="flush-collapseOne" class="accordion-collapse collapse">', file=f)
            print('          <div class="accordion-body">', file=f)
            print('            <table class="table table-bordered align-middle">', file=f)
            print('              <thead class="table-dark">', file=f)
            print('                <tr>', file=f)
            print('                  <th scope="col">Trade #</th>', file=f)
            print('                  <th scope="col">Type</th>', file=f)
            print('                  <th scope="col">Signal</th>', file=f)
            print('                  <th scope="col">Date/time</th>', file=f)
            print('                  <th class="text-end" scope="col">Price</th>', file=f)
            print('                  <th class="text-end" scope="col">Quantity</th>', file=f)
            print('                  <th class="text-end" scope="col">Profit</th>', file=f)
            print('                </tr>', file=f)
            print('              </thead>', file=f)
            print('              <tbody>', file=f)
            
            for i in range(log.shape[0]):
                print('                <tr>', file=f)
                print('                  <td rowspan="2">{}</td>'.format(i + 1), file=f)
                print('                  <td>Exit {}</td>'.format(log.iat[i, 0]), file=f)
                print('                  <td>{}</td>'.format(log.iat[i, 2]), file=f)
                print('                  <td>{}</td>'.format(log.iat[i, 4]), file=f)
                print('                  <td class="text-end">{} USDT</td>'.format(log.iat[i, 6]), file=f)
                print('                  <td class="text-end" rowspan="2">{0} {1}</td>'.format(
                    log.iat[i, 7], self.exchange.symbol[:self.exchange.symbol.rfind('USDT')]), file=f)
                print('                  <td class="text-end" rowspan="2">', file=f)
                print('                    <div>{} USDT</div>'.format(log.iat[i, 8]), file=f)
                print('                    <div>{}%</div>'.format(log.iat[i, 9]), file=f)
                print('                  </td>', file=f)
                print('                </tr>', file=f)
                print('                <tr>', file=f)
                print('                  <td>Entry {}</td>'.format(log.iat[i, 0]), file=f)
                print('                  <td>{}</td>'.format(log.iat[i, 1]), file=f)
                print('                  <td>{}</td>'.format(log.iat[i, 3]), file=f)
                print('                  <td class="text-end">{} USDT</td>'.format(log.iat[i, 5]), file=f)
                print('                </tr>', file=f)

            print('              </tbody>', file=f)
            print('            </table>', file=f)
            print('          </div>', file=f)
            print('        </div>', file=f)
            print('      </div>', file=f)
            print('      <div class="accordion-item">', file=f)
            print('        <h2 class="accordion-header">', file=f)
            print('          <button', file=f)
            print('            class="accordion-button collapsed"', file=f)
            print('            type="button"', file=f)
            print('            data-bs-toggle="collapse"', file=f)
            print('            data-bs-target="#flush-collapseTwo"', file=f)
            print('            aria-expanded="false"', file=f)
            print('            aria-controls="flush-collapseTwo"', file=f)
            print('          >', file=f)
            print('            Performance summary', file=f)
            print('          </button>', file=f)
            print('        </h2>', file=f)
            print('        <div id="flush-collapseTwo" class="accordion-collapse collapse">', file=f)
            print('          <div class="accordion-body">', file=f)
            print('            <table class="table table-hover table-bordered align-middle">', file=f)
            print('              <thead class="table-dark">', file=f)
            print('                <tr>', file=f)
            print('                  <th scope="col">Title</th>', file=f)
            print('                  <th class="text-end" scope="col">All</th>', file=f)
            print('                  <th class="text-end" scope="col">Long</th>', file=f)
            print('                  <th class="text-end" scope="col">Short</th>', file=f)
            print('                </tr>', file=f)
            print('              </thead>', file=f)
            print('              <tbody>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Net profit</td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[0, 1]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[1, 1]), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[0, 2]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[1, 2]), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[0, 3]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[1, 3]), file=f)
            print('                  </td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Gross profit</td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[2, 1]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[3, 1]), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[2, 2]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[3, 2]), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[2, 3]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[3, 3]), file=f)
            print('                  </td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Gross loss</td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[4, 1]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[5, 1]), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[4, 2]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[5, 2]), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[4, 3]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[5, 3]), file=f)
            print('                  </td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Profit factor</td>', file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[10, 1]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[10, 2]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[10, 3]), file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Commission paid</td>', file=f)
            print('                  <td class="text-end">{} USDT</td>'.format(metrics.iat[11, 1]), file=f)
            print('                  <td class="text-end">{} USDT</td>'.format(metrics.iat[11, 2]), file=f)
            print('                  <td class="text-end">{} USDT</td>'.format(metrics.iat[11, 3]), file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Total closed trades</td>', file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[12, 1]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[12, 2]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[12, 3]), file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Number winning trades</td>', file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[13, 1]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[13, 2]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[13, 3]), file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Number losing trades</td>', file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[14, 1]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[14, 2]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[14, 3]), file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Percent profitable</td>', file=f)
            print('                  <td class="text-end">{0}{1}</td>'.format(metrics.iat[15, 1], '%' if metrics.iat[15, 1] != '' else ''), file=f)
            print('                  <td class="text-end">{0}{1}</td>'.format(metrics.iat[15, 2], '%' if metrics.iat[15, 2] != '' else ''), file=f)
            print('                  <td class="text-end">{0}{1}</td>'.format(metrics.iat[15, 3], '%' if metrics.iat[15, 3] != '' else ''), file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Avg trade</td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[16, 1], 'USDT' if metrics.iat[16, 1] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[17, 1], '%' if metrics.iat[17, 1] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[16, 2], 'USDT' if metrics.iat[16, 2] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[17, 2], '%' if metrics.iat[17, 2] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[16, 3], 'USDT' if metrics.iat[16, 3] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[17, 3], '%' if metrics.iat[17, 3] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Avg winning trade</td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[18, 1], 'USDT' if metrics.iat[18, 1] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[19, 1], '%' if metrics.iat[19, 1] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[18, 2], 'USDT' if metrics.iat[18, 2] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[19, 2], '%' if metrics.iat[19, 2] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[18, 3], 'USDT' if metrics.iat[18, 3] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[19, 3], '%' if metrics.iat[19, 3] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Avg losing trade</td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[20, 1], 'USDT' if metrics.iat[20, 1] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[21, 1], '%' if metrics.iat[21, 1] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[20, 2], 'USDT' if metrics.iat[20, 2] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[21, 2], '%' if metrics.iat[21, 2] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[20, 3], 'USDT' if metrics.iat[20, 3] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[21, 3], '%' if metrics.iat[21, 3] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Ratio avg win / avg loss</td>', file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[22, 1]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[22, 2]), file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[22, 3]), file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Largest winning trade</td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[23, 1], 'USDT' if metrics.iat[23, 1] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[24, 1], '%' if metrics.iat[24, 1] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[23, 2], 'USDT' if metrics.iat[23, 2] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[24, 2], '%' if metrics.iat[24, 2] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[23, 3], 'USDT' if metrics.iat[23, 3] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[24, 3], '%' if metrics.iat[24, 3] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Largest losing trade</td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[25, 1], 'USDT' if metrics.iat[25, 1] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[26, 1], '%' if metrics.iat[26, 1] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[25, 2], 'USDT' if metrics.iat[25, 2] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[26, 2], '%' if metrics.iat[26, 2] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{0} {1}</div>'.format(metrics.iat[25, 3], 'USDT' if metrics.iat[25, 3] != '' else ''), file=f)
            print('                    <div>{0}{1}</div>'.format(metrics.iat[26, 3], '%' if metrics.iat[26, 3] != '' else ''), file=f)
            print('                  </td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Max drawdown</td>', file=f)
            print('                  <td class="text-end">', file=f)
            print('                    <div>{} USDT</div>'.format(metrics.iat[6, 1]), file=f)
            print('                    <div>{}%</div>'.format(metrics.iat[7, 1]), file=f)
            print('                  </td>', file=f)
            print('                  <td></td>', file=f)
            print('                  <td></td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Sortino ratio</td>', file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[8, 1]), file=f)
            print('                  <td></td>', file=f)
            print('                  <td></td>', file=f)
            print('                </tr>', file=f)

            print('                <tr>', file=f)
            print('                  <td>Skew</td>', file=f)
            print('                  <td class="text-end">{}</td>'.format(metrics.iat[9, 1]), file=f)
            print('                  <td></td>', file=f)
            print('                  <td></td>', file=f)
            print('                </tr>', file=f)

            print('              </tbody>', file=f)
            print('            </table>', file=f)
            print('          </div>', file=f)
            print('        </div>', file=f)
            print('      </div>', file=f)
            print('    </div>', file=f)
            print('    <script src="../../pytrade/bootstrap/bootstrap.js"></script>', file=f)
            print('  </body>', file=f)
            print('</html>', file=f)

    def start(self):
        strategy = self.strategy['class'](self.exchange)
        initial_capital = strategy.initial_capital
        log = self.log_preprocessing(strategy.log)

        if log.shape[0] > 0:
            metrics = self.calculate_performance(log, initial_capital)
            self.create_tables(log, metrics)