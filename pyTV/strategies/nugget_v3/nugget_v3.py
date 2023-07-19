import random as rand

import numpy as np
import numba as nb

import pytrade.math as math
import pytrade.ta as ta


class NuggetV3():
    # parameters
    direction = 0                # 0 - 'all', 1 - 'longs', 2 - 'shorts'
    initial_capital = 10000.0
    min_capital = 100.0
    commission = 0.075
    order_size_type = 0          # 0 - 'PERCENT', 1 - 'CURRENCY'
    order_size = 100.0
    leverage = 1

    stop_type = 2
    stop = 2.1
    trail_stop = 4
    trail_percent = 5.0
    take_volume_1 = np.array([10.0, 30.0, 10.0, 40.0, 10.0])
    take_volume_2 = np.array([10.0, 5.0, 15.0, 10.0, 20.0, 5.0, 20.0, 5.0, 5.0, 5.0])

    st_atr_period = 11
    st_factor = 14.65
    st_upper_band = 5.9
    st_lower_band = 2.4

    rsi_length = 10
    rsi_long_upper_bound = 46.0
    rsi_long_lower_bound = 24.0
    rsi_short_upper_bound = 72.0
    rsi_short_lower_bound = 51.0

    bb_filter = False
    ma_length = 22
    bb_mult = 1.8
    bb_long_bound = 38.0
    bb_short_bound = 62.0

    pivot_bars = 4
    look_back = 170
    channel_range = 14.0

    # changeable parameters and their possible values
    PARAMETER_VALUES = {
        'stop_type': [i for i in range(1, 4)],
        'stop': [i / 10 for i in range(1, 31)],
        'trail_stop': [i for i in range(0, 11)],
        'trail_percent': [float(i) for i in range(0, 55, 5)],
        'take_volume_1': [
            math.random_trading_volumes(5, 10, 10) for i in range(1000)
        ],
        'take_volume_2': [
            math.random_trading_volumes(10, 10, 5) for i in range(1000)
        ],
        'st_atr_period': [i for i in range(2, 21)],
        'st_factor': [i / 100 for i in range(1000, 2501, 5)],
        'st_upper_band': [i / 10 for i in range(46, 71)],
        'st_lower_band': [i / 10 for i in range(20, 37)],
        'rsi_length': [i for i in range(6, 22)],
        'rsi_long_upper_bound': [float(i) for i in range(29, 51)],
        'rsi_long_lower_bound': [float(i) for i in range(1, 29)],
        'rsi_short_upper_bound': [float(i) for i in range(69, 101)],
        'rsi_short_lower_bound': [float(i) for i in range(50, 69)],
        'bb_filter': [True, False],
        'ma_length': [i for i in range(5, 26)],
        'bb_mult': [i / 10 for i in range(15, 31)],
        'bb_long_bound': [float(i) for i in range(20, 51)],
        'bb_short_bound': [float(i) for i in range(50, 81)],
        'pivot_bars': [i for i in range(1, 31)],
        'look_back': [i for i in range(10, 201, 5)],
        'channel_range': [float(i) for i in range(5, 21)]
    }

    def __init__(self, exchange, parameters=None):
        if parameters is not None:
            self.stop_type = parameters[0]
            self.stop = parameters[1]
            self.trail_stop = parameters[2]
            self.trail_percent = parameters[3]
            self.take_volume_1 = parameters[4]
            self.take_volume_2 = parameters[5]
            self.st_atr_period = parameters[6]
            self.st_factor = parameters[7]
            self.st_upper_band = parameters[8]
            self.st_lower_band = parameters[9]
            self.rsi_length = parameters[10]
            self.rsi_long_upper_bound = parameters[11]
            self.rsi_long_lower_bound = parameters[12]
            self.rsi_short_upper_bound = parameters[13]
            self.rsi_short_lower_bound = parameters[14]
            self.bb_filter = parameters[15]
            self.ma_length = parameters[16]
            self.bb_mult = parameters[17]
            self.bb_long_bound = parameters[18]
            self.bb_short_bound = parameters[19]
            self.pivot_bars = parameters[20]
            self.look_back = parameters[21]
            self.channel_range = parameters[22]

        self.price_precision = exchange.price_precision
        self.qty_precision = exchange.qty_precision
        self.time = exchange.price_data[:, 0]
        self.high = exchange.price_data[:, 2]
        self.low = exchange.price_data[:, 3]
        self.close = exchange.price_data[:, 4]

        self.equity = self.initial_capital
        self.log = np.array([])
        self.deal_type = np.nan
        self.entry_signal = np.nan
        self.entry_date = np.nan
        self.entry_price = np.nan
        self.liquidation_price = np.nan
        self.stop_price = np.nan
        self.position_size = np.nan
        self.take_price_1 = np.full(5, np.nan)
        self.take_price_2 = np.full(10, np.nan)
        self.qty_take_1 = np.full(5, np.nan)
        self.qty_take_2 = np.full(10, np.nan)
        self.stop_moved = False
        self.grid_type = np.nan

        self.pivot_LH_bar_index = np.array([])
        self.pivot_HL_bar_index = np.array([])
        self.last_channel_range = np.nan
        self.last_pivot_LH = np.array([])
        self.last_pivot_HL = np.array([])
        self.last_pivot = np.nan
        self.pivot_HH = np.nan
        self.pivot_LL = np.nan

        self.supertrend = ta.supertrend(
            self.high, self.low, self.close, self.st_factor, self.st_atr_period)
        self.change_supertrend = ta.change(self.supertrend.values[0])
        self.rsi = ta.rsi(self.close, self.rsi_length)

        if self.bb_filter:
            self.bb_rsi = ta.bb(
                self.rsi.values, self.ma_length, self.bb_mult)
        else:
            self.bb_rsi = np.full(self.time.shape[0], np.nan)

        self.pivot_LH = ta.pivothigh(
            self.high, self.pivot_bars, self.pivot_bars)
        self.pivot_HL = ta.pivotlow(
            self.low, self.pivot_bars, self.pivot_bars)
        self.fibo_values = np.array(
            [
                0.0, 0.236, 0.382, 0.5, 0.618, 0.8, 1.0,
                1.618, 2.0, 2.618, 3.0, 3.618, 4.0
            ]
        )
        self.fibo_levels = np.full(13, np.nan)

        (
            self.equity,
            self.log,
            self.deal_type,
            self.entry_price,
            self.liquidation_price,
            self.stop_price,
            self.position_size,
            self.take_price_1,
            self.take_price_2,
            self.qty_take_1,
            self.qty_take_2,
            self.stop_moved,
            self.grid_type,
            self.pivot_LH_bar_index,
            self.pivot_HL_bar_index,
            self.last_channel_range,
            self.last_pivot_LH,
            self.last_pivot_HL,
            self.last_pivot,
            self.pivot_HH,
            self.pivot_LL
        ) = self.calculate(
                self.direction,
                self.min_capital,
                self.commission,
                self.order_size_type,
                self.order_size,
                self.leverage,
                self.stop_type,
                self.stop,
                self.trail_stop,
                self.trail_percent,
                self.take_volume_1,
                self.take_volume_2,
                self.st_upper_band,
                self.st_lower_band,
                self.rsi_long_upper_bound,
                self.rsi_long_lower_bound,
                self.rsi_short_upper_bound,
                self.rsi_short_lower_bound,
                self.bb_filter,
                self.bb_long_bound,
                self.bb_short_bound,
                self.pivot_bars,
                self.look_back,
                self.channel_range,
                self.price_precision,
                self.qty_precision,
                self.time,
                self.high,
                self.low,
                self.close,
                self.equity,
                self.log,
                self.deal_type,
                self.entry_signal,
                self.entry_date,
                self.entry_price,
                self.liquidation_price,
                self.stop_price,
                self.position_size,
                self.take_price_1,
                self.take_price_2,
                self.qty_take_1,
                self.qty_take_2,
                self.stop_moved,
                self.grid_type,
                self.pivot_LH_bar_index,
                self.pivot_HL_bar_index,
                self.last_channel_range,
                self.last_pivot_LH,
                self.last_pivot_HL,
                self.last_pivot,
                self.pivot_HH,
                self.pivot_LL,
                self.supertrend.values[0],
                self.supertrend.values[1],
                self.change_supertrend.values,
                self.rsi.values,
                self.bb_rsi.values[1] if self.bb_filter else self.bb_rsi,
                self.bb_rsi.values[2] if self.bb_filter else self.bb_rsi,
                self.pivot_LH.values,
                self.pivot_HL.values,
                self.fibo_values,
                self.fibo_levels
        )
        self.net_profit = round(self.equity - self.initial_capital, 2)

    @staticmethod
    @nb.jit(
        (
            nb.int8,
            nb.float64,
            nb.float64,
            nb.int8,
            nb.float64,
            nb.int8,
            nb.int8,
            nb.float64,
            nb.int8,
            nb.float64,
            nb.float64[:],
            nb.float64[:],
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.int8,
            nb.float64,
            nb.float64,
            nb.int16,
            nb.int16,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64,
            nb.float64[:],
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.int8,
            nb.float64,
            nb.float64[:],
            nb.float64[:],
            nb.float64,
            nb.float64[:],
            nb.float64[:],
            nb.float64,
            nb.float64,
            nb.float64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:]
        ),
        nopython=True,
        nogil=True,
        cache=True
    )
    def calculate(
        direction,
        min_capital,
        commission,
        order_size_type,
        order_size,
        leverage,
        stop_type,
        stop,
        trail_stop,
        trail_percent,
        take_volume_1,
        take_volume_2,
        st_upper_band,
        st_lower_band,
        rsi_long_upper_bound,
        rsi_long_lower_bound,
        rsi_short_upper_bound,
        rsi_short_lower_bound,
        bb_filter,
        bb_long_bound,
        bb_short_bound,
        pivot_bars,
        look_back,
        channel_range,
        price_precision,
        qty_precision,
        time,
        high,
        low,
        close,
        equity,
        log,
        deal_type,
        entry_signal,
        entry_date,
        entry_price,
        liquidation_price,
        stop_price,
        position_size,
        take_price_1,
        take_price_2,
        qty_take_1,
        qty_take_2,
        stop_moved,
        grid_type,
        pivot_LH_bar_index,
        pivot_HL_bar_index,
        last_channel_range,
        last_pivot_LH,
        last_pivot_HL,
        last_pivot,
        pivot_HH,
        pivot_LL,
        supertrend_indicator,
        supertrend_direction,
        change_supertrend,
        rsi,
        bb_rsi_upper,
        bb_rsi_lower,
        pivot_LH,
        pivot_HL,
        fibo_values,
        fibo_levels
    ):
        def round_to_minqty_or_mintick(number, precision):
            return round(int(number / precision) * precision, 8)

        def update_log(log, equity, commission, deal_type, entry_signal,
                       exit_signal, entry_date, exit_date, entry_price,
                       exit_price, position_size):
            total_commission = round(
                (position_size * entry_price
                    * commission / 100) + (position_size
                    * exit_price * commission / 100),
                2
            )

            if deal_type == 0:
                pnl = round(
                    (exit_price - entry_price) * position_size
                        - total_commission,
                    2
                )
            else:
                pnl = round(
                    (entry_price - exit_price) * position_size
                        - total_commission,
                    2
                )

            if position_size == 0:
                return log, equity

            pnl_per = round(
                (((position_size * entry_price) + pnl)
                    / (position_size * entry_price) - 1) * 100,
                2
            )
            log_row = np.array(
                [
                    deal_type, entry_signal, exit_signal, entry_date,
                    exit_date, entry_price, exit_price, position_size,
                    pnl, pnl_per, total_commission
                ]
            )
            log = np.concatenate((log, log_row))
            equity += pnl
            return log, equity
        
        for i in range(time.shape[0]):
            # indicators
            if pivot_LH_bar_index.shape[0] > 0:
                if i - pivot_LH_bar_index[0] > look_back:
                    pivot_LH_bar_index = pivot_LH_bar_index[1:]
                    last_pivot_LH = last_pivot_LH[1:]

                    if last_pivot_LH.shape[0] > 0:
                        pivot_HH = last_pivot_LH.max()
                    else:
                        pivot_HH = np.nan

            if pivot_HL_bar_index.shape[0] > 0:
                if i - pivot_HL_bar_index[0] > look_back:
                    pivot_HL_bar_index = pivot_HL_bar_index[1:]
                    last_pivot_HL = last_pivot_HL[1:]

                    if last_pivot_HL.shape[0] > 0:
                        pivot_LL = last_pivot_HL.min()
                    else:
                        pivot_LL = np.nan

            if not np.isnan(pivot_LH[i]):
                if last_pivot_LH.shape[0] > 0:
                    if (pivot_LH[i] >= last_pivot_LH.max() or
                            np.isnan(pivot_HH)):
                        pivot_HH = pivot_LH[i]
                        last_pivot = 0
                elif np.isnan(pivot_HH):
                    pivot_HH = pivot_LH[i]
                    last_pivot = 0

                pivot_LH_bar_index = np.concatenate(
                    (pivot_LH_bar_index, np.array([i - pivot_bars]))
                )
                last_pivot_LH = np.concatenate(
                    (last_pivot_LH, np.array([high[i - pivot_bars]]))
                )

            if not np.isnan(pivot_HL[i]):
                if last_pivot_HL.shape[0] > 0:
                    if (pivot_HL[i] <= last_pivot_HL.min() or 
                            np.isnan(pivot_LL)):
                        pivot_LL = pivot_HL[i]
                        last_pivot = 1
                elif np.isnan(pivot_LL):
                    pivot_LL = pivot_HL[i]
                    last_pivot = 1

                pivot_HL_bar_index = np.concatenate(
                    (pivot_HL_bar_index, np.array([i - pivot_bars]))
                )
                last_pivot_HL = np.concatenate(
                    (last_pivot_HL, np.array([low[i - pivot_bars]]))
                )

            # check of liquidation
            if (deal_type == 0 and low[i] <= liquidation_price):
                log, equity = update_log(
                    log,
                    equity,
                    commission,
                    deal_type,
                    entry_signal,
                    0,
                    entry_date,
                    time[i],
                    entry_price,
                    liquidation_price,
                    position_size
                )
                
                deal_type = np.nan
                entry_signal = np.nan
                entry_date = np.nan
                entry_price = np.nan
                liquidation_price = np.nan
                stop_price = np.nan
                position_size = np.nan
                take_price_1 = np.full(5, np.nan)
                take_price_2 = np.full(10, np.nan)
                qty_take_1 = np.full(5, np.nan)
                qty_take_2 = np.full(10, np.nan)
                stop_moved = False
                grid_type = np.nan

            if (deal_type == 1 and high[i] >= liquidation_price):
                log, equity = update_log(
                    log,
                    equity,
                    commission,
                    deal_type,
                    entry_signal,
                    0,
                    entry_date,
                    time[i],
                    entry_price,
                    liquidation_price,
                    position_size
                )

                deal_type = np.nan
                entry_signal = np.nan
                entry_date = np.nan
                entry_price = np.nan
                liquidation_price = np.nan
                stop_price = np.nan
                position_size = np.nan
                take_price_1 = np.full(5, np.nan)
                take_price_2 = np.full(10, np.nan)
                qty_take_1 = np.full(5, np.nan)
                qty_take_2 = np.full(10, np.nan)
                stop_moved = False
                grid_type = np.nan

            # trading logic (longs)
            if deal_type == 0:
                if low[i] <= stop_price:
                    log, equity = update_log(
                        log,
                        equity,
                        commission,
                        deal_type,
                        entry_signal,
                        1,
                        entry_date,
                        time[i],
                        entry_price,
                        stop_price,
                        position_size
                    )

                    deal_type = np.nan
                    entry_signal = np.nan
                    entry_date = np.nan
                    entry_price = np.nan
                    liquidation_price = np.nan
                    stop_price = np.nan
                    position_size = np.nan
                    take_price_1 = np.full(5, np.nan)
                    take_price_2 = np.full(10, np.nan)
                    qty_take_1 = np.full(5, np.nan)
                    qty_take_2 = np.full(10, np.nan)
                    stop_moved = False
                    grid_type = np.nan

                if (stop_type == 1 and
                        change_supertrend[i] and
                        supertrend_direction[i] < 0):
                    stop_price = round_to_minqty_or_mintick(
                        supertrend_indicator[i] * (100 - stop) / 100,
                        price_precision
                    )
                elif (stop_type == 2 or stop_type == 3) and not stop_moved:
                    take = None

                    if grid_type == 0:
                        if (trail_stop == 4 and
                                high[i] >= take_price_1[3]):
                            take = take_price_1[3]
                        elif (trail_stop == 3 and
                                high[i] >= take_price_1[2]):
                            take = take_price_1[2]
                        elif (trail_stop == 2 and
                                high[i] >= take_price_1[1]):
                            take = take_price_1[1]
                        elif (trail_stop == 1 and
                                high[i] >= take_price_1[0]):
                            take = take_price_1[0]
                    elif grid_type == 1:
                        if (trail_stop == 9 and
                                high[i] >= take_price_2[8]):
                            take = take_price_2[8]
                        elif (trail_stop == 8 and
                                high[i] >= take_price_2[7]):
                            take = take_price_2[7]
                        elif (trail_stop == 7 and
                                high[i] >= take_price_2[6]):
                            take = take_price_2[6]
                        elif (trail_stop == 6 and
                                high[i] >= take_price_2[5]):
                            take = take_price_2[5]
                        elif (trail_stop == 5 and
                                high[i] >= take_price_2[4]):
                            take = take_price_2[4]
                        elif (trail_stop == 4 and
                                high[i] >= take_price_2[3]):
                            take = take_price_2[3]
                        elif (trail_stop == 3 and
                                high[i] >= take_price_2[2]):
                            take = take_price_2[2]
                        elif (trail_stop == 2 and
                                high[i] >= take_price_2[1]):
                            take = take_price_2[1]
                        elif (trail_stop == 1 and
                                high[i] >= take_price_2[0]):
                            take = take_price_2[0]

                    if take is not None:
                        stop_moved = True
                        stop_price = round_to_minqty_or_mintick(
                            (take - entry_price) * (trail_percent / 100)
                                + entry_price,
                            price_precision
                        )

                if grid_type == 0:
                    if (not np.isnan(take_price_1[0]) and 
                            high[i] >= take_price_1[0]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            2,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[0],
                            qty_take_1[0]
                        )

                        take_price_1[0] = np.nan
                        position_size = round(position_size - qty_take_1[0], 8)
                        qty_take_1[0] = np.nan

                    if (not np.isnan(take_price_1[1]) and 
                            high[i] >= take_price_1[1]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            3,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[1],
                            qty_take_1[1]
                        )

                        take_price_1[1] = np.nan
                        position_size = round(position_size - qty_take_1[1], 8)
                        qty_take_1[1] = np.nan

                    if (not np.isnan(take_price_1[2]) and 
                            high[i] >= take_price_1[2]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            4,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[2],
                            qty_take_1[2]
                        ) 

                        take_price_1[2] = np.nan
                        position_size = round(position_size - qty_take_1[2], 8)
                        qty_take_1[2] = np.nan

                    if (not np.isnan(take_price_1[3]) and 
                            high[i] >= take_price_1[3]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            5,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[3],
                            qty_take_1[3]
                        ) 

                        take_price_1[3] = np.nan
                        position_size = round(position_size - qty_take_1[3], 8)
                        qty_take_1[3] = np.nan

                    if (not np.isnan(take_price_1[4]) and 
                            high[i] >= take_price_1[4]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            6,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[4],
                            round(qty_take_1[4], 8)
                        )

                        deal_type = np.nan
                        entry_signal = np.nan
                        entry_date = np.nan
                        entry_price = np.nan
                        position_size = np.nan
                        take_price_1[4] = np.nan
                        qty_take_1[4] = np.nan
                        stop_price = np.nan
                        stop_moved = False
                        grid_type = np.nan
                elif grid_type == 1:
                    if (not np.isnan(take_price_2[0]) and 
                            high[i] >= take_price_2[0]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            2,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[0],
                            qty_take_2[0]
                        )

                        take_price_2[0] = np.nan
                        position_size = round(position_size - qty_take_2[0], 8)
                        qty_take_2[0] = np.nan

                    if (not np.isnan(take_price_2[1]) and 
                            high[i] >= take_price_2[1]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            3,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[1],
                            qty_take_2[1]
                        )

                        take_price_2[1] = np.nan
                        position_size = round(position_size - qty_take_2[1], 8)
                        qty_take_2[1] = np.nan

                    if (not np.isnan(take_price_2[2]) and 
                            high[i] >= take_price_2[2]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            4,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[2],
                            qty_take_2[2]
                        ) 

                        take_price_2[2] = np.nan
                        position_size = round(position_size - qty_take_2[2], 8)
                        qty_take_2[2] = np.nan

                    if (not np.isnan(take_price_2[3]) and 
                            high[i] >= take_price_2[3]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            5,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[3],
                            qty_take_2[3]
                        ) 

                        take_price_2[3] = np.nan
                        position_size = round(position_size - qty_take_2[3], 8)
                        qty_take_2[3] = np.nan

                    if (not np.isnan(take_price_2[4]) and 
                            high[i] >= take_price_2[4]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            6,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[4],
                            qty_take_2[4]
                        ) 

                        take_price_2[4] = np.nan
                        position_size = round(position_size - qty_take_2[4], 8)
                        qty_take_2[4] = np.nan

                    if (not np.isnan(take_price_2[5]) and 
                            high[i] >= take_price_2[5]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            7,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[5],
                            qty_take_2[5]
                        ) 

                        take_price_2[5] = np.nan
                        position_size = round(position_size - qty_take_2[5], 8)
                        qty_take_2[5] = np.nan

                    if (not np.isnan(take_price_2[6]) and 
                            high[i] >= take_price_2[6]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            8,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[6],
                            qty_take_2[6]
                        ) 

                        take_price_2[6] = np.nan
                        position_size = round(position_size - qty_take_2[6], 8)
                        qty_take_2[6] = np.nan

                    if (not np.isnan(take_price_2[7]) and 
                            high[i] >= take_price_2[7]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            9,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[7],
                            qty_take_2[7]
                        ) 

                        take_price_2[7] = np.nan
                        position_size = round(position_size - qty_take_2[7], 8)
                        qty_take_2[7] = np.nan

                    if (not np.isnan(take_price_2[8]) and 
                            high[i] >= take_price_2[8]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            10,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[8],
                            qty_take_2[8]
                        ) 

                        take_price_2[8] = np.nan
                        position_size = round(position_size - qty_take_2[8], 8)
                        qty_take_2[8] = np.nan

                    if (not np.isnan(take_price_2[9]) and 
                            high[i] >= take_price_2[9]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            11,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[9],
                            round(qty_take_2[9], 8)
                        ) 

                        deal_type = np.nan
                        entry_signal = np.nan
                        entry_date = np.nan
                        entry_price = np.nan
                        position_size = np.nan
                        take_price_2[9] = np.nan
                        qty_take_2[9] = np.nan
                        stop_price = np.nan
                        stop_moved = False
                        grid_type = np.nan

            pre_entry_long = (
                supertrend_direction[i] < 0 and
                (close[i] / supertrend_indicator[i]
                    - 1) * 100 > st_lower_band and
                (close[i] / supertrend_indicator[i]
                    - 1) * 100 < st_upper_band and
                rsi[i] < rsi_long_upper_bound and
                rsi[i] > rsi_long_lower_bound and
                np.isnan(deal_type) and
                (bb_rsi_upper[i] < bb_long_bound
                    if bb_filter else True) and
                last_pivot == 1 and
                (equity > min_capital) and
                (direction == 0 or direction == 1)
            )

            if pre_entry_long:
                if not np.isnan(pivot_HH) and not np.isnan(pivot_LL):
                    last_channel_range = (pivot_HH / pivot_LL - 1) * 100

                    for j in range(fibo_values.shape[0]):
                        price = pivot_LL + (pivot_HH - pivot_LL) * fibo_values[j]
                        fibo_levels[j] = price

            entry_long = (
                pre_entry_long and 
                close[i] < fibo_levels[3] and
                close[i] >= fibo_levels[0]
            )

            if entry_long:
                deal_type = 0
                entry_signal = 0
                entry_date = time[i]
                entry_price = close[i]

                if order_size_type == 0:
                    initial_position =  (
                        equity * leverage * (order_size / 100.0)
                    )
                    position_size = (
                        initial_position * (1 - commission / 100)
                        / entry_price
                    )
                elif order_size_type == 1:
                    initial_position = (order_size * leverage)
                    position_size = (
                        initial_position * (1 - commission / 100)
                        / entry_price
                    )
                    
                position_size = round_to_minqty_or_mintick(
                    position_size, qty_precision
                )
                liquidation_price = round_to_minqty_or_mintick(
                    entry_price * (1 - (1 / leverage)), price_precision
                )

                if stop_type == 1 or stop_type == 2:
                    stop_price = round_to_minqty_or_mintick(
                        supertrend_indicator[i] * (100 - stop) / 100,
                        price_precision
                    )
                elif stop_type == 3:
                    stop_price = round_to_minqty_or_mintick(
                        fibo_levels[0] * (100 - stop) / 100, price_precision
                    )

                if last_channel_range >= channel_range:
                    grid_type = 0

                    if close[i] >= fibo_levels[0] and close[i] < fibo_levels[1]:
                        take_price_1[0] = round_to_minqty_or_mintick(
                            fibo_levels[2], price_precision
                        )
                        take_price_1[1] = round_to_minqty_or_mintick(
                            fibo_levels[3], price_precision
                        )
                        take_price_1[2] = round_to_minqty_or_mintick(
                            fibo_levels[4], price_precision
                        )
                        take_price_1[3] = round_to_minqty_or_mintick(
                            fibo_levels[5], price_precision
                        )
                        take_price_1[4] = round_to_minqty_or_mintick(
                            fibo_levels[6], price_precision
                        )
                    elif close[i] >= fibo_levels[1]:
                        take_price_1[0] = round_to_minqty_or_mintick(
                            fibo_levels[3], price_precision
                        )
                        take_price_1[1] = round_to_minqty_or_mintick(
                            fibo_levels[4], price_precision
                        )
                        take_price_1[2] = round_to_minqty_or_mintick(
                            fibo_levels[5], price_precision
                        )
                        take_price_1[3] = round_to_minqty_or_mintick(
                            fibo_levels[6], price_precision
                        )
                        take_price_1[4] = round_to_minqty_or_mintick(
                            fibo_levels[7], price_precision
                        )

                    qty_take_1[0] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[0] / 100, qty_precision
                    )
                    qty_take_1[1] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[1] / 100, qty_precision
                    )
                    qty_take_1[2] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[2] / 100, qty_precision
                    )
                    qty_take_1[3] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[3] / 100, qty_precision
                    )
                    qty_take_1[4] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[4] / 100, qty_precision
                    )
                elif last_channel_range < channel_range:
                    grid_type = 1

                    if close[i] >= fibo_levels[0] and close[i] < fibo_levels[2]:
                        take_price_2[0] = round_to_minqty_or_mintick(
                            fibo_levels[2], price_precision
                        )
                        take_price_2[1] = round_to_minqty_or_mintick(
                            fibo_levels[3], price_precision
                        )
                        take_price_2[2] = round_to_minqty_or_mintick(
                            fibo_levels[4], price_precision
                        )
                        take_price_2[3] = round_to_minqty_or_mintick(
                            fibo_levels[5], price_precision
                        )
                        take_price_2[4] = round_to_minqty_or_mintick(
                            fibo_levels[6], price_precision
                        )
                        take_price_2[5] = round_to_minqty_or_mintick(
                            fibo_levels[7], price_precision
                        )
                        take_price_2[6] = round_to_minqty_or_mintick(
                            fibo_levels[8], price_precision
                        )
                        take_price_2[7] = round_to_minqty_or_mintick(
                            fibo_levels[9], price_precision
                        )
                        take_price_2[8] = round_to_minqty_or_mintick(
                            fibo_levels[10], price_precision
                        )
                        take_price_2[9] = round_to_minqty_or_mintick(
                            fibo_levels[11], price_precision
                        )
                    elif close[i] >= fibo_levels[2]:
                        take_price_2[0] = round_to_minqty_or_mintick(
                            fibo_levels[3], price_precision
                        )
                        take_price_2[1] = round_to_minqty_or_mintick(
                            fibo_levels[4], price_precision
                        )
                        take_price_2[2] = round_to_minqty_or_mintick(
                            fibo_levels[5], price_precision
                        )
                        take_price_2[3] = round_to_minqty_or_mintick(
                            fibo_levels[6], price_precision
                        )
                        take_price_2[4] = round_to_minqty_or_mintick(
                            fibo_levels[7], price_precision
                        )
                        take_price_2[5] = round_to_minqty_or_mintick(
                            fibo_levels[8], price_precision
                        )
                        take_price_2[6] = round_to_minqty_or_mintick(
                            fibo_levels[9], price_precision
                        )
                        take_price_2[7] = round_to_minqty_or_mintick(
                            fibo_levels[10], price_precision
                        )
                        take_price_2[8] = round_to_minqty_or_mintick(
                            fibo_levels[11], price_precision
                        )
                        take_price_2[9] = round_to_minqty_or_mintick(
                            fibo_levels[12], price_precision
                        )

                    qty_take_2[0] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[0] / 100, qty_precision
                    )
                    qty_take_2[1] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[1] / 100, qty_precision
                    )
                    qty_take_2[2] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[2] / 100, qty_precision
                    )
                    qty_take_2[3] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[3] / 100, qty_precision
                    )
                    qty_take_2[4] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[4] / 100, qty_precision
                    )
                    qty_take_2[5] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[5] / 100, qty_precision
                    )
                    qty_take_2[6] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[6] / 100, qty_precision
                    )
                    qty_take_2[7] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[7] / 100, qty_precision
                    )
                    qty_take_2[8] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[8] / 100, qty_precision
                    )
                    qty_take_2[9] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[9] / 100, qty_precision
                    )

            # trading logic (shorts)
            if deal_type == 1:
                if high[i] >= stop_price:
                    log, equity = update_log(
                        log,
                        equity,
                        commission,
                        deal_type,
                        entry_signal,
                        1,
                        entry_date,
                        time[i],
                        entry_price,
                        stop_price,
                        position_size
                    )

                    deal_type = np.nan
                    entry_signal = np.nan
                    entry_date = np.nan
                    entry_price = np.nan
                    liquidation_price = np.nan
                    stop_price = np.nan
                    position_size = np.nan
                    take_price_1 = np.full(5, np.nan)
                    take_price_2 = np.full(10, np.nan)
                    qty_take_1 = np.full(5, np.nan)
                    qty_take_2 = np.full(10, np.nan)
                    stop_moved = False
                    grid_type = np.nan

                if (stop_type == 1 and
                        change_supertrend[i] and
                        supertrend_direction[i] > 0):
                    stop_price = round_to_minqty_or_mintick(
                        supertrend_indicator[i] * (100 + stop) / 100,
                        price_precision
                    )
                elif (stop_type == 2 or stop_type == 3) and not stop_moved:
                    take = None

                    if grid_type == 0:
                        if (trail_stop == 4 and
                                low[i] <= take_price_1[3]):
                            take = take_price_1[3]
                        elif (trail_stop == 3 and
                                low[i] <= take_price_1[2]):
                            take = take_price_1[2]
                        elif (trail_stop == 2 and
                                low[i] <= take_price_1[1]):
                            take = take_price_1[1]
                        elif (trail_stop == 1 and
                                low[i] <= take_price_1[0]):
                            take = take_price_1[0]
                    elif grid_type == 1:
                        if (trail_stop == 9 and
                                low[i] <= take_price_2[8]):
                            take = take_price_2[8]
                        elif (trail_stop == 8 and
                                low[i] <= take_price_2[7]):
                            take = take_price_2[7]
                        elif (trail_stop == 7 and
                                low[i] <= take_price_2[6]):
                            take = take_price_2[6]
                        elif (trail_stop == 6 and
                                low[i] <= take_price_2[5]):
                            take = take_price_2[5]
                        elif (trail_stop == 5 and
                                low[i] <= take_price_2[4]):
                            take = take_price_2[4]
                        elif (trail_stop == 4 and
                                low[i] <= take_price_2[3]):
                            take = take_price_2[3]
                        elif (trail_stop == 3 and
                                low[i] <= take_price_2[2]):
                            take = take_price_2[2]
                        elif (trail_stop == 2 and
                                low[i] <= take_price_2[1]):
                            take = take_price_2[1]
                        elif (trail_stop == 1 and
                                low[i] <= take_price_2[0]):
                            take = take_price_2[0]

                    if take is not None:
                        stop_moved = True
                        stop_price = round_to_minqty_or_mintick(
                            (take - entry_price) * (trail_percent / 100)
                                + entry_price,
                            price_precision
                        )
    
                if grid_type == 0:
                    if (not np.isnan(take_price_1[0]) and 
                            low[i] <= take_price_1[0]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            2,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[0],
                            qty_take_1[0]
                        )

                        take_price_1[0] = np.nan
                        position_size = round(position_size - qty_take_1[0], 8)
                        qty_take_1[0] = np.nan

                    if (not np.isnan(take_price_1[1]) and 
                            low[i] <= take_price_1[1]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            3,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[1],
                            qty_take_1[1]
                        )

                        take_price_1[1] = np.nan
                        position_size = round(position_size - qty_take_1[1], 8)
                        qty_take_1[1] = np.nan

                    if (not np.isnan(take_price_1[2]) and 
                            low[i] <= take_price_1[2]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            4,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[2],
                            qty_take_1[2]
                        ) 

                        take_price_1[2] = np.nan
                        position_size = round(position_size - qty_take_1[2], 8)
                        qty_take_1[2] = np.nan

                    if (not np.isnan(take_price_1[3]) and 
                            low[i] <= take_price_1[3]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            5,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[3],
                            qty_take_1[3]
                        ) 

                        take_price_1[3] = np.nan
                        position_size = round(position_size - qty_take_1[3], 8)
                        qty_take_1[3] = np.nan

                    if (not np.isnan(take_price_1[4]) and 
                            low[i] <= take_price_1[4]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            6,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_1[4],
                            round(qty_take_1[4], 8)
                        )

                        deal_type = np.nan
                        entry_signal = np.nan
                        entry_date = np.nan
                        entry_price = np.nan
                        position_size = np.nan
                        take_price_1[4] = np.nan
                        qty_take_1[4] = np.nan
                        stop_price = np.nan
                        stop_moved = False
                        grid_type = np.nan
                elif grid_type == 1:
                    if (not np.isnan(take_price_2[0]) and 
                            low[i] <= take_price_2[0]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            2,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[0],
                            qty_take_2[0]
                        )

                        take_price_2[0] = np.nan
                        position_size = round(position_size - qty_take_2[0], 8)
                        qty_take_2[0] = np.nan

                    if (not np.isnan(take_price_2[1]) and 
                            low[i] <= take_price_2[1]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            3,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[1],
                            qty_take_2[1]
                        )

                        take_price_2[1] = np.nan
                        position_size = round(position_size - qty_take_2[1], 8)
                        qty_take_2[1] = np.nan

                    if (not np.isnan(take_price_2[2]) and 
                            low[i] <= take_price_2[2]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            4,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[2],
                            qty_take_2[2]
                        ) 

                        take_price_2[2] = np.nan
                        position_size = round(position_size - qty_take_2[2], 8)
                        qty_take_2[2] = np.nan

                    if (not np.isnan(take_price_2[3]) and 
                            low[i] <= take_price_2[3]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            5,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[3],
                            qty_take_2[3]
                        ) 

                        take_price_2[3] = np.nan
                        position_size = round(position_size - qty_take_2[3], 8)
                        qty_take_2[3] = np.nan

                    if (not np.isnan(take_price_2[4]) and 
                            low[i] <= take_price_2[4]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            6,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[4],
                            qty_take_2[4]
                        ) 

                        take_price_2[4] = np.nan
                        position_size = round(position_size - qty_take_2[4], 8)
                        qty_take_2[4] = np.nan

                    if (not np.isnan(take_price_2[5]) and 
                            low[i] <= take_price_2[5]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            7,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[5],
                            qty_take_2[5]
                        ) 

                        take_price_2[5] = np.nan
                        position_size = round(position_size - qty_take_2[5], 8)
                        qty_take_2[5] = np.nan

                    if (not np.isnan(take_price_2[6]) and 
                            low[i] <= take_price_2[6]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            8,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[6],
                            qty_take_2[6]
                        ) 

                        take_price_2[6] = np.nan
                        position_size = round(position_size - qty_take_2[6], 8)
                        qty_take_2[6] = np.nan

                    if (not np.isnan(take_price_2[7]) and 
                            low[i] <= take_price_2[7]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            9,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[7],
                            qty_take_2[7]
                        ) 

                        take_price_2[7] = np.nan
                        position_size = round(position_size - qty_take_2[7], 8)
                        qty_take_2[7] = np.nan

                    if (not np.isnan(take_price_2[8]) and 
                            low[i] <= take_price_2[8]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            10,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[8],
                            qty_take_2[8]
                        ) 

                        take_price_2[8] = np.nan
                        position_size = round(position_size - qty_take_2[8], 8)
                        qty_take_2[8] = np.nan

                    if (not np.isnan(take_price_2[9]) and 
                            low[i] <= take_price_2[9]):
                        log, equity = update_log(
                            log,
                            equity,
                            commission,
                            deal_type,
                            entry_signal,
                            11,
                            entry_date,
                            time[i],
                            entry_price,
                            take_price_2[9],
                            round(qty_take_2[9], 8)
                        ) 

                        deal_type = np.nan
                        entry_signal = np.nan
                        entry_date = np.nan
                        entry_price = np.nan
                        position_size = np.nan
                        take_price_2[9] = np.nan
                        qty_take_2[9] = np.nan
                        stop_price = np.nan
                        stop_moved = False
                        grid_type = np.nan

            pre_entry_short = (
                supertrend_direction[i] > 0 and
                (supertrend_indicator[i] / close[i]
                    - 1) * 100 > st_lower_band and
                (supertrend_indicator[i] / close[i]
                    - 1) * 100 < st_upper_band and
                rsi[i] < rsi_short_upper_bound and
                rsi[i] > rsi_short_lower_bound and
                np.isnan(deal_type) and
                (bb_rsi_lower[i] > bb_short_bound
                    if bb_filter else True) and
                last_pivot == 0 and
                (equity > min_capital) and
                (direction == 0 or direction == 2)
            )

            if pre_entry_short:
                if not np.isnan(pivot_HH) and not np.isnan(pivot_LL):
                    last_channel_range = (pivot_HH / pivot_LL - 1) * 100

                    for j in range(fibo_values.shape[0]):
                        price = pivot_HH - (pivot_HH - pivot_LL) * fibo_values[j]
                        fibo_levels[j] = price

            entry_short = (
                pre_entry_short and
                close[i] > fibo_levels[3] and
                close[i] <= fibo_levels[0]
            )

            if entry_short:
                deal_type = 1
                entry_signal = 1
                entry_date = time[i]
                entry_price = close[i]

                if order_size_type == 0:
                    initial_position =  (
                        equity * leverage * (order_size / 100.0)
                    )
                    position_size = (
                        initial_position * (1 - commission / 100)
                        / entry_price
                    )
                elif order_size_type == 1:
                    initial_position = (order_size * leverage)
                    position_size = (
                        initial_position * (1 - commission / 100)
                        / entry_price
                    )
                    
                position_size = round_to_minqty_or_mintick(
                    position_size, qty_precision
                )
                liquidation_price = round_to_minqty_or_mintick(
                    entry_price * (1 + (1 / leverage)), price_precision
                )

                if stop_type == 1 or stop_type == 2:
                    stop_price = round_to_minqty_or_mintick(
                        supertrend_indicator[i] * (100 + stop) / 100,
                        price_precision
                    )
                elif stop_type == 3:
                    stop_price = round_to_minqty_or_mintick(
                        fibo_levels[0] * (100 + stop) / 100, price_precision
                    )

                if last_channel_range >= channel_range:
                    grid_type = 0

                    if close[i] <= fibo_levels[0] and close[i] > fibo_levels[1]:
                        take_price_1[0] = round_to_minqty_or_mintick(
                            fibo_levels[2], price_precision
                        )
                        take_price_1[1] = round_to_minqty_or_mintick(
                            fibo_levels[3], price_precision
                        )
                        take_price_1[2] = round_to_minqty_or_mintick(
                            fibo_levels[4], price_precision
                        )
                        take_price_1[3] = round_to_minqty_or_mintick(
                            fibo_levels[5], price_precision
                        )
                        take_price_1[4] = round_to_minqty_or_mintick(
                            fibo_levels[6], price_precision
                        )
                    elif close[i] <= fibo_levels[1]:
                        take_price_1[0] = round_to_minqty_or_mintick(
                            fibo_levels[3], price_precision
                        )
                        take_price_1[1] = round_to_minqty_or_mintick(
                            fibo_levels[4], price_precision
                        )
                        take_price_1[2] = round_to_minqty_or_mintick(
                            fibo_levels[5], price_precision
                        )
                        take_price_1[3] = round_to_minqty_or_mintick(
                            fibo_levels[6], price_precision
                        )
                        take_price_1[4] = round_to_minqty_or_mintick(
                            fibo_levels[7], price_precision
                        )

                    qty_take_1[0] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[0] / 100, qty_precision
                    )
                    qty_take_1[1] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[1] / 100, qty_precision
                    )
                    qty_take_1[2] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[2] / 100, qty_precision
                    )
                    qty_take_1[3] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[3] / 100, qty_precision
                    )
                    qty_take_1[4] = round_to_minqty_or_mintick(
                        position_size * take_volume_1[4] / 100, qty_precision
                    )
                elif last_channel_range < channel_range:
                    grid_type = 1

                    if close[i] <= fibo_levels[0] and close[i] > fibo_levels[2]:
                        take_price_2[0] = round_to_minqty_or_mintick(
                            fibo_levels[2], price_precision
                        )
                        take_price_2[1] = round_to_minqty_or_mintick(
                            fibo_levels[3], price_precision
                        )
                        take_price_2[2] = round_to_minqty_or_mintick(
                            fibo_levels[4], price_precision
                        )
                        take_price_2[3] = round_to_minqty_or_mintick(
                            fibo_levels[5], price_precision
                        )
                        take_price_2[4] = round_to_minqty_or_mintick(
                            fibo_levels[6], price_precision
                        )
                        take_price_2[5] = round_to_minqty_or_mintick(
                            fibo_levels[7], price_precision
                        )
                        take_price_2[6] = round_to_minqty_or_mintick(
                            fibo_levels[8], price_precision
                        )
                        take_price_2[7] = round_to_minqty_or_mintick(
                            fibo_levels[9], price_precision
                        )
                        take_price_2[8] = round_to_minqty_or_mintick(
                            fibo_levels[10], price_precision
                        )
                        take_price_2[9] = round_to_minqty_or_mintick(
                            fibo_levels[11], price_precision
                        )
                    elif close[i] <= fibo_levels[2]:
                        take_price_2[0] = round_to_minqty_or_mintick(
                            fibo_levels[3], price_precision
                        )
                        take_price_2[1] = round_to_minqty_or_mintick(
                            fibo_levels[4], price_precision
                        )
                        take_price_2[2] = round_to_minqty_or_mintick(
                            fibo_levels[5], price_precision
                        )
                        take_price_2[3] = round_to_minqty_or_mintick(
                            fibo_levels[6], price_precision
                        )
                        take_price_2[4] = round_to_minqty_or_mintick(
                            fibo_levels[7], price_precision
                        )
                        take_price_2[5] = round_to_minqty_or_mintick(
                            fibo_levels[8], price_precision
                        )
                        take_price_2[6] = round_to_minqty_or_mintick(
                            fibo_levels[9], price_precision
                        )
                        take_price_2[7] = round_to_minqty_or_mintick(
                            fibo_levels[10], price_precision
                        )
                        take_price_2[8] = round_to_minqty_or_mintick(
                            fibo_levels[11], price_precision
                        )
                        take_price_2[9] = round_to_minqty_or_mintick(
                            fibo_levels[12], price_precision
                        )

                    qty_take_2[0] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[0] / 100, qty_precision
                    )
                    qty_take_2[1] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[1] / 100, qty_precision
                    )
                    qty_take_2[2] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[2] / 100, qty_precision
                    )
                    qty_take_2[3] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[3] / 100, qty_precision
                    )
                    qty_take_2[4] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[4] / 100, qty_precision
                    )
                    qty_take_2[5] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[5] / 100, qty_precision
                    )
                    qty_take_2[6] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[6] / 100, qty_precision
                    )
                    qty_take_2[7] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[7] / 100, qty_precision
                    )
                    qty_take_2[8] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[8] / 100, qty_precision
                    )
                    qty_take_2[9] = round_to_minqty_or_mintick(
                        position_size * take_volume_2[9] / 100, qty_precision
                    )

        return (
            equity,
            log,
            deal_type,
            entry_price,
            liquidation_price,
            stop_price,
            position_size,
            take_price_1,
            take_price_2,
            qty_take_1,
            qty_take_2,
            stop_moved,
            grid_type,
            pivot_LH_bar_index,
            pivot_HL_bar_index,
            last_channel_range,
            last_pivot_LH,
            last_pivot_HL,
            last_pivot,
            pivot_HH,
            pivot_LL
        )