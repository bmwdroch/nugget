import numpy as np
import numba as nb


class adx:
    def __init__(self, high, low, close, di_length, adx_length):
        (
            self.change_high, 
            self.change_low,
            self.tr,
            self.rma_tr,
            self.rma_plus_dmi,
            self.rma_minus_dmi,
            self.rma,
            self.values
        ) = self.calculate(high, low, close, di_length, adx_length)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:], nb.float64[:], nb.int16, nb.int16),
        nopython=True, nogil=True, cache=True
    )
    def calculate(high, low, close, di_length, adx_length):
        # change_high
        change_high = (
            high - np.concatenate((np.full(1, np.nan), high[: -1]))
        )

        # change_low
        change_low = (
            low - np.concatenate((np.full(1, np.nan), low[: -1]))
        )

        # tr
        hl = high - low
        hc = np.absolute(
            high - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        lc = np.absolute(
            low - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        hl[0] = np.nan
        hc[0] = np.nan
        lc[0] = np.nan
        tr = np.maximum(np.maximum(hl, hc), lc)

        # rma_tr
        rma_tr = tr.copy()
        alpha = 1 / di_length
        na_sum = np.isnan(rma_tr).sum()
        rma_tr[: di_length + na_sum - 1] = np.nan
        rma_tr[di_length + na_sum - 1] = (
            tr[na_sum : di_length + na_sum].mean()
        )

        for i in range(di_length + na_sum, rma_tr.shape[0]):
            rma_tr[i] = alpha * rma_tr[i] + (1 - alpha) * rma_tr[i - 1]

        # rma_plus_dmi
        plus_dmi = change_high.copy()
        plus_dmi[~((change_high > -change_low) & (change_high > 0))] = 0
        plus_dmi[0] = np.nan

        rma_plus_dmi = plus_dmi.copy()
        alpha = 1 / di_length
        na_sum = np.isnan(rma_plus_dmi).sum()
        rma_plus_dmi[: di_length + na_sum - 1] = np.nan
        rma_plus_dmi[di_length + na_sum - 1] = (
            plus_dmi[na_sum : di_length + na_sum].mean()
        )

        for i in range(di_length + na_sum, plus_dmi.shape[0]):
            rma_plus_dmi[i] = (alpha * rma_plus_dmi[i] +
                (1 - alpha) * rma_plus_dmi[i - 1])

        # rma_minus_dmi
        minus_dmi = -change_low.copy()
        minus_dmi[~((-change_low > change_high) & (-change_low > 0))] = 0
        minus_dmi[0] = np.nan

        rma_minus_dmi = minus_dmi.copy()
        alpha = 1 / di_length
        na_sum = np.isnan(rma_minus_dmi).sum()
        rma_minus_dmi[: di_length + na_sum - 1] = np.nan
        rma_minus_dmi[di_length + na_sum - 1] = (
            minus_dmi[na_sum : di_length + na_sum].mean()
        )

        for i in range(di_length + na_sum, minus_dmi.shape[0]):
            rma_minus_dmi[i] = (alpha * rma_minus_dmi[i] +
                (1 - alpha) * rma_minus_dmi[i - 1])

        # dmi
        plus = 100 * rma_plus_dmi / rma_tr
        minus = 100 * rma_minus_dmi / rma_tr
        dmi = (plus, minus)
        
        # rma
        diff = dmi[0] - dmi[1]
        sum = dmi[0] + dmi[1]
        sum[sum == 0] = 1
        source = np.absolute(diff) / sum


        rma = source.copy()
        alpha = 1 / adx_length
        na_sum = np.isnan(rma).sum()
        rma[: adx_length + na_sum - 1] = np.nan
        rma[adx_length + na_sum - 1] = (
            source[na_sum : adx_length + na_sum].mean()
        )

        for i in range(adx_length + na_sum, source.shape[0]):
            rma[i] = (alpha * rma[i] +
                (1 - alpha) * rma[i - 1])

        # values
        values = 100 * rma
        return (change_high, 
                change_low, 
                tr, 
                rma_tr, 
                rma_plus_dmi, 
                rma_minus_dmi,
                rma, 
                values)


class atr:
    def __init__(self, high, low, close, length):
        self.tr, self.values = self.calculate(high, low, close, length)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:], nb.float64[:], nb.int16),
        nopython=True, nogil=True, cache=True
    )
    def calculate(high, low, close, length):
        # tr
        hl = high - low
        hc = np.absolute(
            high - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        lc = np.absolute(
            low - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        hc[0] = abs(high[0] - close[0])
        lc[0] = abs(low[0] - close[0])
        tr = np.maximum(np.maximum(hl, hc), lc)

        # values
        alpha = 1 / length
        values = tr.copy()
        na_sum = np.isnan(values).sum()
        values[length + na_sum - 1] = tr[na_sum : length + na_sum].mean()
        values[: length + na_sum - 1] = np.nan

        for i in range(length + na_sum, values.shape[0]):
            values[i] = alpha * values[i] + (1 - alpha) * values[i - 1]

        return tr, values


class bb:
    def __init__(self, source, length, mult):
        self.stdev, self.values = self.calculate(source, length, mult)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.int16, nb.float32), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(source, length, mult):
        # sma
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        sma = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            sma[i] = rolling[i].mean()

        sma = np.concatenate((np.full(length - 1, np.nan), sma))

        # stdev
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        stdev = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            stdev[i] = rolling[i].std()

        stdev = np.concatenate((np.full(length - 1, np.nan), stdev))

        # values
        upper = sma + mult * stdev
        lower = sma - mult * stdev
        values = (sma, upper, lower)
        return stdev, values


class bbw:
    def __init__(self, source, length, mult):
        (
            self.sma, 
            self.stdev, 
            self.values
        ) = self.calculate(source, length, mult)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.int16, nb.float32), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(source, length, mult):
        # sma
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        sma = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            sma[i] = rolling[i].mean()

        sma = np.concatenate((np.full(length - 1, np.nan), sma))

        # stdev
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        stdev = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            stdev[i] = rolling[i].std()

        stdev = np.concatenate((np.full(length - 1, np.nan), stdev))

        # values
        values = (((sma + mult * stdev) - (sma - mult * stdev)) / sma)
        return sma, stdev, values


class change:
    def __init__(self, source, length=1):
        self.values = self.calculate(source, length)

    @staticmethod
    @nb.jit((nb.float64[:], nb.int16), nopython=True, nogil=True, cache=True)
    def calculate(source, length=1):
        values = (
            source - np.concatenate(
                (np.full(length, np.nan), source[: -length])
            )
        )
        return values


class cross:
    def __init__(self, source1, source2):
        self.values = self.calculate(source1, source2)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:]), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(source1, source2):
        data1 = source1 - source2
        data2 = np.concatenate((np.full(1, np.nan), data1[: -1]))
        values = ((data1 > 0) & (data2 <= 0)) | ((data1 < 0) & (data2 >= 0))
        return values


class crossover:
    def __init__(self, source1, source2):
        self.values = self.calculate(source1, source2)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:]), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(source1, source2):
        data1 = source1 - source2
        data2 = np.concatenate((np.full(1, np.nan), data1[: -1]))
        values = (data1 > 0) & (data2 <= 0)
        return values


class crossunder:
    def __init__(self, source1, source2):
        self.values = self.calculate(source1, source2)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:]), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(source1, source2):
        data1 = source1 - source2
        data2 = np.concatenate((np.full(1, np.nan), data1[: -1]))
        values = (data1 < 0) & (data2 >= 0)
        return values


class cum:
    def __init__(self, source):
        self.values = self.calculate(source)

    @staticmethod
    @nb.jit((nb.float64[:],), nopython=True, nogil=True, cache=True)
    def calculate(source):
        values = source.cumsum()
        return values


class dd:
    def __init__(self, source1, source2, length):
        self.donchian, self.values = self.calculate(source1, source2, length)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:], nb.int16), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(source1, source2, length):
        # upper
        source1 = source1.copy()
        source1[np.isnan(source1)] = -np.inf
        rolling = np.lib.stride_tricks.sliding_window_view(source1, length)
        upper = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            upper[i] = rolling[i].max()

        upper = np.concatenate((np.full(length - 1, np.nan), upper))
        upper[upper == -np.inf] = np.nan

        # lower
        source2 = source2.copy()
        source2[np.isnan(source2)] = np.inf
        rolling = np.lib.stride_tricks.sliding_window_view(source2, length)
        lower = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            lower[i] = rolling[i].min()

        lower = np.concatenate((np.full(length - 1, np.nan), lower))
        lower[lower == np.inf] = np.nan 

        # middle
        middle = (upper + lower) / 2

        # donchian
        donchian = (upper, lower, middle)

        # values
        for i in range(1, middle.shape[0]):
            if not (source1[i] >= upper[i - 1] or
                    source2[i] <= lower[i - 1]):
                if not np.isnan(upper[i - 1]):
                    upper[i] = upper[i - 1]
            
                if not np.isnan(lower[i - 1]):
                    lower[i] = lower[i - 1]

                if not np.isnan(middle[i - 1]):
                    middle[i] = middle[i - 1]

        values = (upper, lower, middle)
        return donchian, values


class dmi:
    def __init__(self, high, low, close, di_length):
        (
            self.change_high, 
            self.change_low,
            self.tr,
            self.rma_tr,
            self.rma_plus_dmi,
            self.rma_minus_dmi,
            self.values
        ) = self.calculate(high, low, close, di_length)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:], nb.float64[:], nb.int16), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(high, low, close, di_length):
        # change_high
        change_high = (
            high - np.concatenate((np.full(1, np.nan), high[: -1]))
        )

        # change_low
        change_low = (
            low - np.concatenate((np.full(1, np.nan), low[: -1]))
        )

        # tr
        hl = high - low
        hc = np.absolute(
            high - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        lc = np.absolute(
            low - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        hl[0] = np.nan
        hc[0] = np.nan
        lc[0] = np.nan
        tr = np.maximum(np.maximum(hl, hc), lc)

        # rma_tr
        rma_tr = tr.copy()
        alpha = 1 / di_length
        na_sum = np.isnan(rma_tr).sum()
        rma_tr[: di_length + na_sum - 1] = np.nan
        rma_tr[di_length + na_sum - 1] = (
            tr[na_sum : di_length + na_sum].mean()
        )

        for i in range(di_length + na_sum, rma_tr.shape[0]):
            rma_tr[i] = alpha * rma_tr[i] + (1 - alpha) * rma_tr[i - 1]

        # rma_plus_dmi
        plus_dmi = change_high.copy()
        plus_dmi[~((change_high > -change_low) & (change_high > 0))] = 0
        plus_dmi[0] = np.nan

        rma_plus_dmi = plus_dmi.copy()
        alpha = 1 / di_length
        na_sum = np.isnan(rma_plus_dmi).sum()
        rma_plus_dmi[: di_length + na_sum - 1] = np.nan
        rma_plus_dmi[di_length + na_sum - 1] = (
            plus_dmi[na_sum : di_length + na_sum].mean()
        )

        for i in range(di_length + na_sum, plus_dmi.shape[0]):
            rma_plus_dmi[i] = (alpha * rma_plus_dmi[i] +
                (1 - alpha) * rma_plus_dmi[i - 1])

        # rma_minus_dmi
        minus_dmi = -change_low.copy()
        minus_dmi[~((-change_low > change_high) & (-change_low > 0))] = 0
        minus_dmi[0] = np.nan

        rma_minus_dmi = minus_dmi.copy()
        alpha = 1 / di_length
        na_sum = np.isnan(rma_minus_dmi).sum()
        rma_minus_dmi[: di_length + na_sum - 1] = np.nan
        rma_minus_dmi[di_length + na_sum - 1] = (
            minus_dmi[na_sum : di_length + na_sum].mean()
        )

        for i in range(di_length + na_sum, minus_dmi.shape[0]):
            rma_minus_dmi[i] = (alpha * rma_minus_dmi[i] +
                (1 - alpha) * rma_minus_dmi[i - 1])

        # values
        plus = 100 * rma_plus_dmi / rma_tr
        minus = 100 * rma_minus_dmi / rma_tr
        values = (plus, minus)
        return (change_high, 
                change_low, 
                tr, 
                rma_tr, 
                rma_plus_dmi, 
                rma_minus_dmi, 
                values)


class donchian:
    def __init__(self, source1, source2, length):
        self.values = self.calculate(source1, source2, length)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:], nb.int16), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(source1, source2, length):
        # upper
        source1 = source1.copy()
        source1[np.isnan(source1)] = -np.inf
        rolling = np.lib.stride_tricks.sliding_window_view(source1, length)
        upper = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            upper[i] = rolling[i].max()

        upper = np.concatenate((np.full(length - 1, np.nan), upper))
        upper[upper == -np.inf] = np.nan

        # lower
        source2 = source2.copy()
        source2[np.isnan(source2)] = np.inf
        rolling = np.lib.stride_tricks.sliding_window_view(source2, length)
        lower = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            lower[i] = rolling[i].min()

        lower = np.concatenate((np.full(length - 1, np.nan), lower))
        lower[lower == np.inf] = np.nan 

        # middle
        middle = (upper + lower) / 2

        # values
        values = (upper, lower, middle)
        return values


class ds:
    def __init__(self, high, low, close, factor, atr_length):
        (
            self.tr,
            self.atr,
            self.values
        ) = self.calculate(high, low, close, factor, atr_length)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:], nb.float64[:], nb.float32, nb.int16),
        nopython=True, nogil=True, cache=True
    )
    def calculate(high, low, close, factor, atr_length):
        # tr
        hl = high - low
        hc = np.absolute(
            high - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        lc = np.absolute(
            low - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        hc[0] = abs(high[0] - close[0])
        lc[0] = abs(low[0] - close[0])
        tr = np.maximum(np.maximum(hl, hc), lc)

        # atr
        atr = tr.copy()
        alpha = 1 / atr_length
        na_sum = np.isnan(atr).sum()
        atr[: atr_length + na_sum - 1] = np.nan
        atr[atr_length + na_sum - 1] = tr[na_sum : atr_length + na_sum].mean()

        for i in range(atr_length + na_sum, atr.shape[0]):
            atr[i] = alpha * atr[i] + (1 - alpha) * atr[i - 1]

        # values
        hl2 = (high + low) / 2
        upper_band = hl2 + factor * atr
        lower_band = hl2 - factor * atr

        for i in range(1, atr.shape[0]):
            if not np.isnan(upper_band[i - 1]):
                prev_upper_band = upper_band[i - 1]
            else:
                prev_upper_band = 0

            if not np.isnan(lower_band[i - 1]):
                prev_lower_band = lower_band[i - 1]
            else:
                prev_lower_band = 0

            if not (upper_band[i] < prev_upper_band
                    or close[i - 1] > prev_upper_band):
                upper_band[i] = prev_upper_band

            if not (lower_band[i] > prev_lower_band
                    or close[i - 1] < prev_lower_band):
                lower_band[i] = prev_lower_band

        upper_band[upper_band == 0] = np.nan
        lower_band[lower_band == 0] = np.nan
        values = (upper_band, lower_band)
        return tr, atr, values


class ema:
    def __init__(self, source, length):
        self.values = self.calculate(source, length)

    @staticmethod
    @nb.jit((nb.float64[:], nb.int16), nopython=True, nogil=True, cache=True)
    def calculate(source, length):
        values = source.copy()
        alpha = 2 / (length + 1)
        na_sum = np.isnan(values).sum()
        values[: length + na_sum - 1] = np.nan
        values[length + na_sum - 1] = source[na_sum : length + na_sum].mean()

        for i in range(length + na_sum, values.shape[0]):
            values[i] = alpha * values[i] + (1 - alpha) * values[i - 1]

        return values


class highest:
    def __init__(self, source, length):
        self.values = self.calculate(source, length)

    @staticmethod
    @nb.jit((nb.float64[:], nb.int16), nopython=True, nogil=True, cache=True)
    def calculate(source, length):
        source = source.copy()
        source[np.isnan(source)] = -np.inf
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        values = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            values[i] = rolling[i].max()

        values = np.concatenate((np.full(length - 1, np.nan), values))
        values[values == -np.inf] = np.nan
        return values


class lowest:
    def __init__(self, source, length):
        self.values = self.calculate(source, length)

    @staticmethod
    @nb.jit((nb.float64[:], nb.int16), nopython=True, nogil=True, cache=True)
    def calculate(source, length):
        source = source.copy()
        source[np.isnan(source)] = np.inf
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        values = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            values[i] = rolling[i].min()

        values = np.concatenate((np.full(length - 1, np.nan), values))
        values[values == np.inf] = np.nan
        return values


class pivothigh:
    def __init__(self, source, leftbars, rightbars):
        self.values = self.calculate(source, leftbars, rightbars)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.int16, nb.int16), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(source, leftbars, rightbars):
        source = source.copy()
        source[np.isnan(source)] = -np.inf
        length = leftbars + rightbars + 1
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        values = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            values[i] = rolling[i].max()

        values = np.concatenate((np.full(length - 1, np.nan), values))
        values[values == -np.inf] = np.nan
        values[
            values != np.concatenate(
                (np.full(rightbars, np.nan), source[: -rightbars])
            )
        ] = np.nan
        return values


class pivotlow:
    def __init__(self, source, leftbars, rightbars):
        self.values = self.calculate(source, leftbars, rightbars)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.int16, nb.int16), 
        nopython=True, nogil=True, cache=True
    )
    def calculate(source, leftbars, rightbars):
        source = source.copy()
        source[np.isnan(source)] = np.inf
        length = leftbars + rightbars + 1
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        values = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            values[i] = rolling[i].min()

        values = np.concatenate((np.full(length - 1, np.nan), values))
        values[values == np.inf] = np.nan
        values[
            values != np.concatenate(
                (np.full(rightbars, np.nan), source[: -rightbars])
            )
        ] = np.nan
        return values


class rma:
    def __init__(self, source, length):
        self.values = self.calculate(source, length)

    @staticmethod
    @nb.jit((nb.float64[:], nb.int16), nopython=True, nogil=True, cache=True)
    def calculate(source, length):
        values = source.copy()
        alpha = 1 / length
        na_sum = np.isnan(values).sum()
        values[: length + na_sum - 1] = np.nan
        values[length + na_sum - 1] = source[na_sum : length + na_sum].mean()

        for i in range(length + na_sum, values.shape[0]):
            values[i] = alpha * values[i] + (1 - alpha) * values[i - 1]

        return values


class rsi:
    def __init__(self, source, length):
        self.rma_u, self.rma_d, self.values = self.calculate(source, length)

    @staticmethod
    @nb.jit((nb.float64[:], nb.int16), nopython=True, nogil=True, cache=True)
    def calculate(source, length):
        u = source - np.concatenate((np.full(1, np.nan), source[: -1]))
        d = np.concatenate((np.full(1, np.nan), source[: -1])) - source
        u[u < 0] = 0
        d[d < 0] = 0

        # rma_u
        rma_u = u.copy()
        alpha = 1 / length
        na_sum = np.isnan(rma_u).sum()
        rma_u[: length + na_sum - 1] = np.nan
        rma_u[length + na_sum - 1] = u[na_sum : length + na_sum].mean()

        for i in range(length + na_sum, rma_u.shape[0]):
            rma_u[i] = alpha * rma_u[i] + (1 - alpha) * rma_u[i - 1]

        # rma_d
        rma_d = d.copy()
        alpha = 1 / length
        na_sum = np.isnan(rma_d).sum()
        rma_d[: length + na_sum - 1] = np.nan
        rma_d[length + na_sum - 1] = d[na_sum : length + na_sum].mean()

        for i in range(length + na_sum, rma_d.shape[0]):
            rma_d[i] = alpha * rma_d[i] + (1 - alpha) * rma_d[i - 1]

        # values
        rs = rma_u / rma_d
        values = 100 - 100 / (1 + rs)
        return rma_u, rma_d, values


class sma:
    def __init__(self, source, length):
        self.values = self.calculate(source, length)

    @staticmethod
    @nb.jit((nb.float64[:], nb.int16), nopython=True, nogil=True, cache=True)
    def calculate(source, length):
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        values = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            values[i] = rolling[i].mean()

        values = np.concatenate((np.full(length - 1, np.nan), values))
        return values


class stdev:
    def __init__(self, source, length):
        self.values = self.calculate(source, length)

    @staticmethod
    @nb.jit((nb.float64[:], nb.int16), nopython=True, nogil=True, cache=True)
    def calculate(source, length):
        rolling = np.lib.stride_tricks.sliding_window_view(source, length)
        values = np.full(rolling.shape[0], np.nan)

        for i in range(rolling.shape[0]):
            values[i] = rolling[i].std()

        values = np.concatenate((np.full(length - 1, np.nan), values))
        return values


class supertrend:
    def __init__(self, high, low, close, factor, atr_length):
        (
            self.tr,
            self.atr,
            self.upper_band,
            self.lower_band,
            self.values
        ) = self.calculate(high, low, close, factor, atr_length)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:], nb.float64[:], nb.float32, nb.int16),
        nopython=True, nogil=True, cache=True
    )
    def calculate(high, low, close, factor, atr_length):
        # tr
        hl = high - low
        hc = np.absolute(
            high - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        lc = np.absolute(
            low - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        hc[0] = abs(high[0] - close[0])
        lc[0] = abs(low[0] - close[0])
        tr = np.maximum(np.maximum(hl, hc), lc)

        # atr
        atr = tr.copy()
        alpha = 1 / atr_length
        na_sum = np.isnan(atr).sum()
        atr[: atr_length + na_sum - 1] = np.nan
        atr[atr_length + na_sum - 1] = tr[na_sum : atr_length + na_sum].mean()

        for i in range(atr_length + na_sum, atr.shape[0]):
            atr[i] = alpha * atr[i] + (1 - alpha) * atr[i - 1]

        # values
        hl2 = (high + low) / 2
        upper_band = hl2 + factor * atr
        lower_band = hl2 - factor * atr
        indicator = np.full(atr.shape[0], np.nan)
        direction = np.full(atr.shape[0], np.nan)

        for i in range(1, atr.shape[0]):
            prev_indicator = indicator[i - 1]

            if not np.isnan(upper_band[i - 1]):
                prev_upper_band = upper_band[i - 1]
            else:
                prev_upper_band = 0

            if not np.isnan(lower_band[i - 1]):
                prev_lower_band = lower_band[i - 1]
            else:
                prev_lower_band = 0

            if not (upper_band[i] < prev_upper_band
                    or close[i - 1] > prev_upper_band):
                upper_band[i] = prev_upper_band

            if not (lower_band[i] > prev_lower_band
                    or close[i - 1] < prev_lower_band):
                lower_band[i] = prev_lower_band

            if np.isnan(atr[i - 1]):
                direction[i] = 1
            elif prev_indicator == prev_upper_band:
                if close[i] > upper_band[i]:
                    direction[i] = -1
                else:
                    direction[i] = 1
            else:
                if close[i] < lower_band[i]:
                    direction[i] = 1
                else:
                    direction[i] = -1

            if direction[i] == -1:
                indicator[i] = lower_band[i]
            else:
                indicator[i] = upper_band[i]

        values = (indicator, direction)
        return tr, atr, upper_band, lower_band, values


class tr:
    def __init__(self, high, low, close, handle_na):
        self.values = self.calculate(high, low, close, handle_na)

    @staticmethod
    @nb.jit(
        (nb.float64[:], nb.float64[:], nb.float64[:], nb.int8),
        nopython=True, nogil=True, cache=True
    )
    def calculate(high, low, close, handle_na):
        hl = high - low
        hc = np.absolute(
            high - np.concatenate((np.full(1, np.nan), close[:-1]))
        )
        lc = np.absolute(
            low - np.concatenate((np.full(1, np.nan), close[:-1]))
        )

        if handle_na:
            hc[0] = abs(high[0] - close[0])
            lc[0] = abs(low[0] - close[0])
        else:
            hl[0] = np.nan
            hc[0] = np.nan
            lc[0] = np.nan

        values = np.maximum(np.maximum(hl, hc), lc)
        return values