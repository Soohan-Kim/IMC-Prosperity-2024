from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import json
import pandas as pd
import numpy as np
import statistics
import math
from collections import deque


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()
WINDOW = 200
rc_WINDOW = 100
sb_lag1, sb_lag2 = 50, 100
ccnt_lag1, ccnt_lag2 = 50, 100

class Trader:
    position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0,
                'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350,
                      'ROSES': 60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}

    reg_params = [
        0.006305986347540827,
        -0.01214216608632497,
        0.009966507857309953,
        -0.011357179548720021,
        0.0144586905386713,
        0.022841418289938183,
        0.05102468136008776,
        0.19978102347281332,
        0.7189835095609063
    ]
    reg_intercept = 0.6951698609096012
    starfruit_midprice = deque(maxlen=len(reg_params))

    conversions = 0
    orchids_buy_prc, orchids_sell_prc = 0, 0
    orchids_sell_qty = 0

    orchids_price = []

    orchids_weighted_sell_price = 0
    orchids_total_volume = 0

    humid_orchids_weighted_sell_price = 0
    humid_orchids_total_volume = 0

    timestamp = 0

    spreads = deque(maxlen=WINDOW)
    rc_spreads = deque(maxlen=rc_WINDOW)
    c_prices = np.array([])
    s_prices = np.array([])

    coconut_spreads = np.array([])
    #coupon_spreads = np.array([])
    ccnt_prices = deque(maxlen=ccnt_lag2+1)

    sb_prices = deque(maxlen=sb_lag2+1)

    def prepare_data(self, product):
        return (
            self.position[product],
            self.POSITION_LIMIT[product]
        )

    def weighted_price(self, order_dict):
        total_vol = 0
        res = 0

        for price, vol in order_dict:
            res += price * abs(vol)
            total_vol += abs(vol)

        return int(round(res / total_vol))

    def compute_prediction(self):
        predict = self.reg_intercept
        for coef, val in zip(self.reg_params, self.starfruit_midprice):
            predict += coef * val

        return int(round(predict))

    def order_orchids(self, state):
        product = 'ORCHIDS'
        curr_position, position_limit = self.prepare_data(product)
        order_depth: OrderDepth = state.order_depths[product]

        book_ask = sorted(order_depth.sell_orders.items())
        book_bid = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])

        cur_obs = state.observations.conversionObservations

        export_sell_prc = cur_obs[product].bidPrice - cur_obs[product].transportFees - cur_obs[product].exportTariff
        import_buy_prc = cur_obs[product].askPrice + cur_obs[product].transportFees + cur_obs[product].importTariff

        orders: list[Order] = []

        # Humidity
        hum = cur_obs[product].humidity
        hum_signal = np.exp(abs(hum-70)/10)

        # Sell if Humidity signal
        for bid_price, vol in book_bid:
            if hum_signal > 13:
                order_for = max(-position_limit - curr_position, -vol)
                orders.append(Order(product, bid_price, order_for))
                curr_position += order_for
                self.humid_orchids_weighted_sell_price += bid_price * abs(order_for)
                self.humid_orchids_total_volume += abs(order_for)

        if self.humid_orchids_total_volume != 0:
            humid_orchids_avg_sell_price = self.humid_orchids_weighted_sell_price / self.humid_orchids_total_volume
        else:
            humid_orchids_avg_sell_price = 0

        # Buy back if hum signal back to normal
        if (import_buy_prc < humid_orchids_avg_sell_price) & (hum_signal < 4):
            self.conversions += self.humid_orchids_total_volume
            self.humid_orchids_weighted_sell_price = 0
            self.humid_orchids_total_volume = 0

        # Add price
        best_ask, best_ask_amount = list(book_ask)[0]
        best_bid, best_bid_amount = list(book_bid)[0]
        mid_price = (best_bid + best_ask) / 2
        self.orchids_price.append(mid_price)

        # Truncate if long
        if len(self.orchids_price) > 300:
            self.orchids_price = self.orchids_price[-300:]

        # Moving average (300)
        if len(self.orchids_price) != 0:
            orchids_ma = sum(self.orchids_price) / len(self.orchids_price)
        else:
            orchids_ma = mid_price

        # Sell if too high
        for bid_price, vol in book_bid:
            if bid_price > orchids_ma + 16:
                order_for = max(-position_limit - curr_position, -vol)
                orders.append(Order(product, bid_price, order_for))
                curr_position += order_for
                self.orchids_weighted_sell_price += bid_price * abs(order_for)
                self.orchids_total_volume += abs(order_for)

        if self.orchids_total_volume != 0:
            orchids_avg_sell_price = self.orchids_weighted_sell_price / self.orchids_total_volume
        else:
            orchids_avg_sell_price = 0

        # Conversion if enough margin
        if (import_buy_prc < orchids_avg_sell_price - 12):
            self.conversions += self.orchids_total_volume
            self.orchids_weighted_sell_price = 0
            self.orchids_total_volume = 0

        # Previous code
        sell_positions = curr_position
        for bid_price, vol in book_bid:
            if sell_positions != 0 and self.orchids_sell_prc > import_buy_prc:
                # if price discrepancy still exists after going short, buy back abroad
                self.conversions += abs(self.orchids_sell_qty)
                self.orchids_sell_prc = 0
                self.orchids_sell_qty = 0
            if sell_positions > -position_limit and (bid_price > import_buy_prc):
                # sell at island
                order_for = -position_limit - sell_positions
                orders.append(Order(product, bid_price, order_for))
                self.orchids_sell_prc = bid_price
                self.orchids_sell_qty = order_for
            break

        return orders

    def order_starfruit(self, state):
        product = 'STARFRUIT'

        curr_position, position_limit = self.prepare_data('STARFRUIT')
        order_depth: OrderDepth = state.order_depths[product]

        book_ask = sorted(order_depth.sell_orders.items())
        book_bid = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])

        best_ask_price = self.weighted_price(book_ask)
        best_bid_price = self.weighted_price(book_bid)

        self.starfruit_midprice.append((best_ask_price + best_bid_price) / 2)

        if len(self.starfruit_midprice) == self.starfruit_midprice.maxlen:
            predict_bid = self.compute_prediction() - 1
            predict_ask = self.compute_prediction() + 1
        else:
            predict_bid = -int(1e9)
            predict_ask = int(1e9)

        orders: list[Order] = []

        buy_positions = curr_position
        for ask_price, vol in book_ask:
            if buy_positions < position_limit and (
                    ask_price <= predict_bid or (curr_position < 0 and ask_price == predict_bid + 1)):
                order_for = min(-vol, position_limit - buy_positions)
                buy_positions += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask_price, order_for))

        high_bid = min(best_bid_price + 1, predict_bid)
        if buy_positions < position_limit:
            orders.append(Order(product, high_bid, position_limit - buy_positions))

        sell_positions = curr_position
        for bid_price, vol in book_bid:
            if sell_positions > -position_limit and (
                    bid_price >= predict_ask or (curr_position > 0 and bid_price + 1 == predict_ask)):
                order_for = max(-vol, -position_limit - sell_positions)
                sell_positions += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid_price, order_for))

        low_ask = max(best_ask_price - 1, predict_ask)
        if sell_positions > -position_limit:
            orders.append(Order(product, low_ask, -position_limit - sell_positions))

        return orders

    def order_amethysts(self, state):
        orders: list[Order] = []

        product = 'AMETHYSTS'
        curr_position, position_limit = self.prepare_data(product)
        order_depth: OrderDepth = state.order_depths[product]

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        if (best_bid > 10000):
            orders.append(Order(product, best_bid, -best_bid_amount))  # The bid and ask amount impact market.
            curr_position += -best_bid_amount
        elif (best_ask < 10000):
            orders.append(Order(product, best_ask, -best_ask_amount))
            curr_position += -best_ask_amount

        if (best_bid <= 9998):
            order_for = position_limit - curr_position
            orders.append(Order(product, best_bid + 1, order_for))

        if (best_ask >= 10002):
            order_for = -position_limit - curr_position
            orders.append(Order(product, best_ask - 1, order_for))

        return orders

    def compute_orders_basket(self, state):

        orders = {'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []}
        products = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, wg_buy, wg_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in products:
            order_depth = state.order_depths[p]
            osell[p] = sorted(order_depth.sell_orders.items())
            obuy[p] = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])

            best_sell[p] = osell[p][0][0]
            best_buy[p] = obuy[p][0][0]

            worst_sell[p] = osell[p][-1][0]
            worst_buy[p] = obuy[p][-1][0]

            wg_buy[p] = self.weighted_price(obuy[p])
            wg_sell[p] = self.weighted_price(osell[p])

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2

        spread = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE'] * 4 - mid_price['STRAWBERRIES'] * 6 - mid_price[
            'ROSES']
        rc_spread = mid_price['ROSES'] - 1.8327*mid_price['CHOCOLATE']
        cs_spread = mid_price['CHOCOLATE'] - mid_price['STRAWBERRIES']*1.9656
        self.spreads.append(spread)
        self.rc_spreads.append(rc_spread)
        self.c_prices = np.append(self.c_prices, mid_price['CHOCOLATE'])
        self.s_prices = np.append(self.s_prices, mid_price['STRAWBERRIES']*1.9656)

        amt = 6

        if len(self.spreads) == WINDOW:

            # GIFT_BASKET spread signal trade
            avg_spread = sum(self.spreads)/WINDOW
            std_spread = 0
            for s in self.spreads:
                std_spread += (s - avg_spread)**2
            std_spread /= WINDOW
            std_spread **= 0.5
            spread_5 = sum(list(self.spreads)[-5:])/5

            if spread_5 < avg_spread - 2*std_spread:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], amt))
                #orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], 4*amt))
                #orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], amt))
                #orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], 2*amt))

            elif spread_5 > avg_spread + 2*std_spread:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -amt))
                #orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -amt*4))
                #orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -amt))
                #orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -2*amt))

            # CHOCOLATE & STRAWBERRIES pair trades based on trailing covariance thresholding (only CHOCOLATE seems to work)
            avg_cs_spread = (self.c_prices - self.s_prices).mean()
            std_cs_spread = (self.c_prices - self.s_prices).std()

            corr_cs_spread = np.corrcoef(self.c_prices, self.s_prices)[0, 1] # may need to further incorporate recent history of cov changes

            if cs_spread < avg_cs_spread - 1.5*std_cs_spread:
                if corr_cs_spread > 0.1: # Maybe add another step to decide to both long or short based on trend
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -10))
                    #orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -19))

                elif corr_cs_spread < -0.1:
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], 10))
                    #orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -19))

            elif cs_spread > avg_cs_spread + 1.5*std_cs_spread:
                if corr_cs_spread > 0.1: # Maybe add another step to decide to both long or short based on trend
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], 10))
                    #orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], 19))

                elif corr_cs_spread < -0.1:
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -10))
                    #orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], 19))

        # Switch to momentum-based for STRAWBERRIES => not trading - too instable
        # self.sb_prices.append(mid_price['STRAWBERRIES'])
        #
        # if len(self.sb_prices) == sb_lag2+1:
        #     lag1_ret = (self.sb_prices[-1] - self.sb_prices[-sb_lag1])/self.sb_prices[-sb_lag1]
        #     lag2_ret = (self.sb_prices[-1] - self.sb_prices[1])/self.sb_prices[1]
        #     lag1_ret_prev = (self.sb_prices[-2] - self.sb_prices[-sb_lag1-1])/self.sb_prices[-sb_lag1-1]
        #     #lag2_ret_prev = (self.sb_prices[-2] - self.sb_prices[0]) / self.sb_prices[0]
        #
        #     if lag1_ret_prev < lag1_ret and lag1_ret_prev > 0 and lag2_ret > 0 and lag2_ret < lag1_ret:
        #         orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], 35))
        #
        #     if lag1_ret_prev > lag1_ret and lag1_ret_prev < 0 and lag2_ret < 0 and lag2_ret > lag1_ret:
        #         orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -35))

            # if lag1_ret_prev < lag2_ret_prev and lag1_ret > lag2_ret:
            #     orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], 60))
            #
            # if lag1_ret_prev > lag2_ret_prev and lag1_ret < lag2_ret:
            #     orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -60))

        if len(self.rc_spreads) == rc_WINDOW: # ROSES spread trading based on CHOCOLATE signal
            avg_spread = sum(self.spreads)/rc_WINDOW
            std_spread = 0
            for s in self.spreads:
                std_spread += (s - avg_spread)**2
            std_spread /= rc_WINDOW
            std_spread **= 0.5

            if rc_spread <= avg_spread - 1.5*std_spread:
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], amt))
            elif rc_spread >= avg_spread + 1.5*std_spread:
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -amt))

        if self.c_prices.shape[0] == WINDOW:
            self.c_prices = self.c_prices[1:]
            self.s_prices = self.s_prices[1:]

        return orders

    def order_coconuts(self, state):
        orders = {'COCONUT': [], 'COCONUT_COUPON': []}
        products = ['COCONUT', 'COCONUT_COUPON']

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}

        for p in products:
            order_depth = state.order_depths[p]
            osell[p] = sorted(order_depth.sell_orders.items())
            obuy[p] = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])

            best_sell[p] = osell[p][0][0]
            best_buy[p] = obuy[p][0][0]

            worst_sell[p] = osell[p][-1][0]
            worst_buy[p] = obuy[p][-1][0]

            mid_price[p] = (best_buy[p] + best_sell[p])/2

        trading_start_thresh = 50

        spread = mid_price['COCONUT_COUPON'] - 0.5*mid_price['COCONUT']
        self.coconut_spreads = np.append(self.coconut_spreads, spread)

        #coupon_spread = mid_price['COCONUT'] - 1.8246*mid_price['COCONUT_COUPON']
        #self.coupon_spreads = np.append(self.coupon_spreads, coupon_spread)

        base_amt = 30

        if state.timestamp > trading_start_thresh*100:
            spread_mean = self.coconut_spreads.mean()
            spread_std = self.coconut_spreads.std()

            if spread > spread_mean + spread_std:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_buy['COCONUT_COUPON'], -2*base_amt))
                #orders['COCONUT'].append(Order('COCONUT', worst_sell['COCONUT'], base_amt))

            if spread < spread_mean - spread_std:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', worst_sell['COCONUT_COUPON'], 2 * base_amt))
                #orders['COCONUT'].append(Order('COCONUT', worst_buy['COCONUT'], -base_amt))

        # if self.coupon_spreads.shape[0] == WINDOW:
        #     coupon_spread_mean = self.coupon_spreads.mean()
        #     coupon_spread_std = self.coupon_spreads.std()
        #
        #     if coupon_spread > coupon_spread_mean + coupon_spread_std:
        #         #orders['COCONUT'].append(Order('COCONUT', worst_buy['COCONUT'], -base_amt))
        #         orders['COCONUT'].append(Order('COCONUT', worst_sell['COCONUT'], base_amt))
        #
        #     if coupon_spread < coupon_spread_mean - coupon_spread_std:
        #         #orders['COCONUT'].append(Order('COCONUT', worst_sell['COCONUT'], base_amt))
        #         orders['COCONUT'].append(Order('COCONUT', worst_buy['COCONUT'], -base_amt))
        #
        #     self.coupon_spreads = self.coupon_spreads[1:]

        # Switching to momentum for COCONUT
        self.ccnt_prices.append(mid_price['COCONUT'])

        if len(self.ccnt_prices) == ccnt_lag2+1:
            lag1_ret = (self.ccnt_prices[-1] - self.ccnt_prices[-ccnt_lag1])/self.ccnt_prices[-ccnt_lag1]
            lag2_ret = (self.ccnt_prices[-1] - self.ccnt_prices[1])/self.ccnt_prices[1]
            lag1_ret_prev = (self.ccnt_prices[-2] - self.ccnt_prices[-ccnt_lag1-1])/self.ccnt_prices[-ccnt_lag1-1]
            #lag2_ret_prev = (self.ccnt_prices[-2] - self.ccnt_prices[0]) / self.ccnt_prices[0]

            if lag1_ret_prev < lag1_ret and lag1_ret_prev > 0 and lag2_ret > 0 and lag2_ret < lag1_ret:
                orders['COCONUT'].append(Order('COCONUT', worst_sell['COCONUT'], 30))

            if lag1_ret_prev > lag1_ret and lag1_ret_prev < 0 and lag2_ret < 0 and lag2_ret > lag1_ret:
                orders['COCONUT'].append(Order('COCONUT', worst_buy['COCONUT'], -30))

        return orders

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        self.conversions = 0

        result = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': [], 'CHOCOLATE': [],
                  'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': [], 'COCONUT': [], 'COCONUT_COUPON': []}

        for key, val in state.position.items():
            self.position[key] = val

        result['STARFRUIT'] += self.order_starfruit(state)
        result['AMETHYSTS'] += self.order_amethysts(state)
        result['ORCHIDS'] += self.order_orchids(state)

        orders = self.compute_orders_basket(state)
        result['GIFT_BASKET'] += orders['GIFT_BASKET']
        result['ROSES'] += orders['ROSES']
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['STRAWBERRIES'] += orders['STRAWBERRIES']

        rnd4_orders = self.order_coconuts(state)
        result['COCONUT'] += rnd4_orders['COCONUT']
        result['COCONUT_COUPON'] += rnd4_orders['COCONUT_COUPON']

        traderData = "yhlee"
        conversions = self.conversions

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData