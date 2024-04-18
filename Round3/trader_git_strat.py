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

class Trader:
    position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0,
                'GIFT_BASKET': 0}
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350,
                      'ROSES': 60, 'GIFT_BASKET': 60}

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

    spreads = deque(maxlen=WINDOW)

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

        buy_positions = curr_position
        for ask_price, vol in book_ask:
            if buy_positions != 0 and self.orchids_buy_prc < export_sell_prc:
                # if price discrepancy still exists after going long, sell abroad
                self.conversions = -buy_positions
                self.orchids_buy_prc = 0
            if buy_positions < position_limit and (ask_price < export_sell_prc):
                # buy at island
                order_for = position_limit - buy_positions
                orders.append(Order(product, ask_price, order_for))
                self.orchids_buy_prc = ask_price
            break

        sell_positions = curr_position
        for bid_price, vol in book_bid:
            if sell_positions != 0 and self.orchids_sell_prc > import_buy_prc:
                # if price discrepancy still exists after going short, buy back abroad
                self.conversions = -sell_positions
                self.orchids_sell_prc = 0
            if sell_positions > -position_limit and (bid_price > import_buy_prc):
                # sell at island
                order_for = -position_limit - sell_positions
                orders.append(Order(product, bid_price, order_for))
                self.orchids_sell_prc = bid_price
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
        self.spreads.append(spread)

        amt = 5

        if len(self.spreads) == WINDOW:
            avg_spread = sum(self.spreads)/WINDOW
            std_spread = 0
            for s in self.spreads:
                std_spread += (s - avg_spread)**2
            std_spread /= WINDOW
            std_spread **= 0.5
            spread_5 = sum(list(self.spreads)[-5:])/5

            if spread_5 < avg_spread - 2*std_spread:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], amt))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -4*amt))
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -6*amt))
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -amt))

            elif spread_5 > avg_spread + 2*std_spread:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -amt))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], 4 * amt))
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], 6 * amt))
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], amt))

        return orders

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        self.conversions = 0

        result = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': [], 'CHOCOLATE': [],
                  'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []}

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

        traderData = "yhlee"
        conversions = self.conversions

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData