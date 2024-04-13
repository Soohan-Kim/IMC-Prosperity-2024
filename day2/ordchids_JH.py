from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List
# import statistics
# import math
import sys
import json
from collections import deque

from typing import Any


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


class Trader:
    position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0}
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100}

    reg_params = [-0.01869561, 0.0455032, 0.16316049, 0.8090892]
    reg_intercept = 4.481696494462085
    starfruit_midprice = deque(maxlen=len(reg_params))

    orchid_regression_params = [-11, 8.5, 5.4]  # [sunlight_less_than_7hrs, shipping_cost, humidity_diff_outside_range]
    orchid_regression_intercept = 1000
    orchid_predictors = deque(maxlen=3)  # assuming three predictors as per coefficients


    def prepare_data(self, product):
        return (
            self.position[product],
            self.POSITION_LIMIT[product]
        )

    def best_price(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        max_vol = -1

        for ask, vol in order_dict:
            if buy == 0:
                vol *= -1
            tot_vol += vol
            if tot_vol > max_vol:
                max_vol = vol
                best_val = ask

        return best_val

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

    def order_starfruit(self, state):
        product = 'STARFRUIT'

        curr_position, position_limit = self.prepare_data('STARFRUIT')
        order_depth: OrderDepth = state.order_depths[product]

        book_ask = sorted(order_depth.sell_orders.items())
        book_bid = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])

        logger.print(book_bid)
        logger.print(book_ask)

        # best_ask_price = book_ask[0][0]
        # best_bid_price = book_bid[0][0]

        best_ask_price = self.best_price(book_ask, 0)
        best_bid_price = self.best_price(book_bid, 1)

        # best_ask_price = self.weighted_price(book_ask)
        # best_bid_price = self.weighted_price(book_bid)

        self.starfruit_midprice.append((best_ask_price + best_bid_price) / 2)

        if len(self.starfruit_midprice) == self.starfruit_midprice.maxlen:
            predict_bid = self.compute_prediction() - 1
            predict_ask = self.compute_prediction() + 1
        else:
            predict_bid = -int(1e9)
            predict_ask = int(1e9)

        orders: list[Order] = []

        sell_positions = self.position['STARFRUIT']
        for p, vol in book_ask:
            if sell_positions < position_limit and \
                    (p <= predict_bid or (curr_position < 0 and p == predict_bid + 1)):
                order_for = min(-vol, position_limit - sell_positions)
                sell_positions += order_for
                assert (order_for >= 0)
                orders.append(Order(product, p, order_for))

        high_bid = min(best_bid_price + 1, predict_bid)
        if sell_positions < position_limit:
            orders.append(Order(product, high_bid, position_limit - sell_positions))

        buy_positions = self.position['STARFRUIT']
        for p, vol in book_bid:
            if buy_positions > -position_limit and \
                    (p >= predict_ask or (curr_position > 0 and p + 1 == predict_ask)):
                order_for = max(-vol, -position_limit - buy_positions)
                buy_positions += order_for
                assert (order_for <= 0)
                orders.append(Order(product, p, order_for))

        low_ask = max(best_ask_price - 1, predict_ask)
        if buy_positions > -position_limit:
            orders.append(Order(product, low_ask, -position_limit - buy_positions))

        return orders

    def compute_orchid_prediction(self, observation):
        predict = self.orchid_regression_intercept
        # Construct predictors based on observation data
        sunlight_less_than_7hrs = 1 if observation.sunlight < 4200 else 0
        shipping_cost = observation.transportFees + observation.exportTariff + observation.importTariff
        humidity_diff_outside_range = abs(observation.humidity - 70) if observation.humidity < 60 or observation.humidity > 80 else 0

        predictors = [sunlight_less_than_7hrs, shipping_cost, humidity_diff_outside_range]

        for coef, val in zip(self.orchid_regression_params, predictors):
            predict += coef * val

        return int(round(predict))

    def order_orchids(self, state):
        product = 'ORCHIDS'
        curr_position, position_limit = self.prepare_data(product)
        order_depth: OrderDepth = state.order_depths[product]

        # Get the latest market observations
        observation = state.observations[product]  # Assuming observations is a dictionary of ConversionObservations

        predict_price = self.compute_orchid_prediction(observation)

        orders = []
        book_ask = sorted(order_depth.sell_orders.items())
        book_bid = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])

        best_ask_price = self.best_price(book_ask, 0)
        best_bid_price = self.best_price(book_bid, 1)

        if predict_price < best_ask_price and curr_position < position_limit:
            orders.append(Order(product, predict_price, position_limit - curr_position))
        if predict_price > best_bid_price and curr_position > -position_limit:
            orders.append(Order(product, predict_price, -position_limit - curr_position))

        return orders

    def run(self, state: TradingState):
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        result = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': []}

        for key, val in state.position.items():
            self.position[key] = val

        result['STARFRUIT'] += self.order_starfruit(state)
        result['ORCHIDS'] += self.order_orchids(state)

        for product, order_depth in state.order_depths.items():
            if product == "AMETHYSTS":

                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                if (best_bid > 10000):
                    result[product].append(Order(product, best_bid, -best_bid_amount))  # The bid and ask amount impact market.
                elif (best_ask < 10000):
                    result[product].append(Order(product, best_ask, -best_ask_amount))

                if (best_bid == 9995):
                    result[product].append(Order(product, 9996, 10))
                elif (best_bid == 9996):
                    result[product].append(Order(product, 9997, 10))
                elif (best_bid == 9997):
                    result[product].append(Order(product, 9998, 10))

                if (best_ask == 10005):
                    result[product].append(Order(product, 10004, -10))
                elif (best_ask == 10004):
                    result[product].append(Order(product, 10003, -10))
                elif (best_ask == 10003):
                    result[product].append(Order(product, 10002, -10))

        traderData = "yhlee"
        conversions = 0

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

# prosperity2bt ~\PycharmProjects\IMC-Prosperity-2024\juhyungkang\trader_regression_ju_modified.py 1
