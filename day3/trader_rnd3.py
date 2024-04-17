from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import json
import pandas as pd
import numpy as np
import statistics
import math
from collections import deque, defaultdict


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
    position = {
        'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0,
        'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0,
    }
    POSITION_LIMIT = {
        'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100,
        'CHOCOLATE': 240, 'STRAWBERRIES': 348, 'ROSES': 58, 'GIFT_BASKET': 58,
    }

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

    correction = 379.4904833333333
    enter_threshold = 70
    clear_threshold = 0

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
    
    def clear_position(self, long_basket, prod_list, position_dict, limit_dict, ask_dict, bid_dict):
        orders_dict = defaultdict(list)
        for prod in prod_list:
            if (long_basket and prod != 'GIFT_BASKET') or (not long_basket and prod == 'GIFT_BASKET'):
                best_bid_price, bid_amount = bid_dict[prod]
                order_for_best = max(-bid_amount, -position_dict[prod])
                if order_for_best < 0:
                    orders_dict[prod].append(Order(prod, best_bid_price, order_for_best))
                order_for_rest = -position_dict[prod]-order_for_best
                if order_for_rest < 0:
                    orders_dict[prod].append(Order(prod, best_bid_price + 1, order_for_rest//2))
                    orders_dict[prod].append(Order(prod, best_bid_price + 2, order_for_rest//2))
            else:
                best_ask_price, ask_amount = ask_dict[prod]
                order_for_best = min(-ask_amount, -position_dict[prod])
                if order_for_best > 0:
                    orders_dict[prod].append(Order(prod, best_ask_price, order_for_best))
                order_for_rest = -position_dict[prod]-order_for_best
                if order_for_rest > 0:
                    orders_dict[prod].append(Order(prod, best_ask_price - 1, order_for_rest//2))
                    orders_dict[prod].append(Order(prod, best_ask_price - 2, order_for_rest//2))
        return orders_dict
    
    def enter_position(self, long_basket, prod_list, position_dict, limit_dict, ask_dict, bid_dict):
        orders_dict = defaultdict(list)
        for prod in prod_list:
            if (long_basket and prod != 'GIFT_BASKET') or (not long_basket and prod == 'GIFT_BASKET'):
                best_bid_price, bid_amount = bid_dict[prod]
                order_for_best = max(-bid_amount, -limit_dict[prod]-position_dict[prod])
                if order_for_best < 0:
                    orders_dict[prod].append(Order(prod, best_bid_price, order_for_best))
                order_for_rest = -limit_dict[prod]-position_dict[prod]-order_for_best
                if order_for_rest < 0:
                    orders_dict[prod].append(Order(prod, best_bid_price + 1, order_for_rest//2))
                    orders_dict[prod].append(Order(prod, best_bid_price + 2, order_for_rest//2))
            else:
                best_ask_price, ask_amount = ask_dict[prod]
                order_for_best = min(-ask_amount, limit_dict[prod]-position_dict[prod])
                if order_for_best > 0:
                    orders_dict[prod].append(Order(prod, best_ask_price, order_for_best))
                order_for_rest = limit_dict[prod]-position_dict[prod]-order_for_best
                if order_for_rest > 0:
                    orders_dict[prod].append(Order(prod, best_ask_price - 1, order_for_rest//2))
                    orders_dict[prod].append(Order(prod, best_ask_price - 2, order_for_rest//2))
        return orders_dict
    
    def order_bakset(self, state):
        prod1 = 'CHOCOLATE'
        prod2 = 'STRAWBERRIES'
        prod3 = 'ROSES'
        basket = 'GIFT_BASKET'
        prod_list = [prod1, prod2, prod3, basket]
        weights = {prod1: 4, prod2: 6, prod3: 1}

        position_dict = {}
        limit_dict = {}
        ask_dict = {}
        bid_dict = {}
        mid_price_dict = {}

        orders_dict = {
            prod1: [],
            prod2: [],
            prod3: [],
            basket: []
        }

        for prod in prod_list:
            position, limit = self.prepare_data(prod)
            position_dict[prod] = position
            limit_dict[prod] = limit
            order_depth = state.order_depths[prod]
            book_ask = sorted(order_depth.sell_orders.items())
            book_bid = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])
            best_ask_price = self.weighted_price(book_ask)
            best_bid_price = self.weighted_price(book_bid)
            mid_price = (best_ask_price + best_bid_price)/2
            mid_price_dict[prod] = mid_price
            ask_dict[prod] = book_ask[0]
            bid_dict[prod] = book_bid[0]
            
        comb_price = sum([weights[prod] * mid_price_dict[prod] for prod in prod_list[:-1]])
        price_residual = mid_price_dict[basket] - self.correction - comb_price
        logger.print(f"Comb price : {comb_price}, Basket price : {mid_price_dict[basket]}, Residual : {price_residual}")
        
        # Exit short-basket (long-basket)
        if -self.enter_threshold < price_residual < -self.clear_threshold:
            return self.clear_position(True, prod_list, position_dict, limit_dict, ask_dict, bid_dict)
        # Exit long-basket (short-basket)
        elif self.clear_threshold < price_residual < self.enter_threshold:
            return self.clear_position(False, prod_list, position_dict, limit_dict, ask_dict, bid_dict)
    
        # Enter long-basket
        if price_residual < -self.enter_threshold:
            return self.enter_position(True, prod_list, position_dict, limit_dict, ask_dict, bid_dict)
        # Enter short-basket
        elif self.enter_threshold < price_residual:
            return self.enter_position(False, prod_list, position_dict, limit_dict, ask_dict, bid_dict)
    
        return orders_dict
    
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

        if curr_position != 0:
            self.conversions = -curr_position

        best_ask_price, _ = book_ask[0]
        if curr_position < position_limit and (best_ask_price < export_sell_prc):
            order_for = position_limit - curr_position
            orders.append(Order(product, best_ask_price, order_for))
        
        best_bid_price, _ = book_bid[0]
        if curr_position > -position_limit and (best_bid_price > import_buy_prc):
            order_for = -position_limit - curr_position
            orders.append(Order(product, best_bid_price, order_for))

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

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        self.conversions = 0

        result = {
            'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS': [],
            'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': [],
        }

        for key, val in state.position.items():
            self.position[key] = val

        # result['STARFRUIT'] = self.order_starfruit(state)
        # result['AMETHYSTS'] = self.order_amethysts(state)
        # result['ORCHIDS'] = self.order_orchids(state)
        basket_orders_dict = self.order_bakset(state)
        for product, orders in basket_orders_dict.items():
            result[product] = orders
        
        traderData = "hwjang"
        conversions = self.conversions

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
