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

class Trader:
    position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0}
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100}
    
    starfruit_reg_params = [
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
    
    starfruit_reg_intercept = 0.6951698609096012
    
    starfruit_midprice = deque(maxlen=len(starfruit_reg_params))

    orchids_reg_intercept = 0.5529136448478766 
    orchids_reg_params = [
        -2.10434807e-03, -5.42746759e-03,  8.87177642e-03, -6.40110869e-03,
        5.77493886e-03, -5.94693972e-04, -5.01163871e-03, -3.23051743e-03,
        1.59362230e-02, -5.45898258e-03, -2.94750033e-03, -5.76994740e-03,
        9.26883850e-04,  1.97981622e-03,  1.69280738e-02, -8.55105816e-03,
        -7.62958119e-03,  1.00749032e-02, -1.24811554e-02,  1.00505265e+00,
        -4.93967959e-03
    ]
    orchids_midprice = deque(maxlen=len(orchids_reg_params)-1)

    conversions = 0

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

    def compute_prediction(self, intercept, params, prices):
        predict = intercept
        for coef, val in zip(params, prices):
            predict += coef * val
        return int(round(predict))

    def order_starfruit(self, state):
        product = 'STARFRUIT'
        orders: list[Order] = []

        curr_position, position_limit = self.prepare_data('STARFRUIT')
        order_depth: OrderDepth = state.order_depths[product]

        book_ask = sorted(order_depth.sell_orders.items())
        book_bid = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])
        
        weighted_ask_price = self.weighted_price(book_ask)
        weighted_bid_price = self.weighted_price(book_bid)

        self.starfruit_midprice.append((weighted_ask_price + weighted_bid_price) / 2)
        predict_price = self.compute_prediction(
            self.starfruit_reg_intercept,
            self.starfruit_reg_params,
            self.starfruit_midprice
        )

        if len(self.starfruit_midprice) == self.starfruit_midprice.maxlen:
            predict_bid = predict_price - 1
            predict_ask = predict_price + 1
        else:
            predict_bid = -int(1e9)
            predict_ask = int(1e9)

        buy_positions = curr_position
        for ask_price, vol in book_ask:
            if buy_positions < position_limit and (ask_price <= predict_bid or (curr_position < 0 and ask_price == predict_bid + 1)):
                order_for = min(-vol, position_limit - buy_positions)
                buy_positions += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask_price, order_for))

        high_bid = min(weighted_bid_price + 1, predict_bid)
        if buy_positions < position_limit:
            orders.append(Order(product, high_bid, position_limit - buy_positions))

        sell_positions = curr_position
        for bid_price, vol in book_bid:
            if sell_positions > -position_limit and (bid_price >= predict_ask or (curr_position > 0 and bid_price + 1 == predict_ask)):
                order_for = max(-vol, -position_limit - sell_positions)
                sell_positions += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid_price, order_for))

        low_ask = max(weighted_ask_price - 1, predict_ask)
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

    def order_orchids(self, state):
        orders: list[Order] = []
        product = 'ORCHIDS'
        curr_position, position_limit = self.prepare_data(product)
        order_depth: OrderDepth = state.order_depths[product]

        observation = state.observations.conversionObservations[product]

        book_ask = sorted(order_depth.sell_orders.items())
        book_bid = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])
        
        # weighted_ask_price = self.weighted_price(book_ask)
        # weighted_bid_price = self.weighted_price(book_bid)

        # self.orchids_midprice.append((weighted_ask_price + weighted_bid_price) / 2)
        # predict_price = self.compute_prediction(
        #     self.orchids_reg_intercept,
        #     self.orchids_reg_params,
        #     self.orchids_midprice
        # )

        # if len(self.orchids_midprice) == self.orchids_midprice.maxlen:
        #     predict_bid = predict_price - 1
        #     predict_ask = predict_price + 1
        # else:
        #     predict_bid = -int(1e9)
        #     predict_ask = int(1e9)

        # buy_positions = curr_position
        # for ask_price, vol in book_ask:
        #     if buy_positions < position_limit and (ask_price <= predict_bid or (curr_position < 0 and ask_price == predict_bid + 1)):
        #         order_for = min(-vol, position_limit - buy_positions)
        #         buy_positions += order_for
        #         assert (order_for >= 0)
        #         orders.append(Order(product, ask_price, order_for))

        # high_bid = min(weighted_bid_price + 1, predict_bid)
        # if buy_positions < position_limit:
        #     orders.append(Order(product, high_bid, position_limit - buy_positions))

        # sell_positions = curr_position
        # for bid_price, vol in book_bid:
        #     if sell_positions > -position_limit and (bid_price >= predict_ask or (curr_position > 0 and bid_price + 1 == predict_ask)):
        #         order_for = max(-vol, -position_limit - sell_positions)
        #         sell_positions += order_for
        #         assert (order_for <= 0)
        #         orders.append(Order(product, bid_price, order_for))

        # low_ask = max(weighted_ask_price - 1, predict_ask)
        # if sell_positions > -position_limit:
        #     orders.append(Order(product, low_ask, -position_limit - sell_positions))


        adj_bid_price = observation.bidPrice - observation.transportFees - observation.exportTariff
        adj_ask_price = observation.askPrice + observation.transportFees + observation.importTariff

        # if curr_position != 0:
        #     self.conversions -= curr_position

        for ask_price, vol in book_ask:
            if (ask_price < adj_bid_price):
                order_for = min(position_limit - curr_position, -vol)
                orders.append(Order(product, ask_price, order_for))
                self.conversions -= order_for
 
        for bid_price, vol in book_bid:
            if (adj_ask_price < bid_price):
                order_for = max(-position_limit - curr_position, -vol)
                orders.append(Order(product, bid_price, order_for))
                self.conversions -= order_for

        return orders
    
    def run(self, state: TradingState):
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        result = {
            'STARFRUIT': [],
            'AMETHYSTS': [],
            'ORCHIDS' : []
        }

        for key, val in state.position.items():
            self.position[key] = val

        # result['STARFRUIT'] = self.order_starfruit(state)
        # result['AMETHYSTS'] = self.order_amethysts(state)
        result['ORCHIDS'] = self.order_orchids(state)
        
        traderData = "hwjang"
        conversions = self.conversions  

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
