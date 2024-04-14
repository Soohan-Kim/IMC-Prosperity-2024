#from round1_datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import logging
import jsonpickle

import json
from samp_datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
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

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))
        #print("Market trades:" + str(state.market_trades))

        pos_limits = {'AMETHYSTS': 20, 'STARFRUIT': 20}

        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            #print("Order book sell:" + str(order_depth.sell_orders))
            #print("Order book buy:" + str(order_depth.buy_orders))

            #acceptable_price = 10000 if product == 'AMETHYSTS' else 5000
            acceptable_price = 0

            #try:
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                total = 0
                for k in order_depth.sell_orders:
                    if k:
                        acceptable_price += k*(-order_depth.sell_orders[k])
                        total -= order_depth.sell_orders[k]
                for k in order_depth.buy_orders:
                    if k:
                        acceptable_price += k*order_depth.buy_orders[k]
                        total += order_depth.buy_orders[k]

                acceptable_price /= total

            #cur_pos = state.position[product] if state.position[product] else 0
            cur_pos = state.position.get(product, 0)
            # except Exception as e:
            #     logging.debug(e)

            #print("Acceptable price : " + str(acceptable_price))
            #print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
            #    len(order_depth.sell_orders)))

            mid_price = 0
            spread = 0
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                mid_price = int((best_ask + best_bid)/2)
                spread = best_ask - best_bid

            #if len(order_depth.sell_orders) != 0:
                #amt = -((pos_limits[product] + cur_pos)//2)
                #orders.append(Order(product, mid_price+1, amt))
                #orders.append(Order(product, mid_price, amt))
                #best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    for (ask_info) in list(order_depth.sell_orders.items()):
                        ask_price, ask_amount = ask_info
                    # print("BUY", str(-best_ask_amount) + "x", best_ask)
                    # orders.append(Order(product, best_ask, -best_ask_amount))
                        amt = pos_limits[product] - cur_pos
                    #print("BUY", str(amt) + "x", best_ask)
                        if int(ask_price) < acceptable_price:
                            orders.append(Order(product, ask_price, min(-ask_amount, amt)))
                            cur_pos += min(-ask_amount, amt)
                # elif acceptable_price > int(best_bid):
                #     allocs = [p for p in range(int(best_ask), int(acceptable_price), -1)]
                #     amt = -((pos_limits[product] + cur_pos)//len(allocs))
                #     for p in allocs:
                #         orders.append(Order(product, p, amt))

                # if mid_price < acceptable_price:
                #     orders.append(Order(product, best_ask, 10))
                #     orders.append(Order(product, best_ask-1, 10))
                    #alloc = -((pos_limits[product] + cur_pos) // 2)
                    #orders.append(Order(product, best_ask, alloc))
                    #orders.append(Order(product, best_ask-1, alloc))
                    #orders.append(Order(product, best_ask-3, alloc))
                    #orders.append(Order(product, best_ask-4, alloc))
                    #orders.append((Order(product, int(mid_price)+1, -10)))
                    #orders.append((Order(product, int(mid_price), -10)))

            #if len(order_depth.buy_orders) != 0:
                #amt = (pos_limits[product] - cur_pos)//2
                #orders.append(Order(product, mid_price, amt))
                #orders.append(Order(product, mid_price-1, amt))
                #best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    # print("SELL", str(best_bid_amount) + "x", best_bid)
                    # orders.append(Order(product, best_bid, -best_bid_amount))
                    for bid_info in list(order_depth.buy_orders.items()):
                        bid_price, bid_amount = bid_info
                        amt = -pos_limits[product] - cur_pos
                        #print("SELL", str(amt) + "x", best_bid)
                        if int(bid_price) > acceptable_price:
                            orders.append(Order(product, bid_price, max(amt, -bid_amount)))
                            cur_pos -= max(amt, -bid_amount)
                # elif acceptable_price < int(best_ask):
                #     allocs = [p for p in range(int(best_bid), int(acceptable_price)+1)]
                #     amt = (pos_limits[product] - cur_pos)//len(allocs)
                #     for p in allocs:
                #         orders.append(Order(product, p, amt))

                # if mid_price > acceptable_price:
                #     orders.append(Order(product, best_bid, -10))
                #     orders.append(Order(product, best_bid+1, -10))
                    #alloc = (pos_limits[product] - cur_pos)//2
                    #orders.append(Order(product, best_bid, alloc))
                    #orders.append(Order(product, best_bid+1, alloc))
                    #orders.append(Order(product, best_bid+3, alloc))
                    #orders.append(Order(product, best_bid+4, alloc))
                    #orders.append((Order(product, int(mid_price), 10)))
                    #orders.append((Order(product, int(mid_price)-1, 10)))

            result[product] = orders

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData






# if __name__ == "__main__":
#     import time
#
#     start_time = time.time()
#
#     pos_limits = {'AMETHYSTS': 20, 'STARFRUIT': 20}
#     sell_orders = {10006:-4, 10005:-29, 10004:-2}
#     buy_orders = {10002:1, 9996:2, 9995:29}
#
#     result = {}
#     for product in pos_limits:
#
#         acceptable_price = 0
#         total = 0
#
#         if len(sell_orders) != 0 and len(buy_orders) != 0:
#             total = 0
#             for k in sell_orders:
#                 acceptable_price += k*(-sell_orders[k])
#                 total -= sell_orders[k]
#             for k in buy_orders:
#                 acceptable_price += k*buy_orders[k]
#                 total += buy_orders[k]
#
#             acceptable_price /= total
#
#         cur_pos = 20
#
#         orders = []
#
#         if len(sell_orders) != 0:
#             best_ask, best_ask_amount = list(sell_orders.items())[-1]
#             if int(best_ask) < acceptable_price:
#                 amt = pos_limits[product] - cur_pos
#                 print("BUY", str(amt) + "x", best_ask)
#                 orders.append(Order(product, best_ask, amt))
#
#         if len(buy_orders) != 0:
#             best_bid, best_bid_amount = list(buy_orders.items())[0]
#             if int(best_bid) > acceptable_price:
#                 amt = -pos_limits[product] - cur_pos
#                 print("SELL", str(amt) + "x", best_bid)
#                 orders.append(Order(product, best_bid, amt))
#
#         result[product] = orders
#
#     print(time.time() - start_time)
