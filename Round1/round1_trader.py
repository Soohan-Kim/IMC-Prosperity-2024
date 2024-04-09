from round1_datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
#import logging


class Trader:

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        print("Market trades:" + str(state.market_trades))

        pos_limits = {'AMETHYSTS': 20, 'STARFRUIT': 20}

        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            print("Order book sell:" + str(order_depth.sell_orders))
            print("Order book buy:" + str(order_depth.buy_orders))

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

            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
                len(order_depth.sell_orders)))

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    # print("BUY", str(-best_ask_amount) + "x", best_ask)
                    # orders.append(Order(product, best_ask, -best_ask_amount))
                    amt = pos_limits[product] - cur_pos
                    print("BUY", str(amt) + "x", best_ask)
                    orders.append(Order(product, best_ask, amt))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    # print("SELL", str(best_bid_amount) + "x", best_bid)
                    # orders.append(Order(product, best_bid, -best_bid_amount))
                    amt = -pos_limits[product] - cur_pos
                    print("SELL", str(amt) + "x", best_bid)
                    orders.append(Order(product, best_bid, amt))

            result[product] = orders

        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
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
