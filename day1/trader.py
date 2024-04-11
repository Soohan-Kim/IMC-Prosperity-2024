from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import json
import pandas as pd
import numpy as np
import statistics
import math

class Trader:    

    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
    
    starfruit_cache = []
    STARFRUIT_DIM = 4
    
    def get_acceptable_price(self, product, order_depth):
        if product == "AMETHYSTS":
            return 10000
        elif product == "STARFRUIT":
            coef = [2.367111652110907, 0.34173523, 0.26105364, 0.20777259, 0.1889694]
            predict = coef[0]
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask, _ = list(order_depth.sell_orders.items())[0]
                best_bid, _ = list(order_depth.buy_orders.items())[0]
                mid_price = (best_bid + best_ask)/2
            try:
                for i in range(1, 5):
                    predict += (coef[i] * self.starfruit_cache[-i])
            except:
                pass
            finally:
                self.starfruit_cache.append(mid_price)
                return predict
            

    def get_lim_position(self, product, state: TradingState):
        curr_position = state.position[product] if product in state.position else 0 
        return self.POSITION_LIMIT[product]-curr_position, -self.POSITION_LIMIT[product]-curr_position

    def order_bid_maker(self, product, orders, bid_price, bid_amount, bid_lim):
        print("BUY", str(min(bid_amount, bid_lim)) + "x", bid_price)
        orders.append(Order(product, bid_price, min(bid_amount, bid_lim)))
        
    def order_ask_maker(self, product, orders, ask_price, ask_amount, ask_lim):
        print("SELL", str(-max(-ask_amount, ask_lim)) + "x", ask_price)
        orders.append(Order(product, ask_price, max(-ask_amount, ask_lim)))
    
    def order_bid_taker(self, product, orders, ask_price, ask_amount, bid_lim):
        print("BUY", str(min(-ask_amount, bid_lim)) + "x", ask_price)
        orders.append(Order(product, ask_price, min(-ask_amount, bid_lim)))
                
    def order_ask_taker(self, product, orders, bid_price, bid_amount, ask_lim):
        print("SELL", str(-max(-bid_amount, ask_lim)) + "x", bid_price)  
        orders.append(Order(product, bid_price, max(-bid_amount, ask_lim)))

    def make_orders(self, product, state):
        orders: List[Order] = []
        order_depth: OrderDepth = state.order_depths[product]
        bid_lim, ask_lim = self.get_lim_position(product, state)
        acceptable_price = self.get_acceptable_price(product, order_depth)  # Participant should calculate this value
        
        print("Acceptable price : " + str(acceptable_price))
        print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
        
        if product == "AMETHYSTS":    
            for ask_info in list(order_depth.sell_orders.items()):
                ask_price, ask_amount = ask_info
                if int(ask_price) < acceptable_price:
                    self.order_bid_taker(product, orders, ask_price, ask_amount, bid_lim)
                    bid_lim -= min(-ask_amount, bid_lim)
        
            for bid_info in list(order_depth.buy_orders.items()):                
                bid_price, bid_amount = bid_info
                if int(bid_price) > acceptable_price: 
                    self.order_ask_taker(product, orders, bid_price, bid_amount, ask_lim)            
                    ask_lim -= max(-bid_amount, ask_lim)  
        
        elif product == "STARFRUIT":
            if len(self.starfruit_cache) < self.STARFRUIT_DIM:
                return orders

            for ask_info in list(order_depth.sell_orders.items()):
                ask_price, ask_amount = ask_info
                if int(ask_price) < acceptable_price:
                    self.order_bid_taker(product, orders, ask_price, ask_amount, bid_lim)
                    bid_lim -= min(-ask_amount, bid_lim)
    
            for bid_info in list(order_depth.buy_orders.items()):                
                bid_price, bid_amount = bid_info
                if int(bid_price) > acceptable_price: 
                    self.order_ask_taker(product, orders, bid_price, bid_amount, ask_lim)            
                    ask_lim -= max(-bid_amount, ask_lim)  
            
        return orders

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        trader_data = ""
        for product in state.order_depths:
            
            orders = self.make_orders(product, state)

            result[product] = orders
    
        traderData = "hwjang" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 0
        return result, conversions, traderData