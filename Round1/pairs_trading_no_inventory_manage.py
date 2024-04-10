from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    map_symbol = {
        'ame' : 'AMETHYSTS',
        'str' : 'STARFRUIT',
    }

    inventory = {}
    maxsize = {}
    # data = {}
    
    weighted_price = {}
    mid_price = {}

    def __construct_data(self, symbols):
        for symbol in symbols:
            self.maxsize[symbol] = {'bid' : 0, 'ask' : 0}
            
    def track_order(self, state):
        if 'data' not in self.__dict__:
            self.__construct_data(state.order_depth.keys())
        
        self.inventory = state.position
        
        for symbol in state.order_depth.keys():
            order_depth = state.order_depth[symbol]
            inventory = self.inventory.get(symbol, 0)
            self.maxsize[symbol]['bid'] = sum(order_depth.buy_orders.values()) - min(0, inventory)
            self.maxsize[symbol]['ask'] = -sum(order_depth.sell_orders.values()) + max(0, inventory)
        pass
    
    def calc_weightedprice(self, orders, counter = 2):
        sums = 0
        total_amount = 0
        max_amount = 0
        cnt = counter
        for price, amount in orders.items():
            if cnt == 0:
                break
            sums += int(price) * amount
            total_amount += amount
            max_amount = max(max_amount, amount)
            cnt -=1

        if cnt == counter:
            return [0, 0, 0, 0]

        return [sums/total_amount, total_amount, max_amount, total_amount/(counter - cnt)]
    
    def track_signals(self, state):
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            
            self.weighted_price[product] = {}
            self.weighted_price[product]['bid'] = self.calc_weightedprice(order_depth.buy_orders, product)
            self.weighted_price[product]['ask'] = self.calc_weightedprice(order_depth.sell_orders, product)
            self.mid_price[product] = (self.weighted_price[product]['bid'][0] + self.weighted_price[product]['ask'][0]) / 2
        
        pass
    
    def pairs_trading(self, strategy):
        res = {}
        if strategy == 'long_ame':
            amount = int(min(
                self.weighted_price[self.map_symbol['ame']]['ask'][-1],
                self.weighted_price[self.map_symbol['str']]['bid'][-1]/2
            ))
            
            res[self.map_symbol['ame']] = [
                Order(
                    self.map_symbol['ame'], 
                    self.weighted_price[self.map_symbol['ame']]['ask'][-1], 
                    amount
                )
            ]
            res[self.map_symbol['str']] = [
                Order(
                    self.map_symbol['str'], 
                    self.weighted_price[self.map_symbol['str']]['bid'][-1], 
                    -amount * 2
                )
            ]
        else:
            amount = int(min(
                self.weighted_price[self.map_symbol['ame']]['bid'][-1],
                self.weighted_price[self.map_symbol['str']]['ask'][-1]/2
            ))
            
            res[self.map_symbol['ame']] = [
                Order(
                    self.map_symbol['ame'], 
                    self.weighted_price[self.map_symbol['ame']]['bid'][-1], 
                    -amount
                )
            ]
            res[self.map_symbol['str']] = [
                Order(
                    self.map_symbol['str'], 
                    self.weighted_price[self.map_symbol['str']]['ask'][-1], 
                    amount * 2
                )
            ]
        return res
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        self.track_order(state)
        self.track_signals(state)
            
        if 2* self.mid_price[self.map_symbol['ame']] > self.mid_price[self.map_symbol['str']]:
            strategy = 'long_ame'
        else:
            strategy = 'short_ame'

        print("Strategy: " + strategy)
        
        result = self.pairs_trading(strategy)
       
        traderData = "SAMPLE" 
        conversions = None
        return result, conversions, traderData

