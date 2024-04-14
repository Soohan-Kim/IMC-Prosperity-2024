from samp_datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List
import json
import jsonpickle
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
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
    
    #starfruit_cache = {'mid_price':[]}
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
                    predict += (coef[i] * self.starfruit_cache['mid_price'][-i])
            except:
                pass
            finally:
                self.starfruit_cache['mid_price'].append(mid_price)
                return predict

    def get_lim_position(self, product, state: TradingState):
        curr_position = state.position[product] if product in state.position else 0 
        return self.POSITION_LIMIT[product]-curr_position, -self.POSITION_LIMIT[product]-curr_position

    def order_bid_maker(self, product, orders, bid_price, bid_amount, bid_lim):
        logger.print("BUY", str(min(bid_amount, bid_lim)) + "x", bid_price)
        orders.append(Order(product, bid_price, min(bid_amount, bid_lim)))
        
    def order_ask_maker(self, product, orders, ask_price, ask_amount, ask_lim):
        logger.print("SELL", str(-max(-ask_amount, ask_lim)) + "x", ask_price)
        orders.append(Order(product, ask_price, max(-ask_amount, ask_lim)))
    
    def order_bid_taker(self, product, orders, ask_price, ask_amount, bid_lim):
        logger.print("BUY", str(min(-ask_amount, bid_lim)) + "x", ask_price)
        orders.append(Order(product, ask_price, min(-ask_amount, bid_lim)))
                
    def order_ask_taker(self, product, orders, bid_price, bid_amount, ask_lim):
        logger.print("SELL", str(-max(-bid_amount, ask_lim)) + "x", bid_price)  
        orders.append(Order(product, bid_price, max(-bid_amount, ask_lim)))

    def make_orders(self, product, state):
        orders: List[Order] = []
        order_depth: OrderDepth = state.order_depths[product]
        bid_lim, ask_lim = self.get_lim_position(product, state)
        acceptable_price = self.get_acceptable_price(product, order_depth)  # Participant should calculate this value
        
        logger.print("Acceptable price : " + str(acceptable_price))
        logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
        
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
            if len(self.starfruit_cache['mid_price']) < self.STARFRUIT_DIM:
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
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        result = {}

        if state.traderData == '':
            self.starfruit_cache = {'mid_price': []}
        else:
            self.starfruit_cache = jsonpickle.decode(state.traderData)

        for product in state.order_depths:
            
            orders = self.make_orders(product, state)

            result[product] = orders
    
        #traderData = "hwjang" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 0

        traderData = jsonpickle.encode(self.starfruit_cache)
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
