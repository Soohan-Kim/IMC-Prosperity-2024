import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get price data
prices_round_1_day_0 = pd.read_csv('../data/prices/prices_round_1_day_0.csv', sep=';')
prices_round_1_day_1 = pd.read_csv('../data/prices/prices_round_1_day_-1.csv', sep=';')
prices_round_1_day_2 = pd.read_csv('../data/prices/prices_round_1_day_-2.csv', sep=';')

# Concat prices
prices = pd.concat([prices_round_1_day_2, prices_round_1_day_1, prices_round_1_day_0])

# Filter price data
prices_STARFRUIT = prices[prices['product']=='STARFRUIT'].reset_index(drop=True)
prices_AMETHYSTS = prices[prices['product']=='AMETHYSTS'].reset_index(drop=True)

# Get trades data
trades_round_1_day_0_nn = pd.read_csv('../data/trades/trades_round_1_day_0_nn.csv', sep=';')
trades_round_1_day_0_nn.insert(0, 'day', 0)

trades_round_1_day_1_nn = pd.read_csv('../data/trades/trades_round_1_day_-1_nn.csv', sep=';')
trades_round_1_day_1_nn.insert(0, 'day', -1)

trades_round_1_day_2_nn = pd.read_csv('../data/trades/trades_round_1_day_-2_nn.csv', sep=';')
trades_round_1_day_2_nn.insert(0, 'day', -2)

# Concat trades data
trades = pd.concat([trades_round_1_day_2_nn, trades_round_1_day_1_nn, trades_round_1_day_0_nn])

# Filter trades data
trades_STARFRUIT = trades[trades['symbol']=='STARFRUIT']
trades_AMETHYSTS = trades[trades['symbol']=='AMETHYSTS']

if __name__ == "__main__":
    import os

    path_with_tilde = '~/my_file.txt'
    expanded_path = os.path.expanduser(path_with_tilde)
    print(
        expanded_path)  # This will print the absolute path to the current user's home directory followed by '/my_file.txt'

