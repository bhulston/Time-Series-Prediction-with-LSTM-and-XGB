
#Python C:\Users\brand\OneDrive\Documents\Python\PyOrderBook\Websocket_self\main.py
import pandas as pd
import numpy as np
import datetime as dt
import plotly as pt
import matplotlib.pyplot as plt
import seaborn as sns
import os

#https://www.geeksforgeeks.org/python-call-function-from-another-file/


def clean_data(data_buys, data_sells):

    buys_df = pd.DataFrame(data_buys)
    buys_df = buys_df.rename(columns = {0: 'price', 1: 'size', 2: 'timestamp'})
    buys_df = buys_df.sort_values(by = ['price'], ascending = False, axis = 0, ignore_index = True)
    buys_df['Liquidity'] = buys_df['price'] * buys_df['size']
    buys_df['Liquidity_running'] = buys_df['Liquidity'].cumsum(axis = 0)

    print(buys_df)


    sells_df = pd.DataFrame(data_sells)
    sells_df = sells_df.rename(columns = {0: 'price', 1: 'size', 2: 'timestamp'})
    sells_df['Liquidity'] = sells_df['price'] * sells_df['size']
    sells_df['Liquidity_running'] = sells_df['Liquidity'].cumsum(axis = 0)

    print(sells_df)

    return buys_df, sells_df

#TEST
#print(sum(buys_df['Liquidity']), sum(sells_df['Liquidity']))

# ECDF plot using
def ECDF_plot(timestamp, buys_df, sells_df):

    fig, ax = plt.subplots()
    ax.set_title('XBT-USD' + ' ' + 'order book on' + ' '+ str('BitMex') +', '+ 'on 2022-11-05 at 1:40pm')#timestamp variable here

# bid side
    sns.ecdfplot(x="price", weights="size", stat="count", complementary=True,
            data=buys_df, ax=ax, color='g')
# ask side
    sns.ecdfplot(x="price", weights="size", stat="count",
            data=sells_df.loc[sells_df['Liquidity_running'] < sum(sells_df['Liquidity']) / 10], ax=ax, color='r')

    ax.set_xlabel("Price")
    ax.set_ylabel("Amount")

    fig.savefig("C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/Data/MarketDepth")

def price_liq_plot(dic):
    COLOR_LIQ = "#69b3a2"
    COLOR_PRICE = "#3399e6"

    timestamps = dic['timestamp']
    liq_002 = dic['ask_liq_.002']
    prices = dic['mid_price']

    fig, ax1 = plt.subplots(figsize=(50, 30))

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()



    fig, ax1 = plt.subplots(figsize=(50, 30))
    ax2 = ax1.twinx()

    ax1.plot(timestamps, liq_002, color = COLOR_LIQ, lw=10)
    ax2.plot(timestamps, prices, color = COLOR_PRICE, lw=14);


    ax1.set_xlabel("Timestamp", fontsize = 50)
    ax1.set_ylabel("Liquidity", color=COLOR_LIQ, fontsize=50)
    ax1.tick_params(axis="y", labelcolor=COLOR_LIQ, labelsize = 45)

    ax2.set_ylabel("Mid Price ($)", color=COLOR_PRICE, fontsize=50)
    ax2.tick_params(axis="y", labelcolor=COLOR_PRICE, labelsize = 45)

    fig.suptitle("Price vs Liquidity", fontsize=70)
    fig.autofmt_xdate()

    fig.savefig("C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/Data/price_liq")

###################################################
#### Columns we need to add to create features ####
###################################################

#columns/parameters I need to add:
    #liquidity at 1%, 3%, 5%, 10%, 20%
        #ask liquidity
        #bid liquidity
    #Midprice, bid and ask
    #

def get_prices(buys, sells):

    buys_levels = buys['price']
    bid_price = buys_levels[0]

    sells_levels = sells['price']
    ask_price = sells_levels[0]

    mid_price = (bid_price + ask_price) / 2

    return [bid_price, ask_price, mid_price]

def get_liquidity(buys, sells):

    buys_levels = buys['price']
    bid_price = buys_levels[0]

    sells_levels = sells['price']
    ask_price = sells_levels[0]

    #price level denominators - #1%, 3% etc
    #prove business reason why we should think about this
    levels = [500, 333, 100, 33.33, 20]
    bid_liq_levels = []
    ask_liq_levels = []

    for level in levels:
        bid = bid_price - bid_price/level
        bid_price = np.max(buys['price'].iloc[(buys['price'] - bid).abs().argsort()[:2]])
        bid_liq = buys.loc[buys['price'] == bid_price, 'Liquidity_running']
        bid_liq_levels.append(bid_liq.values[0])

        ask = ask_price + ask_price/level
        ask_price = np.max( sells['price'].iloc[(ask - sells['price']).abs().argsort()[:2]] )
        ask_liq = sells.loc[sells['price'] == ask_price, 'Liquidity_running']
        ask_liq_levels.append(ask_liq.values[0])

    return (bid_liq_levels, ask_liq_levels)

#need a for loop for the different buy and sell csvs that goes through them and constructs my new dataframe

directory = r'C:\Users\brand\OneDrive\Documents\Python\PyOrderBook\Data'
file_dict = {"timestamp":[], "buys":[], "sells":[]}

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    if os.path.isfile(f):

        filestring = filename.split(".")
        filestring = filestring[0]
        datetime = filestring[:19]
        print("Loading file from {}... Filename is {}".format(datetime, filename))

        if datetime not in file_dict['timestamp']:
            file_dict['timestamp'].append(datetime)

        if 'buys' in filename:
            file_dict['buys'].append(filename)
        else:
            file_dict['sells'].append(filename)

    else:
        print("Skipping non-file...")

print(file_dict)

LOB_dict = {"timestamp":[], "mid_price":[], "bid_price":[], "ask_price":[],
            "bid_liq_.002":[], "bid_liq_.0033":[], "bid_liq_.01":[], "bid_liq_.03":[], "bid_liq_.05":[],
            "ask_liq_.002":[], "ask_liq_.0033":[], "ask_liq_.01":[], "ask_liq_.03":[], "ask_liq_.05":[], }

LOB_dict = pd.DataFrame(LOB_dict)
