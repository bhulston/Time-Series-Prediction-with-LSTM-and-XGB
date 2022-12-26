#Python C:\Users\brand\OneDrive\Documents\Python\PyOrderBook\Websocket_self\main.py
import pandas as pd
import numpy as np
import datetime as dt
import plotly as pt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

import sys

sys.path.insert(1, 'C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/config')
#https://www.geeksforgeeks.org/python-call-function-from-another-file/
import paramaters as pm
from paramaters import *


def clean_data(data_buys, data_sells):

    buys_df = pd.DataFrame(data_buys)
    buys_df = buys_df.rename(columns = {0: 'price', 1: 'size', 2: 'timestamp'})
    buys_df = buys_df.sort_values(by = ['price'], ascending = False, axis = 0, ignore_index = True)
    buys_df['Liquidity'] = buys_df['price'] * buys_df['size']
    buys_df['Liquidity_running'] = buys_df['Liquidity'].cumsum(axis = 0)

    # print(buys_df)

    sells_df = pd.DataFrame(data_sells)
    sells_df = sells_df.rename(columns = {0: 'price', 1: 'size', 2: 'timestamp'})
    sells_df['Liquidity'] = sells_df['price'] * sells_df['size']
    sells_df['Liquidity_running'] = sells_df['Liquidity'].cumsum(axis = 0)

    # print(sells_df)

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

    fig.savefig("C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/images/MarketDepth")

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

    fig.savefig("C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/images/price_liq")

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


## Load the Data ##

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

# print(file_dict)

#Initialize dictionary with the given columns
LOB_dict = {"timestamp":[], "mid_price":[], "bid_price":[], "ask_price":[],
            "bid_liq_.002":[], "bid_liq_.0033":[], "bid_liq_.01":[], "bid_liq_.03":[], "bid_liq_.05":[],
            "ask_liq_.002":[], "ask_liq_.0033":[], "ask_liq_.01":[], "ask_liq_.03":[], "ask_liq_.05":[], }

LOB_dict = pd.DataFrame(LOB_dict)
#####################################################
######## Feature Engineering for ML Models ##########
#####################################################

#we should really do the EDA for this before

#Important notes

#so there are going to be two main kinds of Features
    #1. Features pertaining to the market Depth
    #2. Features pertaining to bid ask and spread

#Note: Due to the fact that many exchanges have lots of orders coming in that get cancelled
    #about 90%, usually far from the mid price. These might come from algorithm trading. So we can focus on prices near mid

#1. Market depth

#liquidity within 2% - 20% of the price at multiple levels
for i, timestamp in enumerate(file_dict['timestamp']):

    data_buys = pd.read_csv(r'{}\{}buys.csv'.format(directory, timestamp), header = None)
    data_sells = pd.read_csv(r'{}\{}sells.csv'.format(directory, timestamp), header = None)

    buys_df, sells_df = clean_data(data_buys, data_sells)

    prices = get_prices(buys_df, sells_df)
    bid_liq, ask_liq = get_liquidity(buys_df, sells_df)

    row = [timestamp] + prices + bid_liq + ask_liq

    # print(row)

    LOB_dict.loc[i] = row

#price_liq_plot(LOB_dict)

##log liq ask change = ln(((ask liq_t - ask liq_t-1) / askliq_t)  + 1)

#LOB_dict["log_bid_liq_.01"] = np.log(((LOB_dict["log_bid_liq_.01"] -

#log liq bid change = ln(((bid liq_t - bid_liq_t-1 ) / bidliq_t) + 1)


#liq change % - log?



#2. Bid ask and spread

#mid price movement smoothing at multiple ks and alphas
mid_smoothing = LOB_dict['mid_price']

params = pm.Params()

k = params.config['Smoothing']['k']
alpha = params.config['Smoothing']['alpha']
smooth_pos = []
smooth_neg = []
dir_t_list = []

for i in range(len(mid_smoothing)):
    if i < k:
        mt = 'null'
        mt_ = 'null'
    else:
        mt = np.sum(mid_smoothing[i:i+k])/k
        mt_ = np.sum(mid_smoothing[i-k:i])/k
    smooth_pos.append(mt)
    smooth_neg.append(mt_)
    # print(smooth_pos, smooth_neg)


for i in range(len(smooth_pos)):
    if i < k or i > len(smooth_pos) - (k + 1):
        dir_t = 0
    else:
        dir_t = (mid_smoothing[i] - smooth_neg[i]) / smooth_neg[i]
    dir_t_list.append(dir_t)

dir_signal = []
for i in dir_t_list:
    if i > alpha:
        dir_signal.append(1)
    elif i < -alpha:
        dir_signal.append(-1)
    else:
        dir_signal.append(0)

LOB_dict['dir'] = dir_signal
# print(dir_signal)
# print("HI", len(dir_t_list))

#we can show multiple k horizons for smoothing and show how it can be a good indicator of general direction movement
#and print multiple distribution graphs at different ks and alphas to find the best one
# plt.plot(LOB_dict['timestamp'][:(88-k)], dir_t_list)
# plt.show()
# fig.savefig("C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/images/dir_signal")


#bid-ask-spread
LOB_dict['spread'] = (LOB_dict['ask_price'] - LOB_dict['bid_price'])

##log return ask = ln(ask_t / ask_t-n) / n x 100%
log_ret_ask = []
log_ret_bid = []
for i in range(len(LOB_dict['ask_price'])):
    if i > 0 and i < (len(LOB_dict['ask_price']) - 1):
        log_ret_ask.append(np.log(LOB_dict['ask_price'][i] / LOB_dict['ask_price'][i-1]))

        log_ret_bid.append(np.log(LOB_dict['bid_price'][i] / LOB_dict['bid_price'][i-1]))
    else:
        log_ret_ask.append(0)
        log_ret_bid.append(0)
LOB_dict['log_ret_ask'] = log_ret_ask
LOB_dict['log_ret_bid'] = log_ret_bid
# print(LOB_dict['log_ret_ask'])

# plt.plot(LOB_dict['timestamp'], LOB_dict['log_ret_ask'])
# plt.show()

# print(LOB_dict)

#maybe the last few log_returns as well?


#moving averages
#SMA = (MP_t-1 + MP_t-2 ... MP_t-n)
    #at row 2, just use the previous
    #at row 4, use previous 3 only
def simple_moving_average(df, k):
  n = len(df)
  if k > n:
    return "Invalid input: k must be less than or equal to the number of rows in the dataframe"
  sma_list = []
  for i in range(n):
    if (i+1) < k:
      sma_list.append(0)
      continue
    sum = 0
    for j in range((i+1) - k, (i+1)):
      sum += df.loc[j, 'mid_price']
    sma = sum / k
    sma_list.append(sma)
  return sma_list

sma_list = simple_moving_average(LOB_dict, 5)
# print(sma_list)

# print(list)
def exponential_moving_average(df, k):
  n = len(df)
  ema_multiplier = (2 / (k+1))
  if k > n:
    return "Invalid input: k must be less than or equal to the number of rows in the dataframe"
  ema_list = []
  for i in range(n):
      if (i+1) < k:
          ema_list.append(0)
          continue
      elif (i+1) == k:
          ema_list.append(sum(df['mid_price'][0:k]) / k)
      else:
          ema = (df['mid_price'][i] * ema_multiplier) + (ema_list[-1] * (1 - ema_multiplier))
          ema_list.append(ema)
  return ema_list

ema_list = exponential_moving_average(LOB_dict, 5)
# print(ema_list)

LOB_dict['sma'] = sma_list
LOB_dict['ema'] = ema_list

print(LOB_dict)


LOB_dict.to_csv('C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/EDA/LOB_data.csv', index = False)
# with open('C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/EDA/LOB_data.csv', 'w') as file:
#     writer = csv.writer(file)
#     for key, value in LOB_dict.items():
#         writer.writerow([key, value])
#
# file.close()
