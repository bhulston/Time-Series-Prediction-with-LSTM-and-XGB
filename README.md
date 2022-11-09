# High-Volume-Liquidations
Build an algorithm that can predict LOB future states, and find the optimal strategy to liquidate over a period of time

# Project Plan
1. Take snap shots of the order book state at different times, both L2 (all open orders) and L1 (bid, mid, ask price) data. (DONE)
2. Process data and perform feature extraction and engineering to create features that can be used properly by the model. (DONE)
3. Design a series of predictive models including LSTM Neural Networks and Gradient Tree Boosting to predict future order book states (In-Progress)
4. Compare model results and find the best model (or combination) to predict order book states (Not Started)
5. Build an agent that uses future predicted order states and finds the optimal strategy that minimizes loss when liquidating over a given period of time (Not Started)



## Data Collection
When trying to collect relevant data on Order Book L2s, many exchange APIs do not retain historical data (only present data). So the best option is to connect to a websocket (in my case, the BitMex one) and stream data over a set period of time. Every minute, I collect data on the current order book and save the data according to each "batch". Here is also where I need to split the data into a training, validation, and test set.

## Feature Engineering
This took a lot of research but it seems like the most relevant features people use for order book modelling are log_return_ask, log_volume_ask, log_return_bid, and log_volume_bid. I also am adding features that take into account the liquidity and price depth near the mid-price. 

## Models
The first model I decided to use was an LSTM Neural Network, which excels at at learning long sequences of observations, thus making it very good for time-series analysis.

The second model/algorithm I am using is Gradient Tree Boosting which performs gradient-descent minimization on a provided loss function. This treats the model as a regression problem and we use trees to get our predictions, which is another good method for time-series data.

## Things/difficulties to note

This article highlights how predicting prices is difficult, as prices seem to change by a very small amount if not 0, with noise.
https://hackernoon.com/dont-be-fooled-deceptive-cryptocurrency-price-predictions-using-deep-learning-bf27e4837151 

Depending on the volume of data and processing time, we might need an auto encoder to perform this analysis in a timely manner since this issue requires relatively quick response times. 

Due to the difficulty of predicting L2 order books (all orders open), we will use the L2 information and L1 information to predict the L1 results (mid, bid, ask prices)

