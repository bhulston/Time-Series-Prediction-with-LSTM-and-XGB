# Short-Frame-Price-Prediction
Build an algorithm that can predict LOB future states, in order to find optimal opportunities for liquidation

While Ripple uses market makers for liquidations, I got the idea that in many cases, businesses or individuals might need to liquidate large amounts of $$ on short time frames. For any organization that has to perform high-volume purchases or liquidations on the market, this could be a helpful model to reduce slippage and other costs.

# Quick Guide
I used my own computer terminal to stream the websocket and to transform the data to create a new dataset. That dataset is used in the visual and modeling notebook
* The websocket folder contains the classes and websocket I use to collect data
* The data and feature processing folder contains the transformations and aggregations I did to create my dataset for visuals and models
* The PyOrderBook_visuals notebook contains all my code for the visuals about the data itself
* The PyOrderBook__modeling notebook contains all my code for the prediction models and the associated visuals as well

# Project Plan
1. Take snap shots of the order book state at different times, both L2 (all open orders) and L1 (bid, mid, ask price) data
2. Process data and perform feature extraction and engineering to create features that can be used properly by the model
3. Design a series of predictive models including LSTM Neural Networks and Gradient Tree Boosting to predict future order book states 
4. Compare model results and find the best model to predict order book states 

## Data Collection
When trying to collect relevant data on Order Book L2s, many exchange APIs do not retain historical data (only present data). So the best option is to connect to a websocket (in my case, the BitMex one) and stream data over a set period of time. Every 30 seconds, I collect data on the current order book and save the data according to each "batch".

Some of the things we collected are: Order quantities at different bid and ask prices. Mid price. Recent purchase price.
* From these we can calculate all our other variables, such as liquidity, moving averages, etc.

Here is a graph of the mid-price data that we collected over about 24 hours:
<img width="480" alt="image" src="https://user-images.githubusercontent.com/79114425/210905065-c9dd4492-2a99-4ae7-a38d-563bf76398fe.png">

When it comes to prices on order books, liquidity definitely plays a larger role than many people seem to think. For price movements to occur, you have to "climb" over the steps on either side.
Here is an example of the market depth(liquidity) near the mid-price:

<img width="476" alt="image" src="https://user-images.githubusercontent.com/79114425/210905099-4d98d045-e9c2-4736-ac9d-4ba04744b3f0.png">


## Feature Engineering
I engineered several features based off of the data we collected. The formulas and algorithms can be found in the feature engineering doc:
* Logarithmic transformation of returns
* Liquidity depth at different levels to the mid-price
* Smoothed data and directional signals
* EMA and SMA

Visualization of logarithmic transformation of returns:

<img width="580" alt="image" src="https://user-images.githubusercontent.com/79114425/208793698-2324802d-453a-4c5e-a244-955cd417f7f6.png">

An example of why this is important is because when data is noisy, it can cause issues in the models. To combat this I looked at the effect of adding a directional smoothing signal. +1 for upward trends, 0 for stationary, and -1 for downtrends.

Non-smoothed data, i.e. a +1 when price moved up (Red for a downtrend, blank for no movement, and green for an uptrend):

<img width="479" alt="image" src="https://user-images.githubusercontent.com/79114425/210905840-ee214bd0-b3d0-48de-b6c8-a9e90d3c88cd.png">

Smoothed data, i.e. setting a threshold value for price movements, as well as using moving averages for the signal values:

<img width="476" alt="image" src="https://user-images.githubusercontent.com/79114425/210905941-38d47593-17b1-493b-bbf0-faadbe0fe9cd.png">

* As you can see, by smoothing the data we are able to reduce the "noise" in this feature and get a better representation of general directional trends

## Baseline Models
First, I built some baseline models using ARIMA and Exponential Smoothing. 
* We used double exponential smoothing to capture trend movements, triple xponential smoothing for taking into account seasonality, and an ARIMA model (Autoregressive Integrated Moving Average)
  * Alpha in exponential smoothing models indicate to us how much we want to smooth the data. Higher alphas indicate we want to more heavily weight recent points.
  * These are the most common statistical methods for time-series predictions.
* All of these baseline models are univarate models that only take into account the mid-price
* Note that the triple exponential smoothing model is probably a bad choice for this task
   * Because we have data that is recorded on such a short time frame (1 day), it's unlikely to have any meaningful seasonality trends

![image](https://user-images.githubusercontent.com/79114425/210890640-2a1affa7-14c2-49d0-93ac-87c3a8a8141f.png)
* While mse is a commmon measurement for success, I think in the case of this project, plotting the values gives us a much better idea of how models are performing relatively. 
* The goal of this project is NOT to get as close as possible at each point, but rather to capture general trends (up, down, or stationary)

# XGB and LSTM Models
After building the baselines, I built two models that take multivariate inputs to see if we can improve, a LSTM and XGBoosted Trees 
* For the LSTM I used a recursive approach, meaning that we make predictions of all variables at each step, then use those new predictions as inputs for the next step
* For the XGBoost model, I used a direct approach, which trains a different model for each future timestep. AKA, for 10 predictions, we train 10 different models, each trained on a different time frame/time step.
* More on these multi-step approaches here: 

## XGBoost
XGBoost is short for "Extreme Gradient Boosting", and uses an ensemble of gradient boosted trees to make regression/classification predictions.
* The main thing this requires is a lot of hyper parameter tuning! With the right tuning, XGBoost models seem to consistently outperform many other models, including ones specifically built for time-series
* Some resources I used for XGBoost:
  * [XGBoost](https://arxiv.org/abs/1603.02754) - arxiv
  * [Time Series Prediction Models](https://arxiv.org/pdf/2103.14250.pdf) -arxiv
  * [Fine tuning XGBoost](https://medium.com/towards-data-science/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e) - medium

Direct Approach explained:

<img width = "400" alt="image" src="https://user-images.githubusercontent.com/79114425/210893972-caa8babc-faa6-4bea-b652-c4ca3483d6fa.png">

Hyperparameters:

<img width="200" alt="image" src="https://user-images.githubusercontent.com/79114425/210907642-69d7499b-7682-4127-8619-0d4f06d10e21.png">

* The most important hyperparameters I focused on when tuning were:
  * **n_estimators**: In a dataset with not that much data, we had to raise this from the default of 100 to get proper results
  * **max_depth**: The max depth of the trees. The whole point of an XGBoost model is to leverage an ensemble of weak learners. Thus, making sure this value is not too high is crucial for good results. 
  * **learning_rate**: How "fast" our model learns, or really the steps that it takes when using gradient descent for optimization. Many models out there have very small learning rates, but due to the stochastic nature of this project, a higher learning rate of 0.1 is more appropriate to avoid getting stuck in local minimas
  * **colsample_bytree**: This value indicates the proportions of columns to be used in each tree that is built. We reduce this because we do have a lot of features (about 20)

Here we can see the performance of the XGBoost model in comparison to the baseline models we created.

![image](https://user-images.githubusercontent.com/79114425/210891496-e57eca25-8ea7-4965-9247-6bb4bce9b37b.png)

In comparison to the baseline models, the XGBoost model returned a reasonable improvement in results

<img width="269" alt="image" src="https://user-images.githubusercontent.com/79114425/210908792-2883a052-7d91-48c8-96f8-2a11e7e79bda.png">

Because the XGBoost model performed pretty well, I wanted to see how it would perform when we trained it on 40 timesteps:

![image](https://user-images.githubusercontent.com/79114425/210891184-6421d4a7-36b5-4353-83ee-d6a55bfb72a8.png)

* As you can see, the model is performing surprisingly well in capturing the trends of future movements

## LSTMs
With a little bit of research, you will find that LSTMs seem to perform pretty poorly on real financial data. The reason for this is that they are extremely prone to over-fitting, and on top of that, they perform poorly when working with auto-regression problems.
* LSTM window sizes don't seem to make a big difference (or can perform worse), despite being one of the main features when used on financial data
* An LSTM model might be better suited as part of a language processing model

Here we can see the LSTM model compared to the training data. Values as inputs for the model are standardized, but I scaled them back up for these representations:

<img width="560" alt="image" src="https://user-images.githubusercontent.com/79114425/208794211-3bc659a8-6447-4674-9a32-304b8a3a298f.png">

Here is the model on the testing data:

<img width="560" alt="image" src="https://user-images.githubusercontent.com/79114425/208794255-59518d74-a9fa-4008-b630-d7160809020d.png">
* Note that while the values seem a little far apart, the actual numerical difference is really small because this is a very zoomed in timeframe

While it seemed to perform really well on the training data, we can see our suspicions are confirmed that the predictions perform poorly. This is probably, in part, due to us using a recursive approach, rather than a direct one. 


## Things/difficulties to note

This article highlights how predicting prices is difficult, as prices seem to change by a very small amount if not 0, with noise.
https://hackernoon.com/dont-be-fooled-deceptive-cryptocurrency-price-predictions-using-deep-learning-bf27e4837151 

Depending on the volume of data and processing time, we might need an auto encoder to perform this analysis in a timely manner since this issue requires relatively quick response times. 

Due to the difficulty of predicting L2 order books (all orders open), we will use the L2 information and L1 information to predict the L1 results (mid, bid, ask prices)

