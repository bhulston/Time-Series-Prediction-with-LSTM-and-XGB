# Short-Frame-Price-Prediction-on-Limit-Order-Book-Data
Businesses or individuals might need to liquidate large amounts of $$ on short time frames. For any organization that has to perform high-volume purchases or liquidations on the market, this could reduce costs

* Several online projects make "multistep" predictions using inputs from future steps to predict those same values, instead we need to make **multi-step** predictions, using _only_ the training data
   * We can either predict what those future inputs would be, or train the model to predict at several different timesteps in the future... More on this below

**The goal of this project is NOT to get as close as possible at each point, but rather to capture general trends and capitalize on those**

# Project Plan
1. Take snap shots of the order book state at different times, both L2 (all open orders) and L1 (bid, mid, ask price) data
2. Process data and perform feature extraction and engineering to create features that can be used properly by the models
3. Design a series of predictive models including LSTM Neural Networks and Gradient Tree Boosting to predict future order book states 
4. Compare model results and find the best model to predict order book states 

# Appendix
1. [Model Results](#model-results)
2. [Data Collection](#data-collection)
3. [Feature Engineering](#feature-engineering)
4. [Baseline Models](#baseline-models)
5. [XGBoost and LSTM models](#xgb-and-lstm-models)

## Model Results

There are a few ways to setup XGBoost (and LSTM) for multi-step predictions:
 1.   Using AutoRegression (or other regression based predictions) to predict univariate values which would be fed into our model for making predictions
 2.   Direct Approach: Fit the regressor for each time point we want to predict. This is essentially its own model per timestep we want to predict.
 3.   Recursive Approach:  Get multiple outputs by creating clusters of models that actually predict features individually at each timestep based on the previous value. And then a larger model that predicts the value we actually are focused on (bid/ask price) based on predicted features. Rinse & Repeat.

Below, we can see the predictions of 10 future timesteps of our ARIMA model, the best Exponential Smoothing model, our LSTM model, and our XGBoost model. I also plotted the previous 20 timesteps.

![image](https://user-images.githubusercontent.com/79114425/211131966-308bac14-764c-412d-b49d-d6a631ae0f27.png)

| Model  | MSE |
| ------------- | ------------- |
| Exponential Smoothing  | 1.833  |
| Arima | 3.496  |
| LSTM - Recursive | 1.972  |
| XGBoost - Direct | 1.725  |


* As you can see, the XGBoost model seems to perform the best, followed by our exponential smoothing baseline, and then the LSTM
* The LSTM model performs even worse as we increase the timesteps
    * By using the recursive approach, we essentially compound our error 
* On the contrary, by using the direct approach with XGBoost, we train several different models that each predict at different timesteps in the future, and avoid compounding errors

Direct Approach explained:

<img width = "400" alt="image" src="https://user-images.githubusercontent.com/79114425/210893972-caa8babc-faa6-4bea-b652-c4ca3483d6fa.png">

Because the XGBoost model performed pretty well, I wanted to see how it would perform when we trained it on 40 timesteps:

![image](https://user-images.githubusercontent.com/79114425/210891184-6421d4a7-36b5-4353-83ee-d6a55bfb72a8.png)

* We don't see an increasing error even at the 40 timesteps in the future, compared to the original 10 that we predicted.
   * By using the direct approach, we don't run into the issue of compounding error!!


## Data Collection
When trying to collect relevant data on Order Book L2s, many exchange APIs do not retain historical data (only present data). So the best option is to connect to a websocket (in my case, the BitMex one) and stream data over a set period of time. Every 30 seconds, I collect data on the current order book and save the data according to each "batch".

Some of the things we collected are: Order quantities at different bid and ask prices, timestamps, mid price. recent purchase price.
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

<img width="450" alt="image" src="https://user-images.githubusercontent.com/79114425/208793698-2324802d-453a-4c5e-a244-955cd417f7f6.png">

I engineered features that **suited my goal (trends, not prices**). This is important because when data is noisy, it can make predictions difficult and more "sporadic". To combat this:
* I added smoothed directional signals and smoothed price columns (like EMA). 
* By smoothing the data, I am hoping for the model to be less sensitive to sharp spikes, making it better at capturing general trends. 
* On top of that, the price is stationary at different points, which can be hard for ML models to learn

A good example is with the directional signal value

Non-smoothed data, i.e. a +1 when price moved up (Red for a downtrend, blank for no movement, and green for an uptrend):

<img width="450" alt="image" src="https://user-images.githubusercontent.com/79114425/210905840-ee214bd0-b3d0-48de-b6c8-a9e90d3c88cd.png">

Smoothed data, i.e. setting a threshold value for price movements to indicate 1 or -1, as well as using moving averages for the signal values:

<img width="450" alt="image" src="https://user-images.githubusercontent.com/79114425/210905941-38d47593-17b1-493b-bbf0-faadbe0fe9cd.png">
By smoothing the data we are able to reduce the "noise" in this feature and get a better representation of general directional trends. Looking at the image above, it is clear we do a much better job of capturing "stationary" trends, because we are less sensitive to small changes.




## Baseline Models

The dataset we are working with is L2 order book information from BitMex. This includes the orders at different price levels. I streamed data for around 24 hours, and data was collected very 30 seconds, amounting to 2500 rows, with 18 features columns after feature engineering. 

First, I built some **univariate** baseline models using ARIMA and Exponential Smoothing. 
* We used double exponential smoothing to capture trend movements, triple xponential smoothing for taking into account seasonality, and an ARIMA model (Autoregressive Integrated Moving Average)
  * Alpha in exponential smoothing models indicate to us how much we want to smooth the data. Higher alphas indicate we want to more heavily weight recent points.
  * These are the most common statistical methods for time-series predictions.
* Note that the triple exponential smoothing model is probably a bad choice for this task because on this short time frame, it's unlikely to have any meaningful seasonality trends

![image](https://user-images.githubusercontent.com/79114425/210890640-2a1affa7-14c2-49d0-93ac-87c3a8a8141f.png)
* While mse is a commmon measurement for success, I think in the case of this project, plotting the values gives us a much better idea of how models are performing relatively. 
* **The goal of this project is NOT to get as close as possible at each point, but rather to capture general trends (up, down, or stationary)**

# XGB and LSTM Models
After building the baselines, I built two models that take **multivariate** inputs to see if we can improve, a LSTM and XGBoosted Trees 
* For the LSTM I used a recursive approach, while for the XGBoost model, I used a direct approach

## XGBoost
XGBoost is short for "Extreme Gradient Boosting", and uses an ensemble of gradient boosted trees to make regression/classification predictions.
* The main thing this requires is a lot of hyper parameter tuning! With the right tuning, XGBoost models seem to consistently outperform many other models, including ones specifically built for time-series
* Some resources I used for XGBoost:
  * [XGBoost](https://arxiv.org/abs/1603.02754) - arxiv
  * [Time Series Prediction Models](https://arxiv.org/pdf/2103.14250.pdf) -arxiv
  * [Fine tuning XGBoost](https://medium.com/towards-data-science/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e) - medium

The main data processing we needed to conduct for XGBoost modeling was adding the lag features. Meaning at each timestep, I wanted to add values for the previous 20 timesteps as well so the XGBoost model would have relevant information on previous timesteps as well.

Hyperparameters:

<img width="200" alt="image" src="https://user-images.githubusercontent.com/79114425/210907642-69d7499b-7682-4127-8619-0d4f06d10e21.png">

* The most important hyperparameters I focused on when tuning were:
  * **n_estimators**: In a dataset with not that much data, we had to raise this from the default of 100 to get proper results
  * **max_depth**: The max depth of the trees. The whole point of an XGBoost model is to leverage an ensemble of weak learners. Thus, making sure this value is not too high is crucial for good results. 
  * **learning_rate**: How "fast" our model learns, or really the steps that it takes when using gradient descent for optimization. Many models out there have very small learning rates, but due to the stochastic nature of this project, a higher learning rate of 0.1 is more appropriate to avoid getting stuck in local minimas
  * **colsample_bytree**: This value indicates the proportions of columns to be used in each tree that is built. We reduce this because we do have a lot of features (about 20)

Here we can see the performance of the XGBoost model in comparison to the baseline models we created.

![image](https://user-images.githubusercontent.com/79114425/210891496-e57eca25-8ea7-4965-9247-6bb4bce9b37b.png)


## LSTMs

To build the LSTM, there is some more data processing that is needed in comparison the XGBoost model. The main things are:
* We need to scale the data to a [-1,1] scale, using an encoder and decoder to transform the values both ways
* Add a lookback transformation that is essentially a moving window of data from past points
  * LSTM models use sequences as inputs that have a shape in 3 dimensions - [batch size, lookback/sequences, num features]
* Add lag features from previous timesteps at the current timestep
* I got a lot of inspiration from this [article](https://arxiv.org/pdf/2107.09055.pdf) as well


Here we can see the LSTM model compared to the training data. Values as inputs for the model are standardized, but I scaled them back up for these representations:

![image](https://user-images.githubusercontent.com/79114425/211132805-0aa5aa92-60af-4a92-9186-0e1f779cc68b.png)

Here is the model on the testing data:

![image](https://user-images.githubusercontent.com/79114425/211132813-43299fd0-41df-4f6b-baff-f352ae195881.png)

While it seemed to perform really well on the training data, we can see our suspicions are confirmed that the predictions perform poorly. This is probably, in part, due to us using a recursive approach, rather than a direct one. When using recursive approaches, errors can compound on each other. Here is a great illustration I found:

![image](https://user-images.githubusercontent.com/79114425/210911074-04370cc2-019b-400d-ba70-592debd205c0.png)


With a little bit of research, you will find that LSTM neural networks seem to perform pretty poorly on real financial data. The reason for this is that they are extremely prone to over-fitting, and on top of that, they perform poorly when working with auto-regression problems.
* LSTM window sizes don't seem to make a big difference (or can perform worse), despite being one of the main features when used on financial data
* An LSTM model might be better suited as part of a language processing model

## Things/difficulties to note

This article highlights how predicting prices is difficult, as prices seem to change by a very small amount if not 0, with noise.
https://hackernoon.com/dont-be-fooled-deceptive-cryptocurrency-price-predictions-using-deep-learning-bf27e4837151 

Depending on the volume of data and processing time, we might need an auto encoder to perform this analysis in a timely manner since this issue requires relatively quick response times. The xgboost direct approach, while performing the best, also took the longest to train each time.

Due to the difficulty of predicting L2 order books (all orders open), we will use the L2 information and L1 information to predict the L1 results (mid, bid, ask prices)

Something I would like to do in the future is collect data over a much longer period of time, and also work with sentiment analysis to see if there is any effect.
