# Short-Frame-Price-Prediction-on-Limit-Order-Book-Data - Short look (Check out the extended read me if you have time)
Businesses or individuals might need to liquidate large amounts of $$ on short time frames. Data was collected live every 30 seconds over the course of 48 hours from BitMex exchange APIs.

# Project Plan
1. Take snap shots of the order book state at different times from BitMex websockets, collect all L2(individual orders) data
2. Process data and perform feature extraction and engineering to create features that can be used properly by the models
3. Design a series of predictive models including LSTM Neural Networks and Gradient Tree Boosting to predict future order book states 
4. Compare model results and find the best model to predict order book states 

# Table of Contents
1. [Model Results](#model-results)
2. [Data Collection](#data-collection)
3. [Feature Engineering](#feature-engineering)
4. [Baseline Models](#baseline-models)
5. [XGBoost and LSTM models](#xgb-and-lstm-models)

## Model Results

There are a few ways to setup XGBoost (and LSTM) for multi-step predictions:
 1.   **Direct Approach:** Fit a new regressor for each future time point we want to predict. 
 2.   **Recursive Approach:**  Creating clusters of models that predict features individually at each timestep for each variable. And then a larger model that regresses the final value we want
 
 We will use the direct approach with an XGBoost model, and the recursive approach with an LSTM neural network. Below, we can see the predictions of 10 future timesteps of our ARIMA, Exponential Smoothing, LSTM, and our XGBoost model. 

![image](https://user-images.githubusercontent.com/79114425/211131966-308bac14-764c-412d-b49d-d6a631ae0f27.png)

| Model  | MSE |
| ------------- | ------------- |
| Exponential Smoothing  | 1.833  |
| Arima | 3.496  |
| LSTM - Recursive | 1.972  |
| XGBoost - Direct | 1.725  |


* XGBoost model seems to perform the best, followed by our exponential smoothing baseline, and then the LSTM
* The LSTM model performs worse as we increase the timesteps
    * This is because by using the recursive approach, we essentially compound our error, more on this in model section

Direct Approach explained:

<img width = "400" alt="image" src="https://user-images.githubusercontent.com/79114425/210893972-caa8babc-faa6-4bea-b652-c4ca3483d6fa.png">

Because the XGBoost model performed pretty well, I wanted to see how it would perform when we trained it on 40 timesteps:

![image](https://user-images.githubusercontent.com/79114425/210891184-6421d4a7-36b5-4353-83ee-d6a55bfb72a8.png)

* We don't see an increasing error even at the 40 timesteps in the future, compared to the original 10 that we predicted.
   * By using the direct approach, we don't run into the issue of compounding error!!

## Data Collection
Every 30 seconds, I collect data on the current order book and save the data according to each "batch" because most exchange APIs don't have historical data.

Here is a graph of the mid-price data that we collected over about 24 hours:

<img width="480" alt="image" src="https://user-images.githubusercontent.com/79114425/210905065-c9dd4492-2a99-4ae7-a38d-563bf76398fe.png">

Here is an example of the market depth(liquidity) near the mid-price:

<img width="476" alt="image" src="https://user-images.githubusercontent.com/79114425/210905099-4d98d045-e9c2-4736-ac9d-4ba04744b3f0.png">

## Feature Engineering
I engineereed about 12 features based off of the data we collected. The formulas and algorithms can be found in the feature engineering doc:
* Logarithmic transformation of returns
* Liquidity depth at different levels to the mid-price
* Smoothed data and directional signals
* EMA and SMA

To combat a lot of the noise:
* I smooth data in hopes for the model to be less sensitive to sharp spikes, making it better at capturing general trends. 

A good example is with the directional signal value...

Non-smoothed data, i.e. a +1 when price moved up (Red for a downtrend, blank for no movement, and green for an uptrend):

<img width="450" alt="image" src="https://user-images.githubusercontent.com/79114425/210905840-ee214bd0-b3d0-48de-b6c8-a9e90d3c88cd.png">

Smoothed data, i.e. setting a threshold value for price movements to indicate 1 or -1, as well as using moving averages for the signal values:

<img width="450" alt="image" src="https://user-images.githubusercontent.com/79114425/210905941-38d47593-17b1-493b-bbf0-faadbe0fe9cd.png">

Looking at the image above, it is clear we do a much better job of capturing "stationary" trends, because we are less sensitive to small changes.


## Baseline Models

First, I built some **univariate** baseline models using ARIMA and Exponential Smoothing. 
* We used double exponential smoothing to capture trend movements, triple xponential smoothing for taking into account seasonality, and an ARIMA model (Autoregressive Integrated Moving Average)

![image](https://user-images.githubusercontent.com/79114425/210890640-2a1affa7-14c2-49d0-93ac-87c3a8a8141f.png)
* While mse is a commmon measurement for success, I think in the case of this project, visualizing the values gives us a much better idea of how models are performing relatively. 

# XGB and LSTM Models
After building the baselines, I built two models that take **multivariate** inputs to see if we can improve, a LSTM and XGBoosted Trees 

## XGBoost

Some resources I used for XGBoost:
  * [XGBoost](https://arxiv.org/abs/1603.02754) - arxiv
  * [Time Series Prediction Models](https://arxiv.org/pdf/2103.14250.pdf) -arxiv
  * [Fine tuning XGBoost](https://medium.com/towards-data-science/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e) - medium

At each timestep, I wanted to add values for the previous 20 timesteps as well so the XGBoost model would have relevant information on previous timesteps as well.

Hyperparameters:

<img width="200" alt="image" src="https://user-images.githubusercontent.com/79114425/210907642-69d7499b-7682-4127-8619-0d4f06d10e21.png">

* The most important hyperparameters I focused on when tuning were:
  * **n_estimators**: In a dataset with not that much data, we had to raise this from the default of 100 to get proper results
  * **max_depth**: The max depth of the trees. Making sure this value is not too high is crucial for good results. 
  * **learning_rate**: Many models out there have very small learning rates, but due to the stochastic nature of this project, a higher learning rate of 0.1 is more appropriate
  * **colsample_bytree**: Proportion of columns in each individual tree. We reduce this because we do have a lot of features (about 20)

Here we can see the performance of the XGBoost model in comparison to the baseline models we created.

![image](https://user-images.githubusercontent.com/79114425/210891496-e57eca25-8ea7-4965-9247-6bb4bce9b37b.png)


## LSTMs

To build the LSTM, there is some more data processing that is needed in comparison the XGBoost model. 
* I got a lot of inspiration from this [article](https://arxiv.org/pdf/2107.09055.pdf) as well

For the LSTM, we use two bidirectional LSTMs with several dense layers. The LSTMs also use a loockback function that allows us to use a sliding window to garner information from the past. On top of the lookback function, we add lag features of those previous timesteps, letting us assign unique weights both to the lookback window as a whole, as well as the individual points in it.

Here we can see the LSTM model compared to the training data. Values as inputs for the model are standardized, but I scaled them back up for these representations:

![image](https://user-images.githubusercontent.com/79114425/211132805-0aa5aa92-60af-4a92-9186-0e1f779cc68b.png)

Here is the model on the testing data:

![image](https://user-images.githubusercontent.com/79114425/211132813-43299fd0-41df-4f6b-baff-f352ae195881.png)

While it seemed to perform really well on the training data, we can see our suspicions are confirmed that the predictions perform poorly. This is probably, in part, due to us using a recursive approach, rather than a direct one. When using recursive approaches, errors can compound on each other. Here is a great illustration I found:

![image](https://user-images.githubusercontent.com/79114425/210911074-04370cc2-019b-400d-ba70-592debd205c0.png)


With a little bit of research, you will find that LSTM neural networks seem to perform pretty poorly on real financial data. The reason for this is that they are extremely prone to over-fitting, and on top of that, they perform poorly when working with auto-regression problems.

## Things/difficulties to note

This article highlights how predicting prices is difficult, as prices seem to change by a very small amount if not 0, with noise.
https://hackernoon.com/dont-be-fooled-deceptive-cryptocurrency-price-predictions-using-deep-learning-bf27e4837151 

Depending on the volume of data and processing time, we might need an auto encoder to perform this analysis in a timely manner since this issue requires relatively quick response times. The xgboost direct approach, while performing the best, also took the longest to train each time.

Due to the difficulty of predicting L2 order books (all orders open), we will use the L2 information and L1 information to predict the L1 results (mid, bid, ask prices)

Something I would like to do in the future is collect data over a much longer period of time, and also work with sentiment analysis to see if there is any effect.
