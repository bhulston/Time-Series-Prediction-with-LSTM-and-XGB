# Short-Frame-Price-Prediction-on-Limit-Order-Book-Data - Short look 
(Check out the extended read me if you have time)

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
 
 We will use the direct approach with an XGBoost model, and the recursive approach with an LSTM neural network. The goal is to capture trends, so we look at the most recent 10 timesteps...

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

40 timesteps with XGBoost

![image](https://user-images.githubusercontent.com/79114425/210891184-6421d4a7-36b5-4353-83ee-d6a55bfb72a8.png)

* We don't see an increasing error even at the 40 timesteps in the future, compared to the original 10 that we predicted.
* 
Direct Approach explained:

<img width = "400" alt="image" src="https://user-images.githubusercontent.com/79114425/210893972-caa8babc-faa6-4bea-b652-c4ca3483d6fa.png">

## Data Collection
Every 30 seconds, I collect data on the current order book and save the data according to each "batch" because most exchange APIs don't have historical data.

Here is an example of the market depth(liquidity) near the mid-price:

<img width="476" alt="image" src="https://user-images.githubusercontent.com/79114425/210905099-4d98d045-e9c2-4736-ac9d-4ba04744b3f0.png">

## Feature Engineering
I engineereed about 12 features based off of the data we collected. The formulas and algorithms can be found in the feature engineering doc

To combat noise:
* I smooth data in hopes for the model to be less sensitive to sharp spikes, making it better at capturing general trends. 

A good example is with the directional signal value...

Non-smoothed data:

<img width="450" alt="image" src="https://user-images.githubusercontent.com/79114425/210905840-ee214bd0-b3d0-48de-b6c8-a9e90d3c88cd.png">

Smoothed data:

<img width="450" alt="image" src="https://user-images.githubusercontent.com/79114425/210905941-38d47593-17b1-493b-bbf0-faadbe0fe9cd.png">

It is clear we do a much better job of capturing "stationary" trends, because we are less sensitive to small changes.

## Baseline Models

First, I built some **univariate** baseline models using ARIMA and Exponential Smoothing. 

![image](https://user-images.githubusercontent.com/79114425/210890640-2a1affa7-14c2-49d0-93ac-87c3a8a8141f.png)

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
  * **max_depth**: The max depth of the trees. Making sure this value is not too high is crucial for good results. 
  * **learning_rate**: Many models out there have very small learning rates, but due to the stochastic nature of this project, a higher learning rate of 0.1 is more appropriate

Here we can see the performance of the XGBoost model in comparison to the baseline models we created.

![image](https://user-images.githubusercontent.com/79114425/210891496-e57eca25-8ea7-4965-9247-6bb4bce9b37b.png)


## LSTMs

To build the LSTM, there is some more data processing that is needed in comparison the XGBoost model. 
* I got a lot of inspiration from this [article](https://arxiv.org/pdf/2107.09055.pdf) as well

For the LSTM, we use two bidirectional LSTMs with several dense layers. The LSTMs also use a loockback function that allows us to use a sliding window to garner information from the past. 

Here we can see the LSTM model compared to the training data:

![image](https://user-images.githubusercontent.com/79114425/211132805-0aa5aa92-60af-4a92-9186-0e1f779cc68b.png)

Here is the model on the testing data:

![image](https://user-images.githubusercontent.com/79114425/211132813-43299fd0-41df-4f6b-baff-f352ae195881.png)

When using recursive approaches, errors can compound on each other. Here is a great illustration I found:

![image](https://user-images.githubusercontent.com/79114425/210911074-04370cc2-019b-400d-ba70-592debd205c0.png)


With a little bit of research, you will find that LSTM neural networks seem to perform pretty poorly on real financial data. The reason for this is that they are extremely prone to over-fitting, and on top of that, they perform poorly when working with auto-regression problems.

