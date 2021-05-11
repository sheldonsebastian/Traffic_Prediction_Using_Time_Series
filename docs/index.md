# Traffic Volume Prediction

Sheldon Sebastian

![](saved_images/banner.jpg)
<center>Photo by <a href="https://unsplash.com/@5tep5?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Alexander Popov</a> on <a href="https://unsplash.com/s/photos/traffic?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a></center>
  

## Abstract

<div style="text-align: justify"> 
We are predicting the traffic volume per day for the I94 interstate. The traffic volume per day is the number of cars which use the I94 interstate between St. Paul and Minneapolis. To make accurate forecasts, 6 models Average Model, Naïve Model, Drift Model, Holt Winter Model, Multiple linear regression and ARMA were used. The performance of the all models are compared and the best performing model is recommended to forecast traffic volume.
<br>
<br>
<i>Keywords:</i> Forecasting, Traffic, Average model, Naïve model, Drift Model, Holt Winter, ARMA, Linear Regression
</div>

## Table of contents

1. Introduction
2. Data description and preprocessing
3. Stationarity check
4. Average Model
5. Naive Model
6. Drift Model
7. Time series decomposition
8. Holt Winters method
9. Multiple Linear Regression
10. ARMA model
11. Best Model
12. Conclusion


## 1. Introduction
<div style="text-align: justify">
We are predicting the number of cars in a day on the I94 interstate between St. Paul and Minneapolis as shown in the below figure:
</div>
<center><img src="saved_images/img1.jpg"/></center>

<br>
<div style="text-align: justify">
Business value of project:
<br>
<ol>
<li>Avoid Traffic Congestion: We can predict the days when there will be heavy traffic congestion and thus take contingencies to avoid them.</li>
<li>Road Maintenance: Using the traffic volume predictions we can estimate how long before the road needs repairs and we can schedule repairs when there is least traffic volume.</li>
</ol>
For achieving the goal of predicting traffic volume, we are considering 6 prediction models: Average, Naïve, Drift, Holt Winter, ARMA model and Multiple Linear Regression model.<br><br>
In average model, all the future predictions are average of the training data. In naïve model, we predict all the future values by taking the last value of the training dataset. In drift model, we plot a line from the first point of the data to the last point and extend it to predict all the future values. In the Holt Winter method, we will find whether traffic volume follows additive or multiplicative trend and then make predictions.
For the Linear Regression Model, we will scale the feature variables and perform data cleaning and then make predictions. Finally, for ARMA model, we will estimate the order of the ARMA process using GPAC table, estimate the parameters for ARMA and check whether the residuals pass the chi square test or not.<br><br>
Once all the models are created, we will compare the performance and recommend the best performing model.
</div>

## 2. Data description and preprocessing

<div style="text-align: justify">
The dataset has hourly traffic volume from October 2012 to September 2018. Traffic volume is defined as count of cars in an hour on the interstate. As described previously, the hourly traffic volume is tracked between Minneapolis and St Paul, MN.

<br><br>
The dataset is sourced from the following website:<br>
https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume

</div>

<br>
<b>Resampling strategy</b>
<br>
<div style="text-align: justify">
For computational purposes and model interpretability the hourly data was resampled into daily data. Also, we are focusing on traffic volume data for September 2016 to September 2018.
</div>

<div style="text-align: justify">
When we perform resampling the following functions were applied to the variables:
<ul>
<li>Mean: temp, clouds_all, traffic_volume, rain_1h, snow_1h.</li>
<li>First: weather_main, holiday.</li>
</ul>

After resampling the shape of the dataset is <b>(731,7)</b>.
</div>

<br>
<b>Summary Statistics</b>
<br>

<div style="text-align: justify">
The summary statistics for numeric columns are as follows:<br>
<center><img src="saved_images/img2.jpg"/></center>
We notice that the snow_1h column has all values as zero, thus we remove that column.
<br><br>
The summary statistics for categorical columns are as follows:<br>
<center><img src="saved_images/img3.jpg"/></center>
We notice that the holiday column has 22 values only, thus we replace all the other NaN values with “No Holiday” values. After replacing all the holiday NaN columns with 'No Holiday' value we get value counts for holiday column as:
<br>
<center><img src="saved_images/img4.jpg"/></center>
<br>
We also notice that the weather_main column contains 9 unique values which are:
<br>
<center><img src="saved_images/img5.jpg"/></center>
<br>
We condense this information as follows:
<ul>
<li>Rain additionally covers the values Drizzle, Thunderstorm</li>
<li>Fog additionally covers the values Mist, Haze, Smoke</li>
</ul>
<br>
Thus, after condensing the value counts are as follows:
<br>
<center><img src="saved_images/img6.jpg"/></center>
<br>
Finally, after resampling and data cleaning the column count with NaN values are:
<br>
<center><img src="saved_images/img7.jpg"/></center>
<br>
Hence, we do not need to perform any data imputation.
</div>

<br>
<b>Traffic Volume over time</b>
<br>
<div style="text-align: justify">
We plot the traffic volume over time, the traffic volume data is resampled to daily data and the scope of data is from 09/2016 to 09/2018.
</div>
<center><img src="saved_images/img8.jpg"/></center>

<br>
<b>ACF of traffic volume</b>
<br>
<div style="text-align: justify">
For plotting the ACF plot we have set the value of lag = 200.
<center><img src="saved_images/img9.jpg"/></center>
<br>
We notice that the ACF values show decay at lag = 200.
</div>

<br>
<b>Train-Test Split</b>
<br>
<div style="text-align: justify">
We split the resampled data into train and test datasets. The dimension for train dataset is <b>(584,6)</b>. The dimension for test dataset is <b>(147,6)</b>
</div>


## 3. Stationarity check

<div style="text-align: justify">
To check if traffic volume is stationary or not, we perform the ADF test.
<br>
<center><img src="saved_images/img10.jpg"/></center>
<br>
From ADF test we conclude that traffic volume is stationary.
</div>


## 4. Average Model

<div style="text-align: justify">
We compute the mean of training data and perform h step predictions to match the size of the test data. The model statistics are as follows:<br>
</div>

<center>
<table>
<tr>
<th>RMSE</th>
<th>MSE</th>
<th>Residual Variance</th>
<th>Residual Mean</th>
</tr>
<tr>
<td>531.918</td>
<td>282937.565</td>
<td>279859.840</td>
<td>55.477</td>
</tr>
</table>
</center>

<br>
The plot of forecasted values with the actual value is shown below:
<center><img src="saved_images/img11.jpg"/></center>
<br><br>
The ACF of residuals is as follows:
<center><img src="saved_images/img12.jpg"/></center>
<br>
We conclude that the ACF plot does not resemble white noise.

## 5. Naive Model

## 6. Drift Model

## 7. Time series decomposition

## 8. Holt Winters method

## 9. Multiple Linear Regression

## 10. ARMA model

## 11. Best Model

## 12. Conclusion

## References

<div style="text-align: justify">

1. Link to all code files: https://github.com/sheldonsebastian/Traffic_Prediction_Using_Time_Series
<br>

</div>


