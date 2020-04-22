import warnings

import numpy as np
import pandas as pd
import winsound

from ToolBox import split_df_train_test, plot_line_using_pandas_index, cal_auto_correlation, plot_acf, plot_heatmap, \
    plot_multiline_chart_pandas_using_index, cal_mse, \
    cal_forecast_errors, \
    create_gpac_table, statsmodels_estimate_parameters, \
    statsmodels_predict_ARMA_process, Q_value, chi_square_test, statsmodels_print_parameters, gpac_order_chi_square_test

if __name__ == "__main__":
    # pandas print options
    pd.set_option("max_columns", 10)

    # TODO refactor for variable names
    # TODO code comments
    # TODO put appropriate Print Commands and run code from console before submitting!!
    # TODO reverse scale transform the outputs of Linear model to get consistent MSE
    # TODO compare MSE for centered and uncentered ARMA process

    # ignore warnings
    warnings.filterwarnings("ignore")

    # read the original data
    traffic_raw = pd.read_csv("./data/Metro_Interstate_Traffic_Volume.csv")

    # replace all the None values with nan
    traffic_copy = traffic_raw.replace("None", np.nan)

    # convert date string column to date time object
    traffic_copy["date_time"] = pd.to_datetime(traffic_copy["date_time"], format="%Y-%m-%d")
    traffic_copy = traffic_copy.set_index(traffic_copy["date_time"])

    # focus on data from 09/2016 to 09/2018
    traffic_clipped = traffic_copy.loc['2016-09-30':'2018-09-30'].copy(deep=True)

    # resample based on daily data
    traffic_resampled = traffic_clipped.groupby(pd.Grouper(freq="D")).aggregate(
        {"temp": "mean", "clouds_all": "mean", "weather_main": "first", "traffic_volume": "mean", "holiday": "first",
         "rain_1h": "mean", "snow_1h": "mean"})

    print()
    print("The dimension of the resampled data is as follows:")
    print(traffic_resampled.shape)

    print()
    print("The summary statistics of numeric data after resampling to daily data is:")
    print(traffic_resampled.describe(include=["float64"]))

    # drop the snow_1h column
    traffic_resampled.drop(["snow_1h"], axis=1, inplace=True)

    print()
    print("The summary statistics of categorical data after resampling to daily data is:")
    print(traffic_resampled.describe(include=["object"]))

    traffic_resampled["holiday"] = traffic_resampled["holiday"].replace({np.nan: "No Holiday"})
    print()
    print("After replacing all the holiday NaN columns with 'No Holiday' value we get value counts for holiday column "
          "as:")
    print(traffic_resampled["holiday"].value_counts())

    # weather_main column value counts before condensing
    print()
    print("The value counts for weather_main column before condensing it:")
    print(traffic_resampled["weather_main"].value_counts())

    # condense the weather main categorical values
    traffic_resampled = traffic_resampled.replace(
        {"weather_main": {"Drizzle": "Rain", "Thunderstorm": "Rain", "Mist": "Fog", "Haze": "Fog", "Smoke": "Fog"}})

    # weather_main column value counts after condensing
    print()
    print("The value counts for weather_main column after condensing it:")
    print(traffic_resampled["weather_main"].value_counts())

    # after resampling we find these many NaN rows
    print()
    print("After resampling and data cleaning, column count with NaN values are:")
    print(traffic_resampled.isnull().sum())

    # Plot of the dependent variable versus time.
    plot_line_using_pandas_index(traffic_resampled, "traffic_volume", "Traffic Volume over time", "Navy", "Time",
                                 y_axis_label="Traffic Volume")

    # ACF of the dependent variable.
    autocorrelation = cal_auto_correlation(list(traffic_resampled["traffic_volume"]), 200)
    plot_acf(autocorrelation, "ACF plot for Traffic Volume")

    # Correlation Matrix with seaborn heatmap and pearson’s correlation coefficient
    corr = traffic_resampled.corr()
    plot_heatmap(corr, "Heatmap for Correlation Coefficient for Traffic Volume Data")

    # split into train and test(20%) dataset
    train, test = split_df_train_test(traffic_resampled, 0.2)

    # # dimension of train data
    # print()
    # print("The dimension of train data is:")
    # print(train.shape)
    #
    # # dimension of test data
    # print()
    # print("The dimension of test data is:")
    # print(test.shape)
    #
    # # combining train and test data
    # combined_data = train.append(test)
    #
    # # Stationarity
    # print()
    # adf_cal(combined_data, "traffic_volume")
    #
    # # Time series Decomposition
    # # the train dataframe already has DateTimeIndex as index which specified the frequency as 'D'
    # plot_seasonal_decomposition(train["traffic_volume"], None, "Multiplicative Residuals", "multiplicative")
    # plot_seasonal_decomposition(train["traffic_volume"], None, "Additive Residuals", "additive")
    #
    # # to keep track of performance for all the models
    # result_performance = pd.DataFrame(
    #     {"Model": [], "MSE": [], "RMSE": [], "Residual Mean": [], "Residual Variance": []})
    #
    # # --------------------------------------------------- HOLT WINTER----------------------------------------------
    # # holt winter prediction
    # holt_winter_prediction = generic_holt_linear_winter(train["traffic_volume"], test["traffic_volume"], None, None,
    #                                                     "mul", None)
    # # holt winter mse
    # holt_winter_mse = cal_mse(test["traffic_volume"], holt_winter_prediction)
    #
    # print()
    # print("The MSE for Holt Winter model is:")
    # print(holt_winter_mse)
    #
    # # holt winter rmse
    # holt_winter_rmse = np.sqrt(holt_winter_mse)
    # print()
    # print("The RMSE for Holt Winter model is:")
    # print(holt_winter_rmse)
    #
    # # holt winter residual
    # residuals_holt_winter = cal_forecast_errors(list(test["traffic_volume"]), holt_winter_prediction)
    # residual_autocorrelation_holt_winter = cal_auto_correlation(residuals_holt_winter, len(holt_winter_prediction))
    #
    # # holt winter residual variance
    # holt_winter_variance = np.var(residuals_holt_winter)
    # print()
    # print("The Variance of residual for Holt Winter model is:")
    # print(holt_winter_variance)
    #
    # # holt winter residual mean
    # holt_winter_mean = np.mean(residuals_holt_winter)
    # print()
    # print("The Mean of residual for Holt Winter model is:")
    # print(holt_winter_mean)
    #
    # # holt winter residual ACF
    # plot_acf(residual_autocorrelation_holt_winter, "ACF plot using Holt Winter Residuals")
    #
    # # add the results to common dataframe
    # result_performance = result_performance.append(
    #     pd.DataFrame(
    #         {"Model": ["Holt Winter Model"], "MSE": [holt_winter_mse], "RMSE": [holt_winter_rmse],
    #          "Residual Mean": [holt_winter_mean], "Residual Variance": [holt_winter_variance]}))
    #
    # # plot the predicted vs actual data
    # holt_winter_df = test.copy(deep=True)
    # holt_winter_df["traffic_volume"] = holt_winter_prediction
    #
    # plot_multiline_chart_pandas_using_index([train, test, holt_winter_df], "traffic_volume",
    #                                         ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
    #                                         "Time", "Traffic Volume",
    #                                         "Traffic Volume Prediction Using Holt Winter",
    #                                         rotate_xticks=True)
    #
    # # ----------------------------------------------- MULTIPLE LINEAR REGRESSION----------------------------------
    # lm_combined = combined_data.copy(deep=True)
    #
    # # convert categorical into numerical columns
    # lm_combined = pd.get_dummies(lm_combined)
    #
    # # separate train and test data
    # lm_train = lm_combined[:len(train)]
    # lm_test = lm_combined[len(train):]
    #
    # # Scaling the data using MixMax Scaler
    # mm_scaler = MinMaxScaler()
    # lm_train_mm_scaled = pd.DataFrame(
    #     mm_scaler.fit_transform(lm_train[np.setdiff1d(lm_train.columns, ["traffic_volume"])]),
    #     columns=np.setdiff1d(lm_train.columns, ["traffic_volume"]))
    # lm_train_mm_scaled.set_index(lm_train.index, inplace=True)
    # lm_train_mm_scaled["traffic_volume"] = lm_train["traffic_volume"]
    #
    # lm_test_mm_scaled = pd.DataFrame(mm_scaler.transform(lm_test[np.setdiff1d(lm_test.columns, ["traffic_volume"])]),
    #                                  columns=np.setdiff1d(lm_test.columns, ["traffic_volume"]))
    # lm_test_mm_scaled.set_index(lm_test.index, inplace=True)
    # lm_test_mm_scaled["traffic_volume"] = lm_test["traffic_volume"]
    #
    # # linear model using all variables
    # basic_model = normal_equation_using_statsmodels(
    #     lm_train_mm_scaled[np.setdiff1d(lm_train_mm_scaled.columns, "traffic_volume")],
    #     lm_train_mm_scaled["traffic_volume"], intercept=False)
    #
    # print()
    # print("The summary of linear model with all variables is:")
    # print(basic_model.summary())
    #
    # features = np.setdiff1d(lm_train_mm_scaled.columns,
    #                         ["rain_1h", "holiday_Christmas Day", "holiday_Memorial Day", "holiday_Thanksgiving Day",
    #                          "holiday_New Years Day", "holiday_Independence Day", "holiday_Labor Day", "clouds_all",
    #                          "holiday_Washingtons Birthday", "holiday_Martin Luther King Jr Day"])
    #
    # # linear model using features which pass the t-test
    # pruned_model = normal_equation_using_statsmodels(lm_train_mm_scaled[np.setdiff1d(features, "traffic_volume")],
    #                                                  lm_train_mm_scaled["traffic_volume"], intercept=False)
    #
    # print()
    # print("The summary of linear model after feature selection:")
    # print(pruned_model.summary())
    #
    # # linear model predictions
    # lm_predictions = normal_equation_prediction_using_statsmodels(pruned_model, lm_test_mm_scaled[
    #     np.setdiff1d(features, "traffic_volume")], intercept=False)
    #
    # # linear model mse
    # lm_mse = cal_mse(test["traffic_volume"], lm_predictions)
    #
    # print()
    # print("The MSE for Linear Model model is:")
    # print(lm_mse)
    #
    # # linear model rmse
    # lm_rmse = np.sqrt(lm_mse)
    # print()
    # print("The RMSE for Linear Model model is:")
    # print(lm_rmse)
    #
    # # linear model residual
    # residuals_lm = cal_forecast_errors(list(test["traffic_volume"]), lm_predictions)
    # residual_autocorrelation = cal_auto_correlation(residuals_lm, len(lm_predictions))
    #
    # # linear model residual variance
    # lm_variance = np.var(residuals_lm)
    # print()
    # print("The Variance of residual for Linear Model model is:")
    # print(lm_variance)
    #
    # # linear model residual mean
    # lm_mean = np.mean(residuals_lm)
    # print()
    # print("The Mean of residual for Linear Model model is:")
    # print(lm_mean)
    #
    # # linear model residual ACF
    # plot_acf(residual_autocorrelation, "ACF plot for Linear Model Residuals")
    #
    # # linear model Q value
    # Q_value_lm = box_pierce_test(len(test), residuals_lm, len(test))
    # print()
    # print("The Q Value of residuals for Linear Model model is:")
    # print(Q_value_lm)
    #
    # # add the results to common dataframe
    # result_performance = result_performance.append(
    #     pd.DataFrame(
    #         {"Model": ["Multiple Linear Regression Model"], "MSE": [lm_mse], "RMSE": [lm_rmse],
    #          "Residual Mean": [lm_mean], "Residual Variance": [lm_variance]}))
    #
    # # plot the actual vs predicted values
    # lm_predictions_scaled = lm_test_mm_scaled.copy(deep=True)
    # lm_predictions_scaled["traffic_volume"] = lm_predictions
    #
    # plot_multiline_chart_pandas_using_index([lm_train_mm_scaled, lm_test_mm_scaled, lm_predictions_scaled],
    #                                         "traffic_volume",
    #                                         ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
    #                                         "Time", "Traffic Volume",
    #                                         "Traffic Volume Prediction Using Multiple Linear Model Scaled",
    #                                         rotate_xticks=True)
    #
    # # correlation coefficient for linear model after feature selection
    # corr = lm_train[features].corr()
    # label_ticks = ["Columbus Day", "No Holiday", "State Fair", "Veterans Day", "temp", "traffic_volume", "Clear",
    #                "Clouds", "Fog", "Rain", "Snow"]
    # plot_heatmap(corr, "Correlation Coefficient for Linear Model after feature selection", label_ticks, label_ticks, 45)

    # --------------------------------------- ARMA ---------------------------------------------------------------
    j = 12
    k = 12
    lags = j + k
    ry = cal_auto_correlation(train["traffic_volume"], lags)

    # create GPAC Table
    gpac_table = create_gpac_table(j, k, ry)
    print(gpac_table.to_string())

    plot_heatmap(gpac_table, "GPAC Table for Traffic Volume")

    # estimate the order of the process
    possible_order = [(6, 5)]

    pass_order = gpac_order_chi_square_test(possible_order, train["traffic_volume"], "2018-05-07 00:00:00",
                                            "2018-09-30 00:00:00", lags, test["traffic_volume"])

    print("Finished")
