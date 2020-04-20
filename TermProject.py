import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ToolBox import split_df_train_test, plot_line_using_pandas_index, cal_auto_correlation, plot_acf, plot_heatmap, \
    adf_cal, plot_seasonal_decomposition, generic_holt_linear_winter, plot_multiline_chart_pandas_using_index, cal_mse, \
    dumb1

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

    # dimension of train data
    print()
    print("The dimension of train data is:")
    print(train.shape)

    # dimension of test data
    print()
    print("The dimension of test data is:")
    print(test.shape)

    # combining train and test data
    combined_data = train.append(test)

    # Stationarity
    print()
    adf_cal(combined_data, "traffic_volume")

    # Time series Decomposition
    # the train dataframe already has DateTimeIndex as index which specified the frequency as 'D'
    plot_seasonal_decomposition(train["traffic_volume"], None, "Multiplicative Residuals", "multiplicative")
    plot_seasonal_decomposition(train["traffic_volume"], None, "Additive Residuals", "additive")

    # to keep track of performance for all the models
    result_performance = pd.DataFrame({"MSE": [], "Model": []})

    # --------------------------------------------------- HOLT WINTER----------------------------------------------

    holt_winter_prediction = generic_holt_linear_winter(train["traffic_volume"], test["traffic_volume"], None, None,
                                                        "mul", None)
    holt_winter_mse = cal_mse(test["traffic_volume"], holt_winter_prediction)

    print("The MSE for Holt Winter model using seasonality='mul' and trend='None' is:")
    print(holt_winter_mse)

    result_performance = result_performance.append(
        pd.DataFrame({"MSE": [holt_winter_mse], "Model": ["Holt Winter Model"]}))

    # plot the predicted vs actual data
    holt_winter_df = test.copy(deep=True)
    holt_winter_df["traffic_volume"] = holt_winter_prediction

    plot_multiline_chart_pandas_using_index([train, test, holt_winter_df], "traffic_volume",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Date Time", "Traffic Volume",
                                            "Traffic Volume Prediction Using Holt Winter",
                                            rotate_xticks=True)

    # ----------------------------------------------- MULTIPLE LINEAR REGRESSION----------------------------------
    lm_combined = combined_data.copy(deep=True)

    # convert categorical into numerical columns
    lm_combined = pd.get_dummies(lm_combined)

    # separate train and test data
    lm_train = lm_combined[:len(train)]
    lm_test = lm_combined[len(train):]

    # Scaling the data using MixMax Scaler
    mm_scaler = MinMaxScaler()
    lm_train_mm_scaled = pd.DataFrame(mm_scaler.fit_transform(lm_train), columns=lm_train.columns)
    lm_train_mm_scaled.set_index(lm_train.index, inplace=True)

    lm_test_mm_scaled = pd.DataFrame(mm_scaler.transform(lm_test), columns=lm_test.columns)
    lm_test_mm_scaled.set_index(lm_test.index, inplace=True)

    # aic = []
    # bic = []
    # r_squared = []
    # adjusted_r_squared = []
    # config = []
    # mse = []
    # rmse = []

    features1 = np.setdiff1d(lm_train_mm_scaled.columns,
                             ["clouds_all", "holiday_Christmas Day", "holiday_Columbus Day",
                              "holiday_Independence Day", "holiday_Labor Day", "holiday_Martin Luther King Jr Day",
                              "holiday_Memorial Day", "holiday_New Years Day", "holiday_Thanksgiving Day",
                              "holiday_Washingtons Birthday"])

    config_list = [
        #  (lm_train, lm_test, "Unscaled data with no intercept", False),
        # (lm_train, lm_test, "Unscaled data with intercept", True),
        # (lm_train_ss_scaled, lm_test_ss_scaled, "Standard Scaled data with no intercept", False),
        # (lm_train_ss_scaled, lm_test_ss_scaled, "Standard Scaled data with intercept", True),
        (lm_train_mm_scaled, lm_test_mm_scaled, "MinMax Scaled data with no intercept", False),
        (lm_train_mm_scaled, lm_test_mm_scaled, "MinMax Scaled data with intercept", True),
        (lm_train_mm_scaled[features1], lm_test_mm_scaled[features1],
         "Feature Pruning using P-value and confidence interval", True)]

    for train_data, test_data, config_string, intercept_value in config_list:
        config_string, aic_value, bic_value, rsquared, rsquared_adj, mse1, rmse1, predictions = dumb1(train_data,
                                                                                                      test_data,
                                                                                                      config_string,
                                                                                                      intercept_value)

        # config.append(config_string)
        # aic.append(aic_value)
        # bic.append(bic_value)
        # r_squared.append(rsquared)
        # adjusted_r_squared.append(rsquared_adj)
        # mse.append(mse1)
        # rmse.append(rmse1)

    # lm_results = pd.DataFrame({"Configuration": config, "AIC": aic, "BIC": bic, "R Squared": r_squared,
    #                            "Adjusted R Squared": adjusted_r_squared, "mse": mse, "rmse": rmse})
    # sorted_lm = lm_results.sort_values("mse")
    # print()
    # print(sorted_lm.to_string())

    # TODO Feature Selection
    # based on p-values; confidence interval
    # Adjusted R-Squared of 0.937

    features1 = np.setdiff1d(lm_train_mm_scaled.columns,
                             ["clouds_all", "holiday_Christmas Day",
                              "holiday_Independence Day",
                              "holiday_Labor Day", "holiday_Memorial Day", "holiday_New Years Day",
                              "holiday_Thanksgiving Day",
                              "holiday_Washingtons Birthday", "weather_main_Snow", "weather_main_Rain",
                              "weather_main_Fog", "holiday_Martin Luther King Jr Day"])

    dumb1(lm_train_mm_scaled[features1], lm_test_mm_scaled[features1],
          "Feature Pruning using P-value and confidence interval", False)

    # Adjusted R-Squared of 0.101 [Weather_main_clear changed the results!! and also not using intercept drastically
    # changed the results!!]
    # This is final LM model
    features2 = np.setdiff1d(lm_train_mm_scaled.columns,
                             ["clouds_all", "holiday_Labor Day", "holiday_Christmas Day",
                              "holiday_New Years Day", "holiday_Thanksgiving Day", "holiday_Memorial Day",
                              "holiday_Independence Day", "holiday_Washingtons Birthday",
                              "holiday_Martin Luther King Jr Day", "weather_main_Clear", "weather_main_Fog",
                              "weather_main_Clouds"])

    config_string_final_scaled, aic_final_scaled, bic_final_scaled, rsquared_final_scaled, rsquared_adj_final_scaled, mse_final_scaled, rmse_final_scaled, prediction_final_scaled = dumb1(
        lm_train_mm_scaled[features2], lm_test_mm_scaled[features2],
        "Feature Pruning using P-value and confidence interval", False)
    lm_predictions_scaled = lm_test_mm_scaled.copy(deep=True)
    lm_predictions_scaled["traffic_volume"] = prediction_final_scaled

    plot_multiline_chart_pandas_using_index([lm_train_mm_scaled, lm_test_mm_scaled, lm_predictions_scaled],
                                            "traffic_volume",
                                            ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                            "Date Time", "Traffic Volume",
                                            "Traffic Volume Prediction Using Multiple Linear Model Scaled",
                                            rotate_xticks=True)

    # # -----------------------
    # # This is Final Model based on Non Scaled data
    # features3 = np.setdiff1d(lm_train.columns,
    #                          ["clouds_all", "holiday_Labor Day", "holiday_New Years Day",
    #                           "holiday_Thanksgiving Day", "holiday_Memorial Day", "holiday_Christmas Day",
    #                           "holiday_Independence Day", "weather_main_Snow", "month", "holiday_Washingtons Birthday",
    #                           "weather_main_Rain", "weather_main_Fog", "holiday_Martin Luther King Jr Day"])
    #
    # config_string_final, aic_final, bic_final, rsquared_final, rsquared_adj_final, mse_final, rmse_final, prediction_final = dumb1(
    #     lm_train[features3], lm_test[features3], "Feature Pruning using P-value and confidence interval", False)
    #
    # lm_predictions = lm_test.copy(deep=True)
    # lm_predictions["traffic_volume"] = prediction_final
    #
    # plot_multiline_chart_pandas_using_index([lm_train, lm_test, lm_predictions], "traffic_volume",
    #                                         ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
    #                                         "Date Time", "Traffic Volume",
    #                                         "Traffic Volume Prediction Using Multiple Linear Model",
    #                                         rotate_xticks=True)

    # TODO interpret correlation matrix and which columns to keep and whether to keep categorical variables? To check for multicolinearity!!
    # Correlation Matrix with seaborn heatmap and pearson’s correlation coefficient
    # Based on Correlation Matrix removed the  weather_main_Clear column which was causing the collinearity
    # columns like clouds_all,
    # based on multicollinearity

    # least mse score for LM model is for mm model
    # corr = lm_train[features3].corr()
    # print()
    # print(corr.to_string())
    # print()
    # plot_heatmap(corr, "Heatmap for Unscaled Linear Model")

    # # --------------------------------------- ARMA ---------------------------------------------------------------
    # # # create GPAC Table
    # # j = 8
    # # k = 8
    # # lags = j + k
    # # ry = cal_auto_correlation(train["traffic_volume"], lags)
    # # gpac_table = create_gpac_table(j, k, ry)
    # # print(gpac_table.to_string())
    # #
    # # plot_heatmap(gpac_table, "GPAC Table for Traffic Volume")
    # #
    # # # estimate the order of the process
    # # possible_order = [(2, 0), (2, 5), (2, 7), (4, 0), (4, 2), (4, 5), (4, 7), (7, 0), (7, 8), (8, 1), (9, 5), (6, 5)]
    # # # possible_order = [(n_a, n_b) for n_a in range(1, j + 1) for n_b in range(k)]
    # #
    # # print()
    # #
    # # for n_a, n_b in possible_order:
    # #     try:
    # #         # estimate the model parameters
    # #         model = statsmodels_estimate_parameters(n_a, n_b, train["traffic_volume"])
    # #
    # #         # predict the traffic_volume on test data
    # #         predictions = statsmodels_predict_ARMA_process(model, start="2018-05-07 00:00:00",
    # #                                                        stop="2018-09-30 00:00:00")
    # #
    # #         residuals = cal_forecast_errors(list(test["traffic_volume"]), list(predictions))
    # #         re = cal_auto_correlation(residuals, lags)
    # #
    # #         # confused here should it be test or train? => definitely should be test since the residual size
    # #         # corresponds to the size of test data
    # #         Q = Q_value(test, re)
    # #
    # #         # does it pass the chi square test?
    # #         if chi_square_test(Q, lags, n_a, n_b, alpha=0.01):
    # #             print()
    # #             statsmodels_print_parameters(model, n_a, n_b)
    # #             plot_acf(re, "ACF for residuals")
    # #
    # #             predictions_arma_df = test.copy(deep=True)
    # #             predictions_arma_df["traffic_volume"] = predictions
    # #
    # #             plot_multiline_chart_pandas_using_index([train, test, predictions_arma_df], "traffic_volume",
    # #                                                     ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
    # #                                                     "Date Time", "Traffic Volume",
    # #                                                     f"Traffic Volume Prediction Using ARMA n_a={n_a} and n_b={n_b}",
    # #                                                     rotate_xticks=True)
    # #             print()
    # #             mse = cal_mse(list(test["traffic_volume"]), list(predictions))
    # #             print(f"MSE n_a={n_a} and n_b={n_b} = {mse}")
    # #             print()
    # #             print(f"Model Summary for order n_a={n_a} and n_b={n_b}")
    # #             print(model.summary())
    # #             print()
    # #             # winsound.Beep(1000, 2000)
    # #
    # #
    # #     except Exception as e:
    # #         print(e)
    # #
    # # # TODO I am stuck cause I have only one model instead of 2 ARMA process models, Hmmmmmmmm try professor's
    # # #  technique of using previous values as input !!
    # #
    # # # OR Subtract mean from the data to relax contraint for ARMA
    # #
    # # # TODO finally simplify model!!
    # #
    # # # zero pole cancellation
    # #
    # # # winsound.Beep(2000, 2000)
    # # print("Finished")
    #
    # # --------------------------------------- ARMA Mean Subtracted --------------------------------------------------
    # # create GPAC Table
    # y = np.subtract(np.mean(combined_data["traffic_volume"]), combined_data["traffic_volume"])
    # j = 8
    # k = 8
    # lags = j + k
    # ry = cal_auto_correlation(y, lags)
    # gpac_table = create_gpac_table(j, k, ry)
    # print(gpac_table.to_string())
    #
    # plot_heatmap(gpac_table, "GPAC Table for Traffic Volume")
    #
    # # estimate the order of the process
    # # possible_order = [(2, 0), (2, 5), (2, 7), (4, 0), (4, 2), (4, 5), (4, 7), (7, 0), (7, 8), (8, 1), (9, 5), (6, 5)]
    # possible_order = [(n_a, n_b) for n_a in range(1, j + 1) for n_b in range(k)]
    #
    # print()
    # mse_arma = []
    # arma_config = []
    # for n_a, n_b in possible_order:
    #     try:
    #         # estimate the model parameters
    #         model = statsmodels_estimate_parameters(n_a, n_b, y)
    #
    #         # predict the traffic_volume on test data
    #         predictions = statsmodels_predict_ARMA_process(model, start="2018-05-07 00:00:00",
    #                                                        stop="2018-09-30 00:00:00")
    #
    #         residuals = cal_forecast_errors(list(y[len(train):]), list(predictions))
    #         re = cal_auto_correlation(residuals, lags)
    #
    #         # confused here should it be test or train? => definitely should be test since the residual size
    #         # corresponds to the size of test data
    #         Q = Q_value(test, re)
    #
    #         # does it pass the chi square test?
    #         if chi_square_test(Q, lags, n_a, n_b, alpha=0.01):
    #             print()
    #             statsmodels_print_parameters(model, n_a, n_b)
    #             plot_acf(re, "ACF for residuals")
    #
    #             predictions_arma_df = test.copy(deep=True)
    #             predictions_arma_df["traffic_volume"] = np.add(np.mean(combined_data["traffic_volume"]), predictions)
    #
    #             plot_multiline_chart_pandas_using_index([train, test, predictions_arma_df], "traffic_volume",
    #                                                     ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
    #                                                     "Date Time", "Traffic Volume",
    #                                                     f"Traffic Volume Prediction Using ARMA n_a={n_a} and n_b={n_b}",
    #                                                     rotate_xticks=True)
    #             print()
    #             mse = cal_mse(list(test["traffic_volume"]), list(predictions))
    #             mse_arma.append(mse)
    #             arma_config.append((n_a, n_b))
    #             print(f"Model Summary for order n_a={n_a} and n_b={n_b}")
    #             print(model.summary())
    #             print()
    #             winsound.Beep(1000, 2000)
    #
    #     except Exception as e:
    #         print("Exception is", e)
    #
    # # TODO I am stuck cause I have only one model instead of 2 ARMA process models, Hmmmmmmmm try professor's
    # #  technique of using previous values as input !!
    #
    # # OR Subtract mean from the data to relax contraint for ARMA
    #
    # # TODO finally simplify model!!
    #
    # # zero pole cancellation
    #
    # winsound.Beep(2000, 2000)
    #
    # sorted_arma = pd.DataFrame({"MSE": mse_arma, "Config": arma_config}).sort_values(["MSE"])
    # print(sorted_arma.to_string())
    # print("Finished")
