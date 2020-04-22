import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.holtwinters as ets
from lifelines import KaplanMeierFitter
from scipy.signal import dlsim
from scipy.stats import chi2
from scipy.stats import t
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def plot_line_using_pandas_index(df, y_axis_data, titleOfPlot, color, x_axis_label, legend=False,
                                 y_axis_label=None, legendList=None, x_tick_data=False):
    """
    Plots line chart based on index as x axis and y axis label passed
    """
    line_chart = df[y_axis_data].plot(kind="line", rot=90, legend=legend, title=titleOfPlot, color=color)
    line_chart.set_xlabel(x_axis_label)

    line_chart.set_ylabel(y_axis_data if y_axis_label is None else y_axis_label)
    if legend:
        line_chart.legend(legendList)
    if x_tick_data:
        plt.xticks(df.index.values)
    plt.show()


def plot_line(df, x_axis_data, y_axis_data, titleOfPlot, color, legend=False, x_axis_label=None, y_axis_label=None,
              legendList=None, x_tick_data=False):
    """
    Plots line chart based on x axis label and y axis label passed
    """
    line_chart = df.plot.line(x=x_axis_data, y=y_axis_data, rot=90, legend=legend, title=titleOfPlot, color=color)
    line_chart.set_xlabel(x_axis_data if x_axis_label is None else x_axis_label)
    line_chart.set_ylabel(y_axis_data if y_axis_label is None else y_axis_label)
    if legend:
        line_chart.legend(legendList)
    if x_tick_data:
        plt.xticks(df[x_axis_data])
    plt.show()


def plot_line_subplot(df, x_axis_data, y_axis_data, titleOfPlot, color, axes, legend=False, x_axis_label=None,
                      y_axis_label=None,
                      legend_list=None):
    """
    Creates line chart based on x axis label and y axis label passed and the axes object, but does not plot it
    """

    line_chart = df.plot.line(x=x_axis_data, y=y_axis_data, rot=45, legend=legend, title=titleOfPlot, color=color,
                              ax=axes)
    line_chart.set_xlabel(x_axis_data if x_axis_label is None else x_axis_label)
    line_chart.set_ylabel(y_axis_data if y_axis_label is None else y_axis_label)
    if legend:
        line_chart.legend(legend_list)


def get_descriptive_stats(df, attribute):
    """"
    Computes mean, standard deviation and variance for a Dataframe attribute
    """
    mean_data = df[attribute].mean().round(3)
    variance_data = df[attribute].var().round(3)
    std_data = df[attribute].std().round(3)

    return mean_data, variance_data, std_data


def compute_stepwise_stats(df, time_attribute, data_attribute):
    """
    Computes stepwise mean and variance based on data frame and attribute specified
    """

    # initialize empty data frame
    stepwise_df = pd.DataFrame(columns=["Time", "Stepwise Mean", "Stepwise Variance"])

    for index in range(0, len(df)):
        # compute mean and variance and append it to empty data frame
        stepwise_df = stepwise_df.append({"Time": df.iloc[index][time_attribute],
                                          # using index + 1 since head(0) is NaN and hence start from next index
                                          "Stepwise Mean": df.head(index + 1)[data_attribute].mean(),
                                          "Stepwise Variance": df.head(index + 1)[data_attribute].var()},
                                         ignore_index=True)
    return stepwise_df


def adf_cal(df, attribute):
    """
    Computes and prints ADF Statistics using statsmodels.tsa.stattools.adfuller()
    """
    print("ADF Test for", attribute)
    result = adfuller(df[attribute])
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values: ")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))
    print()

    if result[1] < 0.05:
        print("Since p-value is less than 0.05, reject null hypothesis thus time series data is Stationary")
    else:
        print(
            "Since p-value is not less than 0.05, we failed to reject null hypothesis thus time series data is "
            "Non-Stationary")
    print()
    return result


def plot_seasonal_decomposition(input_list, frequency_of_data: int, title, type_of_decomposition="additive"):
    decomposed_data = seasonal_decompose(input_list, type_of_decomposition, period=frequency_of_data)
    decomposed_data.plot()
    plt.title(title)
    plt.show()


def transform_using_differencing(df, time_attribute, data_attribute):
    """
    Transforming using first order differencing corrects trend in non stationary data. Differencing has a caveat that
    we lose the first data point.
    """

    # initialize empty data frame
    difference_df = pd.DataFrame(columns=[time_attribute, data_attribute])

    for index in range(0, len(df) - 1):
        difference_df = difference_df.append({
            time_attribute: df.iloc[index][time_attribute],
            # Difference Value = Next - Current
            data_attribute: (df.iloc[index + 1][data_attribute] - df.iloc[index][data_attribute]),
        }, ignore_index=True)

    return difference_df


def reverse_transform_for_differencing(original_input_list, differenced_df_list_with_predicted_values):
    """ returns transformed values for predicted values only"""
    last_index = len(original_input_list) - 1
    prediction_range = len(differenced_df_list_with_predicted_values) - len(original_input_list) + 1

    back_transformed = []
    predicted_sum = 0
    for i in range(prediction_range):
        predicted_sum += differenced_df_list_with_predicted_values[last_index + i]
        predicted_value = original_input_list[last_index] + predicted_sum
        back_transformed.append(predicted_value)

    return back_transformed


def transform_using_logarithms(df, data_attribute):
    """
    Transforming using logarithm corrects variance in non stationary data. \nNote: We are using log to the base 10.
    """
    log_df = df.copy(deep=True)
    log_df[data_attribute] = np.log10(log_df[data_attribute])

    return log_df


def reverse_transform_using_logarithms(original_input_list, log_transformed_list):
    reversed_log = np.power(10, log_transformed_list[len(original_input_list):])
    return list(reversed_log)


def correlation_coefficient_cal(x, y):
    """
    Python function that returns correlation coefficient based on formula of,
    r = (cross correlation of x and y) / ((std dev of x) * (std dev of y))
    Takes 2 dataset [series data] as input and returns the correlation coefficient
    """

    # find the mean of x
    x_mean = np.mean(x)

    # find the mean of y
    y_mean = np.mean(y)

    # multiply the difference between x mean and x with y mean and y
    numerator = np.sum(np.multiply(np.subtract(x, x_mean), np.subtract(y, y_mean)))

    # find standard deviation of x
    x_std_dev = np.sqrt(np.sum(np.square(np.subtract(x, x_mean))))

    # find standard deviation of y
    y_std_dev = np.sqrt(np.sum(np.square(np.subtract(y, y_mean))))

    # multiply x_std_dev and y_std_dev
    denominator = x_std_dev * y_std_dev

    # perform division
    if denominator != 0:
        # round the division to 3 decimal places
        return round(numerator / denominator, 3)
    else:
        return 0


def create_scatter_plot(x, y, x_label, y_label, title_of_plot, color):
    """Title should contain correlation coefficient"""
    plt.scatter(x, y, c=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title_of_plot)
    plt.show()


def plot_hist(input_data, title, color, x_axis_label=None, y_axis_label=None):
    """ Plots histogram based on list of input_data"""
    plt.hist(input_data, color=color)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.show()


def cal_auto_correlation(input_array, number_of_lags, precision=3):
    """
    :param precision: tells the precision of rounding
    :param input_array: a vector or array which contains the values
    :param number_of_lags: how many time shifts are
    desired when number_of_lags = 0, it means no time shift
    :return: a list containing the values of auto correlation for the number of lags specified
    """

    # find the mean
    mean_of_input = np.mean(input_array)

    # create empty result array
    result = []

    # compute denominator for autocorrelation equation
    denominator = np.sum(np.square(np.subtract(input_array, mean_of_input)))

    # iterate for the number of lags mentioned
    for k in range(0, number_of_lags):

        # initialize numerator
        numerator = 0

        # iterate from k to the size of input array
        for i in range(k, len(input_array)):
            numerator += (input_array[i] - mean_of_input) * (input_array[i - k] - mean_of_input)

        if denominator != 0:
            # perform division and append output to list
            result.append(np.round(numerator / denominator, precision))

    return result


def compute_autocorrelation_single_lag(input_array, lag, precision=3):
    # find the mean
    mean_of_input = np.mean(input_array)

    # compute denominator for autocorrelation equation
    denominator = np.sum(np.square(np.subtract(input_array, mean_of_input)))

    # initialize numerator
    numerator = 0

    # iterate from k to the size of input array
    for i in range(lag, len(input_array)):
        numerator += (input_array[i] - mean_of_input) * (input_array[i - lag] - mean_of_input)

    if denominator != 0:
        # perform division and append output to list
        return round(numerator / denominator, precision)


def plot_acf(autocorrelation, title_of_plot, x_axis_label="Lags", y_axis_label="Magnitude"):
    # make a symmetric version of autocorrelation using slicing
    symmetric_autocorrelation = autocorrelation[:0:-1] + autocorrelation
    x_positional_values = [i * -1 for i in range(0, len(autocorrelation))][:0:-1] + [i for i in
                                                                                     range(0, len(autocorrelation))]
    # plot the symmetric version using stem
    plt.stem(x_positional_values, symmetric_autocorrelation, use_line_collection=True)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title_of_plot)
    plt.show()


def cal_mse(actual_values, predicted_values):
    # mean square error
    return np.round(np.mean(np.square(np.subtract(predicted_values, actual_values))), 3)


def cal_sse(actual_values, predicted_values):
    # sum square errors
    return np.round(np.sum(np.square(np.subtract(predicted_values, actual_values))), 3)


def cal_forecast_errors(actual_values, predicted_values):
    # forecast errors is difference between  observed values and predicted values
    return np.subtract(actual_values, predicted_values)


def plot_multi_scatter_plot(list_of_y_data, list_of_x_data, list_of_legends, title_of_chart, x_axis_label, y_axis_label,
                            list_of_colors, size_of_marker=50):
    """Plots multiple scatter plots on same chart"""
    for i in range(0, len(list_of_x_data)):
        plt.scatter(list_of_y_data[i], list_of_x_data[i], s=size_of_marker, c=list_of_colors[i])
    plt.xlabel(x_axis_label)
    plt.legend(list_of_legends)
    plt.ylabel(y_axis_label)
    plt.title(title_of_chart)
    plt.show()


def plot_multi_line_chart(list_of_y_data, x_common_data, list_of_colors, list_of_legends, title_of_chart,
                          x_axis_label, y_axis_label):
    """ Plots multiple lines on same chart, using common x-axis data"""
    for i in range(0, len(list_of_y_data)):
        # create line charts
        plt.plot(list_of_y_data[i], color=list_of_colors[i], label=list_of_legends[i], marker="o", linestyle="--")

        # add the x axis data
        plt.xticks(x_common_data)

    # set the x axis label
    plt.xlabel(x_axis_label)

    # set the y axis label
    plt.ylabel(y_axis_label)

    # create the legend
    plt.legend()

    # set the title of chart
    plt.title(title_of_chart)

    plt.show()


def box_pierce_test(number_of_samples, residuals, lags):
    """

    :param number_of_samples: Total number of samples in the data :param residuals: residuals are difference between
    predicted and observed values :param lags: To perform autocorrelation we specify the lags (if h = 2,
    it means ignore zeroth and find first and second, to do this we add 1 to the lag and then ignore the 0th ACF)
    :return: Q statistic rounded to 3 decimals
    """
    return round(number_of_samples * np.sum(np.square(cal_auto_correlation(residuals, lags + 1)[1:])), 3)


def Q_value(y, autocorrelation_of_residuals):
    """ Computes Q value for comparing with chi_critical for Chi Square Test. Same as box_pierce_test(..)"""
    Q = len(y) * np.sum(np.square(autocorrelation_of_residuals[1:]))
    return Q


def generic_average_method(input_data, step_ahead):
    """Predicts the average value for the specified steps"""
    # returns a flat prediction
    return [np.round(np.mean(input_data), 3) for i in range(0, step_ahead)]


def generic_naive_method(input_data, step_ahead):
    """Predicts using naive method for specified steps"""
    return [input_data[-1] for i in range(0, step_ahead)]


def generic_drift_method(input_data, step_ahead):
    """Predicts using drift method for specified steps"""
    predicted_values = []

    for i in range(0, step_ahead):
        predicted_value = input_data[-1] + (i + 1) * ((input_data[-1] - input_data[0]) / (len(input_data) - 1))

        predicted_values.append(round(predicted_value, 3))

    return predicted_values


def generic_ses_method(input_data, step_ahead, alpha, initial_condition):
    """Predicts using SES method for specified steps. SES has a flat prediction curve and works best for data with no
    trend and no seasonality """

    summation_part = 0
    for h in range(0, len(input_data)):
        summation_part += (alpha * ((1 - alpha) ** h)) * input_data[len(input_data) - h - 1]

    predicted_value = summation_part + ((1 - alpha) ** len(input_data)) * initial_condition

    return [round(predicted_value, 3) for i in range(0, step_ahead)]


def generic_holt_linear_trend(train_data, test_data):
    """ Works best for data with trend only"""
    holt_linear = ets.Holt(train_data).fit()
    predictions = list(holt_linear.forecast(len(test_data)))
    return predictions


def generic_holt_linear_winter(train_data, test_data, seasonal_period: int, trend="mul", seasonal="mul",
                               trend_damped=False):
    """ Works best for data with trend and seasonality"""
    holt_winter = ets.ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal,
                                           seasonal_periods=seasonal_period, damped=trend_damped).fit()
    holt_winter_forecast = list(holt_winter.forecast(len(test_data)))
    return holt_winter_forecast


def split_df_train_test(df, test_size, random_seed=42):
    """ Test set size should be equal to the size of the prediction we want."""
    train, test = train_test_split(df, shuffle=False, test_size=test_size, random_state=random_seed)
    return train, test


def plot_multiline_chart_pandas_using_index(list_of_dataframes, y_axis_common_data, list_of_label, list_of_color,
                                            x_label, y_label, title_of_plot, rotate_xticks=False):
    """Plots multiple line charts into single chart. This API uses list of pandas data having same x_axis label and
    same y_axis label """
    for i, df in enumerate(list_of_dataframes):
        df[y_axis_common_data].plot(label=list_of_label[i], color=list_of_color[i])
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title_of_plot)
    if rotate_xticks:
        plt.xticks(rotation=90)
    plt.show()


def plot_multiline_chart_pandas(list_of_dataframes, x_axis_common_data, y_axis_common_data, list_of_label,
                                list_of_color,
                                x_label, y_label, title_of_plot, rotate_xticks=False):
    """Plots multiple line charts into single chart. This API uses list of pandas data having same x_axis label and
    same y_axis label """
    for i, df in enumerate(list_of_dataframes):
        plt.plot(df[x_axis_common_data], df[y_axis_common_data], label=list_of_label[i], color=list_of_color[i])
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title_of_plot)
    if rotate_xticks:
        plt.xticks(rotation=90)
    plt.show()


def cal_standard_error(number_of_features, forecast_errors):
    """Calculate standard deviation of error using the forecast residuals"""
    denominator = len(forecast_errors) - number_of_features - 1
    return np.sqrt(np.sum(np.square(forecast_errors)) / denominator)


def cal_variance_of_error(number_of_features, forecast_errors):
    """Calculate variance of error using the forecast residuals"""
    return np.square(cal_standard_error(number_of_features, forecast_errors))


def cal_r_squared(predicted_values, actual_values):
    """Calculates R-square value"""
    return np.square(correlation_coefficient_cal(predicted_values, actual_values))


def cal_adjusted_r_squared(predicted_values, actual_values, number_of_features):
    """Calculates adjusted r squared value"""
    r_squared = cal_r_squared(predicted_values, actual_values)
    return 1 - ((1 - r_squared) * ((len(predicted_values) - 1) / (len(predicted_values) - number_of_features - 1)))


def cal_95_confidence(predicted_values, std_error, x_matrix, intercept=True):
    """predicted values computed using predict_using_normal_equation_parameters,
        std_error computed using cal_standard_error(..)
        x_matrix is the test feature matrix
    """
    if intercept:
        # append ones if intercept present
        x_matrix = np.column_stack((np.ones(shape=x_matrix.shape[0]), x_matrix))

    confidence_tuples = []
    for i in range(0, len(predicted_values)):
        interval = 1.96 * std_error * (np.sqrt(
            1 + np.dot(np.dot(x_matrix[i], np.linalg.inv(np.dot(x_matrix.transpose(), x_matrix))),
                       np.vstack(x_matrix[i]))))

        confidence_tuples.append([predicted_values[i] - interval, predicted_values[i] + interval])

    return confidence_tuples


def normal_equation_regression(train_df, target_label: object, intercept=True):
    """Performs linear regression using the normal equation"""
    if intercept:
        x_train = np.column_stack(
            (np.ones(shape=len(train_df)), train_df[np.setdiff1d(train_df.columns, target_label)]))
    else:
        x_train = train_df[np.setdiff1d(train_df.columns, target_label)]

    y_train = np.vstack(train_df[target_label])

    normal_equation_coefficient = np.round(np.dot(
        np.dot(np.linalg.inv(np.dot(x_train.transpose(), x_train)), x_train.transpose()), y_train), 3)

    return normal_equation_coefficient.flatten()


def predict_using_normal_equation_parameters(test_data_nested_list, normal_equation_coefficient_list, intercept=True):
    """Predict the test data inputs based on normal equation containing the intercepts and parameters of input
    variable"""

    if intercept:
        # add the intercept
        predicted_values = normal_equation_coefficient_list[0]
        start_index = 1
    else:
        predicted_values = 0
        start_index = 0

    for i in range(len(test_data_nested_list)):
        predicted_values += test_data_nested_list[i] * normal_equation_coefficient_list[i + start_index]

    return list(predicted_values)


def normal_equation_using_statsmodels(train_feature_list, train_target_list, intercept=True):
    if intercept:
        train_feature_list = sm.add_constant(train_feature_list)

    model = sm.OLS(train_target_list, train_feature_list)
    results = model.fit()
    return results


def normal_equation_prediction_using_statsmodels(OLS_model, test_feature_list, intercept=True):
    if intercept:
        test_feature_list = sm.add_constant(test_feature_list, has_constant='add')

    predicted_values_OLS = OLS_model.predict(test_feature_list)
    return predicted_values_OLS


def cyclic_shift(input_list, shift_by, clip_by):
    """Shifts the array to the left by amount specified in shift_by variables. The input array shrinks by clip_by due
    to loss in data point; which is a caveat of autoregression """
    return np.roll(input_list, -shift_by)[:-clip_by]


def autoregression_data_prepper(y_input_list, order_of_auto_regressor):
    """Prepares the input array for autoregression by creating data with order_of_auto_regressor shift
    order_of_auto_regressor = n_a
    """

    prepared_data = pd.DataFrame()

    for i in range(1, order_of_auto_regressor + 1):
        shifted_data = cyclic_shift(y_input_list, order_of_auto_regressor - i, order_of_auto_regressor)
        prepared_data["x(" + str(i) + ")"] = np.multiply(shifted_data, -1)

    prepared_data["y(t)"] = y_input_list[order_of_auto_regressor:]

    return prepared_data


def generate_auto_regressor_data(number_of_samples, initial_condition, parameter_list, mean_of_white_noise=0,
                                 std_dev_of_white_noise=1, seed=0):
    """Generates white noise and then creates AR data based on the order which is inferred from size of
    parameter_list """
    np.random.seed(seed)
    white_noise = np.random.normal(mean_of_white_noise, std_dev_of_white_noise, number_of_samples)
    y = np.zeros(shape=len(white_noise))

    # multiply by -1 since we take AR coefficients to the RHS
    parameter_list = np.multiply(parameter_list, -1)

    for t in range(len(white_noise)):

        temp = 0

        # iterate over each coefficient
        for i in range(len(parameter_list)):
            # if the index goes below zero then use initial condition
            if (t - (i + 1)) < 0:
                temp += parameter_list[i] * initial_condition
            else:
                temp += parameter_list[i] * y[t - (i + 1)]

        # add white noise
        y[t] = temp + white_noise[t]

    return y


def generate_moving_averages_data(number_of_samples, initial_condition, parameter_list, mean_of_white_noise=0,
                                  std_dev_of_white_noise=1, seed=0):
    """Generates white noise and then creates MA data based on the order which is inferred from size of
    parameter_list """
    np.random.seed(seed)
    white_noise = np.random.normal(mean_of_white_noise, std_dev_of_white_noise, number_of_samples)
    y = np.zeros(shape=len(white_noise))

    for t in range(len(white_noise)):

        # add white noise
        temp_sum = white_noise[t]

        # iterate over each coefficient
        for i in range(len(parameter_list)):
            # if the index goes below zero then use initial condition
            if (t - (i + 1)) < 0:
                temp_sum += parameter_list[i] * initial_condition
            else:
                temp_sum += parameter_list[i] * white_noise[t - (i + 1)]

        # store value in list
        y[t] = temp_sum

    return y


def generate_arma_data(number_of_samples, initial_condition, parameter_list_ar, parameter_list_ma,
                       mean_of_white_noise=0, std_dev_of_white_noise=1, seed=0):
    """Generates white noise and then creates ARMA data based on the order of AR which is inferred from size of
    parameter_list_ar and order of MA which is inferred from size of parameter_list_ma"""

    np.random.seed(seed)

    white_noise = np.random.normal(mean_of_white_noise, std_dev_of_white_noise, number_of_samples)
    y = np.zeros(shape=len(white_noise))

    # multiply by -1 since we take AR coefficients to the RHS
    parameter_list_ar = np.multiply(parameter_list_ar, -1)

    for t in range(len(white_noise)):

        # add white noise
        temp_sum = white_noise[t]

        # iterate over each coefficient of AR process [denominator]
        for i in range(len(parameter_list_ar)):
            # if the index goes below zero then use initial condition
            if (t - (i + 1)) < 0:
                temp_sum += parameter_list_ar[i] * initial_condition
            else:
                temp_sum += parameter_list_ar[i] * y[t - (i + 1)]

        # iterate over each coefficient of MA process [numerator]
        for i in range(len(parameter_list_ma)):
            # if the index goes below zero then use initial condition
            if (t - (i + 1)) < 0:
                temp_sum += parameter_list_ma[i] * initial_condition
            else:
                temp_sum += parameter_list_ma[i] * white_noise[t - (i + 1)]

        # store value in list
        y[t] = temp_sum

    return y


def generate_arma_data_user_input():
    # Generates ARMA(na,nb) process using inputs from the user
    print()
    number_of_samples = int(input("Enter the number of data samples:"))

    ar_order = int(input("Enter the order of AR portion:"))
    ar_coefficients = []
    for order in range(ar_order):
        ar_coefficients.append(float(input("Enter coefficient excluding coefficient for y(t) and the sign of "
                                           "coefficients \n "
                                           "should be entered as though the coefficients are on LHS of ARMA equation")))

    ma_order = int(input("Enter the order of MA portion:"))
    ma_coefficients = []
    for order in range(ma_order):
        ma_coefficients.append(float(input("Enter coefficients excluding coefficient for e(t) and press enter")))

    # set seed
    seed = int(input("Enter the seed for random data:"))

    return generate_arma_data(number_of_samples, 0, ar_coefficients, ma_coefficients, 0, 1, seed)


def perform_auto_regression():
    """Performs auto regression using input from user"""
    print()
    number_of_samples = int(input("Enter number of samples:\n"))
    order_of_auto_regressor = int(input("Enter the order # of the AR process:\n"))

    parameter_list = []
    print("Enter coefficients excluding coefficient for y(t) and the sign of coefficients "
          "should be entered as though the coefficients are on LHS of AR equation")
    for i in range(order_of_auto_regressor):
        parameter_list.append(float(input()))

    intercept_str = input("Do you want intercept? (Y/N)")
    intercept = True if intercept_str.lower() == "y" else False

    y = generate_auto_regressor_data(number_of_samples, 0, parameter_list)
    train_df = autoregression_data_prepper(y, order_of_auto_regressor)
    coefficients = normal_equation_regression(train_df, "y(t)", intercept)

    return parameter_list, coefficients


def get_max_denominator_indices(j, k_scope):
    # create denominator indexes based on formula for GPAC
    denominator_indices = np.zeros(shape=(k_scope, k_scope), dtype=np.int64)

    for k in range(k_scope):
        denominator_indices[:, k] = np.arange(j - k, j + k_scope - k)

    return denominator_indices


def get_apt_denominator_indices(max_denominator_indices, k):
    apt_denominator_indices = max_denominator_indices[-k:, -k:]
    return apt_denominator_indices


def get_numerator_indices(apt_denominator_indices, k):
    numerator_indices = np.copy(apt_denominator_indices)
    # take the 0,0 indexed value and then create a range of values from (indexed_value+1, indexed_value+k)
    indexed_value = numerator_indices[0, 0]
    y_matrix = np.arange(indexed_value + 1, indexed_value + k + 1)

    # replace the last column with this new value
    numerator_indices[:, -1] = y_matrix

    return numerator_indices


def get_ACF_by_index(numpy_indices, acf):
    # select values from an array based on index specified
    result = np.take(acf, numpy_indices)
    return result


def get_phi_value(denominator_indices, numerator_indices, ry, precision=5):
    # take the absolute values since when computing phi value, we use ACF and ACF is symmetric in nature
    denominator_indices = np.abs(denominator_indices)
    numerator_indices = np.abs(numerator_indices)

    # replace the indices with the values of ACF
    denominator = get_ACF_by_index(denominator_indices, ry)
    numerator = get_ACF_by_index(numerator_indices, ry)

    # take the determinant
    denominator_det = np.round(np.linalg.det(denominator), precision)
    numerator_det = np.round(np.linalg.det(numerator), precision)

    # divide it and return the value of phi
    return np.round(np.divide(numerator_det, denominator_det), precision)


def create_gpac_table(j_scope, k_scope, ry, precision=5):
    # initialize gpac table
    gpac_table = np.zeros(shape=(j_scope, k_scope), dtype=np.float64)

    for j in range(j_scope):
        # create the largest denominator
        max_denominator_indices = get_max_denominator_indices(j, k_scope)

        for k in range(1, k_scope + 1):
            #  slicing largest denominator as required
            apt_denominator_indices = get_apt_denominator_indices(max_denominator_indices, k)

            # for numerator replace denominator's last columnn with index starting from j+1 upto k times
            numerator_indices = get_numerator_indices(apt_denominator_indices, k)

            # compute phi value
            phi_value = get_phi_value(apt_denominator_indices, numerator_indices, ry, precision)
            gpac_table[j, k - 1] = phi_value

    gpac_table_pd = pd.DataFrame(data=gpac_table, columns=[k for k in range(1, k_scope + 1)])

    return gpac_table_pd


def cal_t_test_correlation_coefficient(correlation_coefficient, number_of_observations, number_of_confounding_variables,
                                       alpha_level, two_tail=False):
    degree_of_freedom = number_of_observations - 2 - number_of_confounding_variables
    t_value = np.abs(
        correlation_coefficient * np.sqrt(np.divide(degree_of_freedom, 1 - np.square(correlation_coefficient))))
    # t value from t table
    if two_tail:
        alpha_level = alpha_level / 2
    critical_t_test = t.ppf(1 - alpha_level, degree_of_freedom)

    print()
    if t_value > critical_t_test:
        print(
            f"The absolute value of test statistic {t_value} exceeded the critical t-value {critical_t_test} from the table; "
            f"hence the correlation coefficient (partial correlation) is statistically significant.")
    else:
        print(
            f"The absolute value of test statistic {t_value} did not exceed the critical t-value {critical_t_test} from the table;\n "
            f"hence the correlation coefficient (partial correlation) is not statistically significant.")

    return t_value, critical_t_test


def cal_partial_correlation(a, b, c):
    r_ab = correlation_coefficient_cal(a, b)
    r_ac = correlation_coefficient_cal(a, c)
    r_bc = correlation_coefficient_cal(b, c)

    return np.divide(r_ab - r_ac * r_bc, np.multiply(np.sqrt(1 - np.square(r_ac)), np.sqrt(1 - np.square(r_bc))))


# LMA steps
def LMA_step_0(n_a, n_b):
    # initially theta are all zeroes
    theta = np.zeros(shape=(n_a + n_b, 1))
    return theta.flatten()


def LMA_step_1(n_a, n_b, theta, delta, y):
    X_list = []

    # compute e using theta
    e = extract_white_noise_from_y(n_a, theta, y)

    # sse_old
    sse_old = np.matmul(np.transpose(e), e)

    # add delta to each theta values
    for i in range((n_a + n_b)):
        # theta_second = theta.deepcopy()
        theta_second = theta.copy()
        theta_second[i] = theta[i] + delta

        # compute e with the modified theta value
        e_second = extract_white_noise_from_y(n_a, theta_second, y)

        x_i = (e - e_second) / delta
        X_list.append(x_i)

    X = np.column_stack(X_list)

    A = np.matmul(np.transpose(X), X)
    g = np.matmul(np.transpose(X), e)

    return A, g, sse_old


def extract_num_den_from_theta(theta, n_a):
    theta_ar = theta.copy()
    theta_ma = theta.copy()
    # ar
    ar = np.insert(theta_ar[:n_a], 0, 1)
    # ma
    ma = np.insert(theta_ma[n_a:], 0, 1)

    # pad zeroes
    if len(ar) < len(ma):
        ar = np.append(ar, [0 for i in range(len(ma) - len(ar))])
    elif len(ma) < len(ar):
        ma = np.append(ma, [0 for i in range(len(ar) - len(ma))])

    return ar, ma


def extract_white_noise_from_y(n_a, theta, y):
    den_ar, num_ma = extract_num_den_from_theta(theta, n_a)

    # extract white noise from the given data of y(t)
    system = (den_ar, num_ma, 1)
    extracted_white_noise = dlsim(system, y)[1].flatten()

    return extracted_white_noise


def LMA_step_2(mu, A, g, n_a, n_b, theta, y):
    # we are updating theta in Step 2

    # compute delta_theta and return it
    mu_identity = np.multiply(mu, np.identity(n_a + n_b))
    delta_theta = np.round(np.matmul(np.linalg.inv(A + mu_identity), g), 8)

    # compute theta_new
    theta_new = np.add(theta.flatten(), delta_theta.flatten())

    extracted_white_noise = extract_white_noise_from_y(n_a, theta_new, y)

    # compute SSE_new using theta_new
    sse_new = np.matmul(extracted_white_noise.transpose(), extracted_white_noise)

    # # check if SSE_new is NaN
    if np.isnan(sse_new):
        sse_new = 10 ** 10

    return delta_theta, sse_new, theta_new


def LMA_step_3(y, sse_old, sse_new, n_a, n_b, mu, delta, delta_theta, A, g, theta_new, mu_max=10 ** 27,
               MAX_ITERATIONS=70):
    iterations = 0
    iteration_list = []
    sse_list = []

    while iterations < MAX_ITERATIONS:
        # keeping track of iterations and sse new
        iteration_list.append(iterations)
        sse_list.append(sse_new)

        if sse_new < sse_old:
            if np.linalg.norm(delta_theta) < 10 ** -3:

                theta = theta_new
                var_error = sse_new / (len(y) - (n_a + n_b))
                covariance_theta = np.multiply(var_error, np.linalg.inv(A))
                return theta, var_error, covariance_theta, pd.DataFrame({"SSE": sse_list, "Iteration": iteration_list})

            else:
                theta = theta_new
                mu /= 10

        # theta is theta_old
        theta = theta_new

        # return to step 1
        A, g, sse_old = LMA_step_1(n_a, n_b, theta, delta, y)

        # return to step 2
        delta_theta, sse_new, theta_new = LMA_step_2(mu, A, g, n_a, n_b, theta, y)

        while sse_new >= sse_old:
            mu *= 10
            if mu > mu_max:
                print("MU Error")
                return None, None, None, None

            # return to step 2
            delta_theta, sse_new, theta_new = LMA_step_2(mu, A, g, n_a, n_b, theta, y)

        iterations += 1

        if iterations > MAX_ITERATIONS:
            print("Iterations Error")
            return None, None, None, None


def perform_LMA_parameter_estimation(n_a, n_b, y, seed=42):
    np.random.seed(seed)
    # step 0
    # theta are unknown parameters
    theta = LMA_step_0(n_a, n_b)

    # step 1
    delta = 10 ** -6
    A, g, sse_old = LMA_step_1(n_a, n_b, theta, delta, y)

    # step 2
    mu = 0.01
    delta_theta, sse_new, theta_new = LMA_step_2(mu, A, g, n_a, n_b, theta, y)

    # step 3
    return LMA_step_3(y, sse_old, sse_new, n_a, n_b, mu, delta, delta_theta, A, g, theta_new)


def compute_confidence_interval(estimated_parameters, covariance_matrix, n):
    confidence_interval = []

    estimated_parameters = list(estimated_parameters)
    covariance_matrix = covariance_matrix.reset_index(drop=True)
    covariance_matrix.columns = [i for i in range(covariance_matrix.shape[0])]

    for i in range(n):
        upper = estimated_parameters[i] + 2 * np.sqrt(covariance_matrix[i][i])
        lower = estimated_parameters[i] - 2 * np.sqrt(covariance_matrix[i][i])
        confidence_interval.append([upper, lower])

        # For printing the results
        print(f"{lower} < {estimated_parameters[i]} < {upper}")

    return confidence_interval


def plot_survival_curve(duration_list: list, event_list: list, label_list: list, title_of_chart=object):
    if len(duration_list) == len(event_list):
        kmf = KaplanMeierFitter()
        for i in range(len(duration_list)):
            kmf.fit(durations=duration_list[i], event_observed=event_list[i], label=label_list[i])
            kmf.plot()
        plt.title(title_of_chart)
        plt.show()
    else:
        print("Duration and event list size are not same, thus cannot create survival plot.")


def plot_heatmap(corr_df, title, xticks=None, yticks=None, x_axis_rotation=0, annotation=True):
    sns.heatmap(corr_df, annot=annotation)
    plt.title(title)
    if xticks is not None:
        plt.xticks([i for i in range(len(xticks))], xticks, rotation=x_axis_rotation)
    if yticks is not None:
        plt.yticks([i for i in range(len(yticks))], yticks)
    plt.show()


def chi_square_test(Q, lags, n_a, n_b, alpha=0.01):
    dof = lags - n_a - n_b
    chi_critical = chi2.isf(alpha, df=dof)

    if Q < chi_critical:
        print(f"The residual is white and the estimated order is n_a= {n_a} and n_b = {n_b}")
    else:
        print(f"The residual is not white with n_a={n_a} and n_b={n_b}")

    return Q < chi_critical


# ----------------------------- statsmodels related wrapper classes--------------------------------
def statsmodels_estimate_parameters(n_a, n_b, y, trend="nc"):
    model = sm.tsa.ARMA(y, (n_a, n_b)).fit(trend=trend, disp=0)
    return model


def statsmodels_print_parameters(model, n_a, n_b):
    # print the parameters which are estimated
    for i in range(n_a):
        print("The AR coefficients a {}".format(i), "is:", model.params[i])
    print()
    for i in range(n_b):
        print("The MA coefficients b {}".format(i), "is:", model.params[i + n_a])
    print()


def statsmodels_print_covariance_matrix(model, n_a, n_b):
    print(f"Estimated covariance matrix for n_a = {n_a} and n_b = {n_b}: \n{model.cov_params()}")
    print()
    return model.cov_params()


def statsmodels_print_variance_error(model, n_a, n_b):
    print(f"Estimated variance of error for n_a = {n_a} and n_b = {n_b}: \n{model.sigma2}")
    print()
    return model.sigma2


def statsmodels_print_confidence_interval(model, n_a, n_b):
    # confidence interval
    print(
        f"The confidence interval for estimated parameters for n_a = {n_a} and n_b = {n_b}: \n {model.conf_int()}")
    print()
    return model.conf_int()


def statsmodels_predict_ARMA_process(model, start, stop):
    model_hat = model.predict(start=start, end=stop)
    return model_hat


def statsmodels_plot_predicted_true(y, model_hat, n_a, n_b):
    true_data = pd.DataFrame({"Magnitude": y, "Samples": [i for i in range(len(y))]})
    fitted_data = pd.DataFrame({"Magnitude": model_hat, "Samples": [i for i in range(len(model_hat))]})

    plot_multiline_chart_pandas([true_data, fitted_data], "Samples", "Magnitude", ["True data", "Fitted data"],
                                ["red", "blue"], "Samples", "Magnitude",
                                f"ARMA process with n_a={n_a} and n_b={n_b}")


def statsmodels_print_roots_AR(model):
    print("Real part:")
    for root in model.arroots:
        print(root.real)
    print("Imaginary part:")
    for root in model.arroots:
        print(root.imag)


def statsmodels_print_roots_MA(model):
    print("Real part:")
    for root in model.maroots:
        print(root.real)
    print("Imaginary part:")
    for root in model.maroots:
        print(root.imag)


# check whether order passes chi square test
def gpac_order_chi_square_test(possible_order_ARMA, train_data, start, stop, lags, actual_outputs):
    results = []

    for n_a, n_b in possible_order_ARMA:
        try:
            # estimate the model parameters
            model = statsmodels_estimate_parameters(n_a, n_b, train_data)

            # predict the traffic_volume on test data
            predictions = statsmodels_predict_ARMA_process(model, start=start, stop=stop)

            # calculate forecast errors
            residuals = cal_forecast_errors(actual_outputs, predictions)

            # autocorrelation of residuals
            re = cal_auto_correlation(residuals, lags)

            # compute Q value for chi square test
            Q = Q_value(actual_outputs, re)

            # checking the chi square test
            if chi_square_test(Q, lags, n_a, n_b):
                results.append((n_a, n_b))

        except Exception as e:
            # print(e)
            pass

    return results
