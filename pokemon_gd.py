import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
pd.set_option('max_column', 200)
pd.set_option('display.width', 200)

pokemon_dataframe = pd.read_csv('pokemon.csv', sep=',')
pokemon_dataframe = pokemon_dataframe.reindex(np.random.permutation(pokemon_dataframe.index))

# 一开始，我们只使用一个数值输入特征 total_rooms。以下代码会从 california_housing_dataframe 中提取 total_rooms 数据，并使
# 用 numeric_column 定义特征列，这样会将其数据指定为数值：
my_feature = pokemon_dataframe[['cp','species']]

feature_columns = [tf.feature_column.numeric_column('cp'),tf.feature_column.numeric_column('species')]

print(pokemon_dataframe.describe())

# 定义目标
targets = pokemon_dataframe["cp_new"]

# 配置 LinearRegressor
# 接下来，我们将使用 LinearRegressor 配置线性回归模型，并使用 GradientDescentOptimizer（它会实现小批量随机梯度下降法 (SGD)）
# 训练该模型。learning_rate 参数可控制梯度步长的大小。

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

period = 10

for period in range(period):
    _ = linear_regressor.train(
        input_fn=lambda: my_input_fn(my_feature, targets),
        steps=100
    )

    # 评估模型

    # Create an input function for predictions.
    # Note: Since we're making just one prediction for each example, we don't
    # need to repeat or shuffle the data here.
    prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

    # Call predict() on the linear_regressor to make predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)

    # Format predictions as a NumPy array, so we can calculate error metrics.
    predictions = np.array([item['predictions'][0] for item in predictions])

    # Print Mean Squared Error and Root Mean Squared Error.
    mean_squared_error = metrics.mean_squared_error(predictions, targets)
    root_mean_squared_error = np.math.sqrt(mean_squared_error)
    print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
    print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

    min_house_value = pokemon_dataframe["cp_new"].min()
    max_house_value = pokemon_dataframe["cp_new"].max()
    min_max_difference = max_house_value - min_house_value

    print("Min. cp_new Value: %0.3f" % min_house_value)
    print("Max. cp_new Value: %0.3f" % max_house_value)
    print("Difference between Min. and Max.: %0.3f" % min_max_difference)
    print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    calibration_data.describe()

    sample = pokemon_dataframe.sample(n=50)

    # Get the min and max total_rooms values.R
    x_0 = sample["cp"].min()
    x_1 = sample["cp"].max()

    # Retrieve the final weight and bias generated during training.
    weight = linear_regressor.get_variable_value('linear/linear_model/cp/weights')[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    # Get the predicted median_house_values for the min and max total_rooms values.
    y_0 = weight * x_0 + bias
    y_1 = weight * x_1 + bias
    plt.clf()
    # Plot our regression line from (x_0, y_0) to (x_1, y_1).
    plt.plot([x_0, x_1], [y_0, y_1], c='r')

    # Label the graph axes.
    plt.ylabel("cp_new")
    plt.xlabel("cp")

    # Plot a scatter plot from our data sample.
    plt.scatter(sample["cp"], sample["cp_new"])

    # Display graph.
    plt.pause(0.01)
plt.show()