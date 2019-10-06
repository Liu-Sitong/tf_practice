import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pokemon_dataframe = pd.read_csv('pokemon.csv', sep=',')
pokemon_dataframe = pokemon_dataframe.reindex(np.random.permutation(pokemon_dataframe.index))

train_size = 50
iteration = 15000
lr = 1  # learning rate

X = tf.placeholder(tf.float32, [None, 3])
w = tf.Variable(tf.zeros([3, 1]))
w2 = tf.Variable(tf.zeros([3, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.sigmoid(tf.matmul(tf.square(X), w2) + tf.matmul(X, w) + b)
Y = tf.placeholder(tf.float32, [None, 1])

# 成本函数 sum(sqr(y_-y))/n
cost = tf.reduce_mean(tf.square(Y - y))

# 用梯度下降训练
train_step = tf.train.AdagradOptimizer(lr).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

x_train = pokemon_dataframe.head(train_size)[['cp','height','weight']].values
y_train = pokemon_dataframe.head(train_size)[['cp_new']].values
x_test = pokemon_dataframe.tail(75 - train_size)[['cp','height','weight']].values
y_test = pokemon_dataframe.tail(75 - train_size)[['cp_new']].values

history_cost = []

for i in range(iteration):
    sess.run(train_step, feed_dict={X: x_train, Y: y_train})
    # print(sess.run(w))
    history_cost.append(sess.run(cost, feed_dict={X: x_train, Y: y_train}))

w2_result = sess.run(w2)
w_result = sess.run(w)
b_result = sess.run(b)
print("w " + str(w_result))
print("w2 " + str(w2_result))
print("b " + str(b_result))
print("train cost:%f" % sess.run(cost, feed_dict={X: x_train, Y: y_train}))
print("test cost:%f" % sess.run(cost, feed_dict={X: x_test, Y: y_test}))

# print(history_w)
plt.plot(range(len(history_cost)), np.log(history_cost))
plt.title('change of cost')
plt.ylabel("log cost")
plt.xlabel("iteration")
# plt.savefig('cost_change.jpeg')
plt.show()

#
# plt.plot(x_test, y_test, '.r')
# plt.plot(x_train, y_train, '.b')
# plt.title('red:test; blue:train')
# maxx = max(max(x_train),max(x_test))[0] * 1.1
#
# x_space = np.linspace(0, maxx, maxx * 10)
# plt.ylabel("b")
# plt.xlabel("iteration")
# plt.plot(x_space, sess.run(y, feed_dict={X: [[xx] for xx in x_space]}))
# plt.savefig('regression.jpg')
# plt.show()
