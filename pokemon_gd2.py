import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pokemon_dataframe = pd.read_csv('pokemon.csv', sep=',')
pokemon_dataframe = pokemon_dataframe.reindex(np.random.permutation(pokemon_dataframe.index))

train_size = 50
iteration = 1000
# para_num =

X = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.zeros([1, 1]))
# w2 = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
Y = tf.placeholder(tf.float32, [None, 1])

# 成本函数 sum(sqr(y_-y))/n
cost = tf.reduce_mean(tf.square(Y - y))

# 用梯度下降训练
train_step = tf.train.AdagradOptimizer(5).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

x_train = [[a] for a in pokemon_dataframe.head(train_size)['cp']]
y_train = [[a] for a in pokemon_dataframe.head(train_size)['cp_new']]
x_test = [[a] for a in pokemon_dataframe.tail(75 - train_size)['cp']]
y_test = [[a] for a in pokemon_dataframe.tail(75 - train_size)['cp_new']]

history_w = []
history_b = []
history_cost = []

for i in range(iteration):
    sess.run(train_step, feed_dict={X: x_train, Y: y_train})
    # print(sess.run(w))
    history_w.append(sess.run(w)[0])
    history_b.append(sess.run(b)[0])
    history_cost.append(sess.run(cost, feed_dict={X: x_train, Y: y_train}))
w_result = sess.run(w)
b_result = sess.run(b)
print("w:%f" % w_result)
print("b:%f" % b_result)
print("train cost:%f" % sess.run(cost, feed_dict={X: x_train, Y: y_train}))
print("test cost:%f" % sess.run(cost, feed_dict={X: x_test, Y: y_test}))

# print(history_w)
plt.plot(range(len(history_cost)), history_cost)
plt.title('change of cost')
plt.ylabel("cost")
plt.xlabel("iteration")
plt.savefig('cost_change.jpg')
plt.show()

plt.plot(x_test, y_test, '.')
maxx = max(x_train)[0] * 1.1
plt.ylabel("b")
plt.xlabel("iteration")
plt.plot([0, maxx], [b_result, maxx * w_result + b_result])
plt.savefig('regression.jpg')
plt.show()
