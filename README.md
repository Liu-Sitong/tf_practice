I write these simple codes in this repo in the summer of 2018 to practice Tensorflow. It is inspired by online course of [Hung-yi Lee](<http://speech.ee.ntu.edu.tw/~tlkagk/>) and official Tensorflow [tutorials](<https://www.tensorflow.org/tutorials>).

The data I was studying was some data of Pokémon. Basically I simply attempted to use gradient descent to find the relationship between the origin and new cp value of a Pokémon.

## Linear function

cost change during the training:

![linear cost](一次函数结果/cost_change.jpg)

Result

![linear result](一次函数结果/regression.jpg)

## Quadratic function

cost change during the training:

![linear cost](二次函数结果/cost_change.jpg)

Result

![linear result](二次函数结果/regression.jpg)

## Cubic function

cost change during the training:

![linear cost](三次函数结果/cost_change.jpg)

Result (Blue points are training data and red points are testing data)

![linear result](三次函数结果/regression.jpg)



## Multiple dimensions

Use origin cp, height and weight.

cost change during the training:

![linear cost](多参数/cost_change.jpg)