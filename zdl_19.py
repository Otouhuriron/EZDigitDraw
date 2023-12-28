import sys, os
import numpy as np
from dataset.mnist import load_mnist
from zdl_18 import TwoLayerNet2

(t_train, x_train), (t_test, x_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = x_train.shape[0]

# ハイパーパラメータ
iters_num = 10000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 1epochあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet2(input_size=10, hidden_size=50, output_size=784)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) hispeed ver!

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * 101/(i + 1) * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # 1epochごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc |" + str(train_acc) + ", " + str(test_acc))


