# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import sys
import cv2
import copy
import csv
import pylab as plt

plt.style.use('ggplot')

def cross_split(data, label, k_cross, n, perm, N, length):
    x, y = [], []
    for i in range(k_cross):
        x.append(mnist.data[perm[i*N/cross:(i+1)*N/cross]])
        y.append(mnist.target[perm[i*N/cross:(i+1)*N/cross]])
    x_train, y_train = [], []
    for i in range(k_cross):
        if i == n:
            x_test = copy.deepcopy(x[i])
            y_test = copy.deepcopy(y[i])
        else:
            x_train.append(copy.deepcopy(x[i]))
            y_train.append(copy.deepcopy(y[i]))
    x_train = np.array(x_train).reshape(N*(k_cross-1)/k_cross, 1, length, length)
    y_train = np.array(y_train).reshape(N*(k_cross-1)/k_cross)
    x_test = x_test.reshape(N/k_cross, 1, length, length)
    return copy.deepcopy(x_train), copy.deepcopy(x_test), copy.deepcopy(y_train), copy.deepcopy(y_test)


def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.max_pooling_2d(F.relu(model.conv1(x)), ksize=2, stride=2)
    h2 = F.max_pooling_2d(F.relu(model.conv2(h1)), ksize=3, stride=3)
    h3 = F.dropout(F.relu(model.l3(h2)), train=train)
    y  = model.l4(h3)
    
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def cross_optimizers(opt):
    if opt == 'SGD':
        optimizer = optimizers.SGD()
    elif opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD()
    elif opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad()
    elif opt == 'RMSprop':
        optimizer = optimizers.RMSprop()
    elif opt == 'AdaDelta':
        optimizer = optimizers.AdaDelta()
    elif opt == 'Adam':
        optimizer = optimizers.Adam()
    return copy.deepcopy(optimizer)


if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    mnist.data = mnist.data.astype(np.float32)
    mnist.target = mnist.target.astype(np.int32)
    mnist.data /= mnist.data.max()
    n_epoch = 50
    batchsize = 100
    cross = 5   # 5 Cross Validation
    optimizer_list = ['SGD', 'MomentumSGD', 'AdaGrad', 'RMSprop', 'AdaDelta', 'Adam']   # Optimizers List

    opt_train_loss = []
    opt_train_acc  = []
    opt_test_loss = []
    opt_test_acc  = []
    N, imagesize = mnist.data.shape
    length = int(np.sqrt(imagesize))
    cross_perm = np.random.permutation(N)
    for opt in optimizer_list:
        print '========================='
        print 'Set Optimizer : ' + opt
        cross_acc_sum = 0
        cross_train_loss = []
        cross_train_acc  = []
        cross_test_loss = []
        cross_test_acc  = []
        for k in range(cross):
            print '-------------------------'
            print 'Cross Validation : ' + str(k + 1)
            model = FunctionSet(
                conv1 = F.Convolution2D(1, 28, 5),
                conv2 = F.Convolution2D(28, 28, 5),
                l3 = F.Linear(252, 252),
                l4 = F.Linear(252, 10) )
            optimizer = cross_optimizers(opt)
            optimizer.setup(model)
            x_train, x_test, y_train, y_test = cross_split(mnist.data, mnist.target, cross, k, cross_perm, N, length)
            N_train = x_train.shape[0]
            cross_acc = 0
            train_loss = []
            train_acc  = []
            test_loss = []
            test_acc  = []
            for epoch in range(1, n_epoch+1):
                print 'epoch' + str(epoch)
                loss_sum, acc_sum = 0, 0
                perm = np.random.permutation(N_train)
                for i in range(0, N_train, batchsize):
                    x_batch = x_train[perm[i:i+batchsize]]
                    y_batch = y_train[perm[i:i+batchsize]]
                    optimizer.zero_grads()
                    loss, acc = forward(x_batch, y_batch)
                    loss.backward()
                    optimizer.update()
                    real_batchsize = len(x_batch)
                    loss_sum += float(cuda.to_cpu(loss.data)) * real_batchsize
                    acc_sum += float(cuda.to_cpu(acc.data)) * real_batchsize
                print 'Train Mean Loss={}, Accuracy={}'.format(loss_sum / N_train, acc_sum / N_train)
                train_loss.append(loss_sum / N_train)
                train_acc.append(1 - (acc_sum / N_train))
                N_test = x_test.shape[0]
                loss_sum, acc_sum = 0, 0
                for i in range(0, N_test):
                    x_batch = x_train[i].reshape(1, 1, length, length)
                    y_batch = np.array(y_train[i]).reshape(1)
                    loss, acc = forward(x_batch, y_batch)
                    loss_sum += float(cuda.to_cpu(loss.data))
                    acc_sum += float(cuda.to_cpu(acc.data))
                print 'Test Mean Loss={}, Accuracy={}'.format(loss_sum / N_test, acc_sum / N_test)
                test_loss.append(loss_sum / N_test)
                test_acc.append(1 - (acc_sum / N_test))
                if cross_acc <= acc_sum / N_test:
                    cross_acc = acc_sum / N_test
            cross_acc_sum += cross_acc
            cross_train_loss.append(train_loss)
            cross_train_acc.append(train_acc)
            cross_test_loss.append(test_loss)
            cross_test_acc.append(test_acc)            
        print '====Cross Validation===='
        print opt + ' 5 Cross Validation Mean Accuracy : ' + str(cross_acc_sum / cross)
        opt_train_loss.append(cross_train_loss)
        opt_train_acc.append(cross_train_acc)
        opt_test_loss.append(cross_test_loss)
        opt_test_acc.append(cross_test_acc)


    f = open('opt_train_loss.csv', 'ab')
    csvWriter = csv.writer(f)
    csvWriter.writerow(opt_train_loss)
    f.close()
    f = open('opt_train_acc.csv', 'ab')
    csvWriter = csv.writer(f)
    csvWriter.writerow(opt_train_acc)
    f.close()
    f = open('opt_test_loss.csv', 'ab')
    csvWriter = csv.writer(f)
    csvWriter.writerow(opt_test_loss)
    f.close()
    f = open('opt_test_acc.csv', 'ab')
    csvWriter = csv.writer(f)
    csvWriter.writerow(opt_test_acc)
    f.close()

    mean_opt_train_loss = []
    mean_opt_train_acc = []
    mean_opt_test_loss = []
    mean_opt_test_acc = []
    for i in range(len(optimizer_list)):
        m1, m2, m3, m4 = [], [], [], []
        for j in range(epoch):
            motrl = 0
            motra = 0
            motel = 0
            motea = 0
            for k in range(cross):
                motrl += opt_train_loss[i][k][j]
                motra += opt_train_acc[i][k][j]
                motel += opt_test_loss[i][k][j]
                motea += opt_test_acc[i][k][j]
            m1.append(motrl / cross)
            m2.append(motra / cross)
            m3.append(motel / cross)
            m4.append(motea / cross)
        mean_opt_train_loss.append(m1)
        mean_opt_train_acc.append(m2)
        mean_opt_test_loss.append(m3)
        mean_opt_test_acc.append(m4)

    # 誤差をグラフ描画
    plt.figure(figsize=(8,6))
    for i in range(len(optimizer_list)):
        plt.plot(range(len(mean_opt_train_loss[i])), mean_opt_train_loss[i])
        plt.plot(range(len(mean_opt_test_loss[i])), mean_opt_test_loss[i])
    plt.legend(optimizer_list,loc=4)
    plt.title("Each Optimizer of Train_Loss and Test_Loss")
    plt.plot()
    figname = 'CNN_Optimizer_Error_epoch_' + str(n_epoch) + '.pdf'
    plt.savefig(figname)
    
    # 誤差率をグラフ描画
    plt.figure(figsize=(8,6))
    for i in range(len(optimizer_list)):
        plt.plot(range(len(mean_opt_train_acc[i])), mean_opt_train_acc[i])
        plt.plot(range(len(mean_opt_test_acc[i])), mean_opt_test_acc[i])
    plt.legend(optimizer_list,loc=4)
    plt.title("Each Optimizer of Train_ErrorRate and Test_ErrorRate")
    plt.plot()
    figname = 'CNN_Optimizer_ErrorRate_epoch_' + str(n_epoch) + '.pdf'
    plt.savefig(figname)        
