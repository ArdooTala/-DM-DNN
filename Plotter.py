from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

def pred_In_fn(features, labels):
    #This is the input function for evaluating the DNN.
    #Convert to Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    #Shuffle and batch the Dataset before returning.
    return dataset.batch(1)

def Plot_Learning(accs, tacc, loss, loss_t):
    plt.figure(figsize=(20,10))

    plt.subplot(2,2,4)
    #plt.ylim(0, .2)
    plt.plot(range(len(accs)), accs, c='C0', label='Test-Data')
    plt.plot(range(len(tacc)), tacc, c='C1', label='Train-Data')
    plt.legend()

    plt.subplot(2,2,2)
    # plt.ylim(0, 2)
    plt.plot(range(len(loss)), loss, c='C0', label='Test-Data')
    plt.plot(range(len(loss_t)), loss_t, c='C1', label='Train-Data')
    plt.legend()

def Plot_Result(clf, features, labels, count):
    plt.subplot(1,2,1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    predictions = clf.predict(
        input_fn=lambda: pred_In_fn(
            features[:count],
            labels[:count]
            )
        )
    labs = dict(labels)
    for i, j, k in zip(predictions, labs['tar1'], labs['tar2']):
        # print (j, '>>>>>>>>>>>>', i['predictions'])
        # plt.scatter(j, i['predictions'][0], c='b', s=2)
        # plt.scatter(k, i['predictions'][1], c='b', s=2)
        plt.plot(
            [j, i['predictions'][0]],
            [k, i['predictions'][1]],
            linewidth=1)

def PlotDone(logFolder, logFile):
    plt.savefig(logFolder+logFile+'.png')
    plt.show()
