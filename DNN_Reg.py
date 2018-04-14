import tensorflow as tf
import numpy as np
import pandas as pd
import math
from time import time
from matplotlib import pyplot as plt
from Data_Log import *
from Plotter import *


#Inputs:
dataFolder = "/Users/Ardoo/Desktop/20180320"
logFile = 'Learn-Log_031'
logFolder = '/Users/Ardoo/Desktop/Learn-Log/'
Network = [200,100,50]
learning_rate = 3e-03
Training_Batch_Size = 2000
Training_Steps = 100
Test_Batch_Size = 1000
iterations = 1000
regularization = .7
bucketSize = 0.02


#Settings for the program.
pd.options.display.max_rows = 15
tf.logging.set_verbosity(tf.logging.ERROR)

#Read the Data and Create Sets.
df = read_Folder(dataFolder, normalize=False)
features_train, labels_train, features_test, labels_test = Bake_Data(df, 10000)

#Feature-Columns are defined Here.
feature_Columns = []
for key in features_train.keys():
    fc = tf.feature_column.numeric_column(key=key)
    fcb = tf.feature_column.bucketized_column(
        source_column=fc,
        boundaries=list(np.arange(0, 1, bucketSize))
        )
    feature_Columns.append(fc)
    feature_Columns.append(fcb)

#Initiate the Classifier
my_optimizer=tf.train.ProximalAdagradOptimizer(
    learning_rate=learning_rate,
    l2_regularization_strength=regularization
    )

reg = tf.estimator.DNNRegressor(
    optimizer=my_optimizer,
    hidden_units=Network,
    feature_columns=feature_Columns,
    label_dimension=2
    )

with open(logFolder+logFile+'.txt', mode='w') as learn_Log:
    learn_Log.write(
"""Network = {0}
learning_rate = {1}
Training_Batch_Size = {2}
Training_Steps = {3}
Test_Batch-Size = {4}
Regularization_Rate = {5}
Bucket_Size = {6}
""".format(
        Network,
        learning_rate,
        Training_Batch_Size,
        Training_Steps,
        Test_Batch_Size,
        regularization,
        bucketSize
        )
    )

    t0 = time()
    loss = []
    loss_t = []
    accs = []
    tacc = []

    for i in range(iterations):

        try:
            #Train the Classifier
            reg.train(
                input_fn=lambda: train_In_fn(
                    features_train,
                    labels_train,
                    batchSize=Training_Batch_Size),
                steps=Training_Steps
                )

            #Evaluate the classifier
            eval_Result = reg.evaluate(
                input_fn=lambda: eval_In_fn(
                    features_test,
                    labels_test,
                    batchSize=Test_Batch_Size)
                )

            eval_Result_t = reg.evaluate(
                input_fn=lambda: eval_In_fn(
                    features_train,
                    labels_train,
                    batchSize=Test_Batch_Size)
                )


            accs.append(eval_Result['loss'])
            loss.append(math.sqrt(eval_Result['average_loss']))
            tacc.append(eval_Result_t['loss'])
            loss_t.append(math.sqrt(eval_Result_t['average_loss']))

            #Print the results and write to the file.
            if i == 0:
                print ("\n[ # ]----[     Accuracy     ]----[  Time  ]")
                learn_Log.write(
                    "\n[ # ]----[     Accuracy     ]----[  Time  ]\n"
                    )

                print ("[   ]----[ Tests ]--[ Train ]----[        ]")
                learn_Log.write(
                    "[   ]----[Test   ]--[  Train]----[        ]\n"
                    )

            print (
                "[{0:03d}]----[{1:7.5f}]--[{2:7.5f}]----[{3:7.1f}\"]".format(
                    i,
                    math.sqrt(eval_Result['average_loss']),
                    math.sqrt(eval_Result_t['average_loss']),
                    time() - t0
                    )
                )

            learn_Log.write(
                "[{0:03d}]----[{1:7.5f}]--[{2:7.5f}]----[{3:7.1f}\"]\n".format(
                    i,
                    math.sqrt(eval_Result['average_loss']),
                    math.sqrt(eval_Result_t['average_loss']),
                    time() - t0
                    )
                )

            if i == iterations-1:
                Plot_Learning(loss, loss_t, accs, tacc)
                Plot_Result(reg, features_test, labels_test, 200)
                PlotDone(logFolder, logFile)

        except KeyboardInterrupt:
            Plot_Learning(loss, loss_t, accs, tacc)
            Plot_Result(reg, features_test, labels_test, 200)
            PlotDone(logFolder, logFile)
            break
