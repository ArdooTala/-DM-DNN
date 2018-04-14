import tensorflow as tf
import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt
from Data_Log import *
from Plotter import *


#Inputs:
dataFolder = "/Users/Ardoo/Desktop/20180314_Robot-Data/"
logFile = 'Learn-Log_013'
logFolder = '/Users/Ardoo/Desktop/Learn-Log/'
Network = [30, 40, 50]
learning_rate = 0.00005
Training_Batch_Size = 500
Training_Steps = 500
Test_Batch_Size = 500


#Settings for the program.
pd.options.display.max_rows = 15
tf.logging.set_verbosity(tf.logging.ERROR)

#Read the Data and Create Sets.
df = read_Folder(dataFolder)
print (df.describe())
features_train, labels_train, features_test, labels_test = Bake_Data(df, 9000)

#Feature-Columns are defined Here.
feature_Columns = []
for key in features_train.keys():
    feature_Columns.append(tf.feature_column.numeric_column(key=key))

#Initiate the Classifier
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
clf = tf.estimator.DNNClassifier(
    optimizer=my_optimizer,
    hidden_units=Network,
    feature_columns=feature_Columns,
    n_classes=5
    )

with open(logFolder+logFile+'.txt', mode='w') as learn_Log:
    learn_Log.write(
"""Network = {0}
learning_rate = {1}
Training_Batch_Size = {2}
Training_Steps = {3}
Test_Batch-Size = {4}
""".format(
        Network,
        learning_rate,
        Training_Batch_Size,
        Training_Steps,
        Test_Batch_Size
        )
    )

    t0 = time()
    loss = []
    loss_t = []
    accs = []
    tacc = []

    for i in range(600):
    # i = 0
    # while True:

        try:
            #Train the Classifier
            clf.train(
                input_fn=lambda: train_In_fn(
                    features_train,
                    labels_train,
                    batchSize=Training_Batch_Size),
                steps=Training_Steps
                )

            #Evaluate the classifier
            eval_Result = clf.evaluate(
                input_fn=lambda: eval_In_fn(
                    features_test,
                    labels_test,
                    batchSize=Test_Batch_Size)
                )

            eval_Result_t = clf.evaluate(
                input_fn=lambda: eval_In_fn(
                    features_train,
                    labels_train,
                    batchSize=Test_Batch_Size)
                )


            accs.append(1 - eval_Result['accuracy'])
            loss.append(eval_Result['average_loss'])
            tacc.append(1 - eval_Result_t['accuracy'])
            loss_t.append(eval_Result_t['average_loss'])

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
                "[{0:03d}]----[{1:.3f}%]--[{2:.3f}%]----[{3:7.1f}\"]".format(
                    i,
                    eval_Result['accuracy']*100,
                    eval_Result_t['accuracy']*100,
                    time() - t0)
                )

            learn_Log.write(
                "[{0:03d}]----[{1:.3f}%]--[{2:.3f}%]----[{3:7.1f}\"]\n".format(
                    i,
                    eval_Result['accuracy']*100,
                    eval_Result_t['accuracy']*100,
                    time() - t0)
                )

            if i == 599:
                Plot_Learning(accs, tacc, loss, loss_t)
                PlotDone(logFolder, logFile)

        except KeyboardInterrupt:
            Plot_Learning(accs, tacc, loss, loss_t)
            PlotDone(logFolder, logFile)
            break
