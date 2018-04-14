import math
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os



def read_Values(path, labels_Path, normalize=True):
    """
    Read a file and return the Resistance-Data as a pandas.DataFrame.
    """
    values = []
    with open(path, 'r') as logFile, open(labels_Path, 'r') as labFile:
        labs = labFile.readlines()
        for line in logFile:
            line = line.rstrip()
            if len(line) < 1: continue
            if line[-1] == 'G':
                #Recognize the data by cheking the line-Flag in the end.
                line_Data = line.split('S')
                data_index = line_Data[0]
                vals = [int(val) for val in line_Data[1][:-1].split(',') if val and int(val) != 0]

                if normalize:
                    #Normalize the data:
                    """
                    minV = min(vals)
                    maxV = max(vals)
                    vals = [(val - minV) / (maxV - minV) for val in vals]
                    """
                    vals = [(val - 50) / 110 for val in vals]
                vals += labs[math.floor(int(data_index) / 10)].rstrip().split(',')[1:]

                values.append(vals)

    df = pd.DataFrame(
        values, columns=['val'+str(i) for i in range(len(values[0])-3)] + ['tar1', 'tar2', 'tar3']
        ).astype('float64')

    df['tar3'] -= 15
    df['tar3'] /= 35
    df['tar3'] *= 4

    df['tar3'] = df['tar3'].astype('int32')



    return df


def read_Folder(path, normalize=True):
    """
    Read all .txt files in a folder and combine them into one Pandas DataFrame.
    """
    df_t = pd.DataFrame()

    for root, dirs, files in os.walk(path):
        for file_Name in files:
            #print (file_Name)
            if file_Name[0] != 'R': continue
            print ("Reading File:   %s" %file_Name)
            dfTemp = read_Values(
                os.path.join(root, file_Name),
                os.path.join(root, dirs[0], "DataCollection-180320-Seed%s.TXT" %file_Name[-6:-4]),
                normalize
                )
            df_t = pd.concat([df_t, dfTemp], ignore_index = True)

    df_t = df_t.reindex(np.random.permutation(df_t.index))
    #print(df_t)
    return df_t


def Bake_Data(df, size):
    data = df.iloc[:, :12]
    labels = df.iloc[:, 12:14]
    normalizedData = (data - data.min().min()) / (data.max().max() - data.min().min())
    train_data = normalizedData.iloc[:size]
    test_data = normalizedData.iloc[size:]
    train_labels = df.iloc[:size, 12:14]
    test_labels = df.iloc[size:, 12:14]
    print(train_data.describe())
    print(test_data.describe())

    return train_data, train_labels, test_data, test_labels



def train_In_fn(features, labels, batchSize=None):
    #This is the input function for training the DNN.
    #Convert to Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    #Shuffle and batch the Dataset before returning.
    inp = dataset.repeat().shuffle(14000).batch(batchSize).make_one_shot_iterator().get_next()
    return inp

def eval_In_fn(features, labels, batchSize=None):
    #This is the input function for evaluating the DNN.
    #Convert to Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    #Shuffle and batch the Dataset before returning.
    return dataset.shuffle(5000).batch(batchSize).make_one_shot_iterator().get_next()

def pred_In_fn(features, labels):
    #This is the input function for predictions of the DNN.
    #Convert to Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    #Shuffle and batch the Dataset before returning.
    return dataset.batch(1)




def Visualize(df):
    plt.figure(figsize=(20,10))
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.xlim(0, 1)
        #plt.ylim(0, 1)
        x = df['val'+str(i)]
        y = df['tar3']
        plt.scatter(x, y, c='C0', s=1)

    plt.show()



if __name__ == '__main__':
    pd.options.display.max_rows = 15
    df = read_Folder("/Users/soroush/Google Drive/++ Digital Matter/[DM] Shared Media /024_Scripts/20180320/", True)
    df.describe()
    trF, trL, tsF, tsL = Bake_Data(df, 9000)
    print("#############")
    print(trF.describe())
    print(trL.describe())
    print(tsF.describe())
    print(tsL.describe())
    print("#############")
    Visualize(df)
