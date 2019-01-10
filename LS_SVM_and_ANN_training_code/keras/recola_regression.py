# script written by Joris De Turck
# code is partially based on the scripts from the AVEC 2018 challenge found here:
# https://github.com/AudioVisualEmotionChallenge/AVEC2018
from __future__ import print_function
import arff
import numpy as np
import matplotlib.pyplot as plt
import csv
from keras import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
from keras.optimizers import RMSprop
from keras import regularizers

# read in the data using liac-arff


def get_features(partition):

    path = '/home/joris/Documents/RECOLA/features_audio/arousal/' + partition + "_"
    x_rec = None

    for i in range(1, 10):

        new_path = path + str(i) + ".arff"
        dataset = arff.load(open(new_path))
        data = np.array(dataset['data'])

        if i == 1:
            x_rec = data
        else:
            x_rec = np.concatenate((x_rec, data), axis=0)

    return x_rec


def get_arousal_labels(partition):

    path = '/home/joris/Documents/RECOLA/ratings_gold_standard/arousal/' + partition + "_"
    y_rec = None

    for i in range(1, 10):
        new_path = path + str(i) + ".arff"
        dataset = arff.load(open(new_path))
        data = np.array(dataset['data'])
        data = (data[:, -1:]).astype(np.float)

        if i == 1:
            y_rec = data
        else:
            y_rec = np.concatenate((y_rec, data), axis=0)

    return y_rec

def get_valence_labels(partition):

    path = '/home/joris/Documents/RECOLA/ratings_gold_standard/valence/' + partition + "_"
    y_rec = None

    for i in range(1, 10):
        new_path = path + str(i) + ".arff"
        dataset = arff.load(open(new_path))
        data = np.array(dataset['data'])
        data = (data[:, -1:]).astype(np.float)

        if i == 1:
            y_rec = data
        else:
            y_rec = np.concatenate((y_rec, data), axis=0)

    return y_rec


def load_data(target):

    x_train_rec = get_features(partition="train")      # read in the data and discard the first column, these are the timestamps
    x_train_rec = x_train_rec[:, 1:]

    x_dev_rec = get_features(partition="dev")
    x_dev_rec = x_dev_rec[:, 1:]

    if(target == 'valence'):
        y_train_rec = get_valence_labels(partition="train")        # for features this was already dones
        y_dev_rec = get_valence_labels(partition="dev")

    if target == 'arousal':                                        # target must be either valence or arousal
        y_train_rec = get_arousal_labels(partition="train")
        y_dev_rec = get_arousal_labels(partition="dev")

    # normalize labels and features with respect to training mean and std deviation

    mean_x_train = np.mean(x_train_rec, axis=0, dtype=np.float64)
    mean_y_train = np.mean(y_train_rec[:, 0], dtype=np.float64)
    std_x_train = np.std(x_train_rec, axis=0, dtype=np.float64)
    std_y_train = np.std(y_train_rec[:, 0], dtype=np.float64)

    for i in range(0, len(x_train_rec[0])):
        x_train_rec[:, i] = (x_train_rec[:, i] - mean_x_train[i]) / std_x_train[i]
        x_dev_rec[:, i] = (x_dev_rec[:, i] - mean_x_train[i]) / std_x_train[i]

    y_train_rec = (y_train_rec[:,0] - mean_y_train) / std_y_train
    y_dev_rec = (y_dev_rec[:,0] - mean_y_train) / std_y_train

    return x_train_rec, x_dev_rec, y_train_rec, y_dev_rec


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def train_model(nr_epochs, btch_size, nr_hidden_layers, layer_size, learning_rate, activ_fctn, x_train,
                x_dev, y_train, y_dev, drops, regul_kernel, i, j, k, l, save_path, plot=True):

    print('size of x_train = ' + str(len(x_train)) + 'x' + str(len(x_train[0])))
    print('size of x_dev = ' + str(len(x_dev)) + 'x' + str(len(x_dev[0])))
    print('size of y_train = ' + str(len(y_train)) + 'x1')
    print('size of y_dev = ' + str(len(y_dev)) + 'x1')

    model = Sequential()   # sequential model
    model.add(Dropout(drops, input_shape=(88,)))
    model.add(Dense(layer_size, activation=activ_fctn, kernel_regularizer=regularizers.l2(regul_kernel)))  # first hidden layer
    for idx in range(0, nr_hidden_layers-1):
        model.add(Dense(layer_size, activation=activ_fctn, kernel_regularizer=regularizers.l2(regul_kernel)))            # hidden layers

    model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(regul_kernel)))   # output layer
    model.summary()

    rmsprop = RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss=ccc_loss)

    performances = np.empty((nr_epochs, 6))
    history = []                   # Creating a empty list for holding the loss later

    ccc_max_raw = 0.0000           # keep track of best CCC to save corresponding predicted labels for plotting
    ccc_max_scaled = 0.0000
    ccc_max_smoothed = 0.0000

    yh_best_raw = np.empty(len(y_dev))
    yh_best_scaled = np.empty(len(y_dev))
    yh_best_smoothed = np.empty(len(y_dev)-100)

    best_raw = 0
    best_scaled = 0
    best_smoothed = 0

    ep = 1
    while ep < nr_epochs+1:

        result = model.fit(x_train, y_train, epochs=1, batch_size=btch_size)
        history.append(result.history['loss'])                                    # save training loss

        yh_train = model.predict(x_train)                                         # get prediction on training data
        yh_train = np.array(yh_train)
        ccc_train0 = calc_ccc(yh_train[:, 0], y_train)                            # save CCC of raw training prediction
        yh_train = yh_train/np.std(yh_train)
        ccc_train1 = calc_ccc(yh_train[:, 0], y_train)                            # save CCC for scaled training prediction
        yh_train = running_mean(yh_train, 101)
        ccc_train2 = calc_ccc(yh_train, y_train[50:-50])                          # save the CCC after smoothing and scaling for trainig prediction

        yh = model.predict(x_dev)                                                 # get prediction on development data
        yh = np.array(yh)
        ccc0 = calc_ccc(yh[:, 0], y_dev)                                          # save the CCC of the raw prediction
        yh_scaled = yh/np.std(yh)
        ccc1 = calc_ccc(yh_scaled[:, 0], y_dev)                                   # save the CCC of the scaled prediction
        yh_smoothed = running_mean(yh_scaled, 101)
        ccc2 = calc_ccc(yh_smoothed, y_dev[100:])                                 # save the CCC after scaling and smoothing

        if ccc_max_raw < ccc0:                                                    # check for max CCC, if new max is found, save corresponding pred labels
            ccc_max_raw = ccc0
            yh_best_raw = yh
            best_raw = ep

        if ccc_max_scaled < ccc1:                                                 # check for max CCC, if new max is found, save corresponding pred labels
            ccc_max_scaled = ccc1
            yh_best_scaled = yh_scaled
            best_scaled = ep

        if ccc_max_smoothed < ccc2:                                               # check for max CCC, if new max is found, save corresponding pred labels
            ccc_max_smoothed = ccc2
            yh_best_smoothed = yh_smoothed
            best_smoothed = ep

        performances[ep-1, 0] = ccc0         # ep-1 because ep starts counting from 1 while array indices start from 0
        performances[ep-1, 1] = ccc1
        performances[ep-1, 2] = ccc2
        performances[ep-1, 3] = ccc_train0
        performances[ep-1, 4] = ccc_train1
        performances[ep-1, 5] = ccc_train2

        print('############## EPOCH ------>>  ' + str(ep))
        print('CCC Training partition raw   = ' + str(ccc_train0))
        print('CCC Training part scaled     = ' + str(ccc_train1))
        print('CCC Training scaled + smooth = ' + str(ccc_train2))
        print('CCC Devel partition raw      = ' + str(ccc0))
        print('CCC Devel partition scaling  = ' + str(ccc1))
        print('CCC Dev  smoothing & scaling = ' + str(ccc2))
        print('____________________________________ ' + '\n')
        print('\n')

        ep += 1     # ep gets updated here

        if plot and (ep == (nr_epochs+1)):     # last iteration of the loop

            #save the performances of this configuration of hyper-params

            with open(save_path + 'perf' + str(i) + str(j) + str(k) + str(l) + '.csv', 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

                filewriter.writerow(['nr_epochs', nr_epochs])
                filewriter.writerow(['btch_size', btch_size])
                filewriter.writerow(['nr_hidden_layers', nr_hidden_layers])
                filewriter.writerow(['layer_size', layer_size])
                filewriter.writerow(['learning_rate', learning_rate])
                filewriter.writerow(['activ_fctn', activ_fctn])

                filewriter.writerow(['CCC train raw', performances[:, 3]])
                filewriter.writerow(['CCC train scaled', performances[:, 4]])
                filewriter.writerow(['CCC train smoothed', performances[:, 5]])
                filewriter.writerow(['CCC devel raw', performances[:, 0]])
                filewriter.writerow(['CCC devel scaled', performances[:, 1]])
                filewriter.writerow(['CCC devel smoothed', performances[:, 2]])

            xas = np.arange(len(yh))

            fig = plt.figure(1)
            plt.plot(xas, y_dev)
            plt.plot(xas, yh_best_raw)
            plt.title('Raw prediction on development data, best CCC=' + str(round(ccc_max_raw, 3)))
            fig.savefig(str(save_path + str(i) + str(j) + str(k) + str(l) + '#fig1.png'))
            plt.close(fig)

            fig = plt.figure(2)
            plt.plot(xas, y_dev)
            plt.plot(xas, yh_best_scaled)
            plt.title('Scaled prediction on devel data, best CCC=' + str(round(ccc_max_scaled, 3)))
            fig.savefig(str(save_path + str(i) + str(j) + str(k) + str(l) + '#fig2.png'))
            plt.close(fig)

            fig = plt.figure(3)
            plt.plot(xas[:-100], y_dev[50:-50])
            plt.plot(xas[:-100], yh_best_smoothed)
            plt.title('Scaled + smoothed prediction on devel data, best CCC=' + str(round(ccc_max_smoothed, 3)))
            fig.savefig(str(save_path + str(i) + str(j) + str(k) + str(l) + '#fig3.png'))
            plt.close(fig)

            xas2 = np.arange(nr_epochs+1)
            xas2 = xas2[1:]

            fig = plt.figure(4)
            plt.plot(xas2, performances[:, 0])
            plt.plot(xas2, performances[:, 1])
            plt.plot(xas2, performances[:, 2])
            plt.title('DEVEL CCCs: raw=' + str(round(ccc_max_raw, 3)) + ' @' + str(best_raw) + ', scaled=' + str(round(ccc_max_scaled, 3)) + ' @' + str(best_scaled)
                      + ', smoothed=' + str(round(ccc_max_smoothed, 3)) + ' @' + str(best_smoothed))

            fig.savefig(str(save_path + str(i) + str(j) + str(k) + str(l) + '#fig4.png'))
            plt.close(fig)

            fig = plt.figure(5)
            plt.plot(xas2, performances[:, 3])
            plt.plot(xas2, performances[:, 4])
            plt.plot(xas2, performances[:, 5])
            plt.title('CCCs of training data RUN = ' + str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
            fig.savefig(str(save_path + str(i) + str(j) + str(k) + str(l) + '#fig5.png'))
            plt.close(fig)

            fig = plt.figure(6)
            plt.plot(xas2, history)
            plt.title('Evolution of training loss')
            fig.savefig(str(save_path + str(i) + str(j) + str(k) + str(l) + '#fig6.png'))
            plt.close(fig)

            print('Just finished experiment # ' + str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
            print('Results saved in ' + str(save_path))

    return ccc_max_raw, ccc_max_scaled, ccc_max_smoothed, best_raw, best_scaled, best_smoothed, yh_best_raw, yh_best_scaled, yh_best_smoothed



def ccc_loss(gold, pred):  # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics

    # input (num_batches, seq_len, 1)
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1, keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    ccc_loss   = K.constant(1.) - ccc
    return ccc_loss


def calc_ccc(x, y):                                   # Function to calculate the CCC (=concordance correlation coefficient)

    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean(np.multiply((x - x_mean), (y - y_mean)))

    x_var = 1.0 / (len(x) - 1) * np.nansum((x - x_mean) ** 2)  # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    y_var = 1.0 / (len(y) - 1) * np.nansum((y - y_mean) ** 2)

    ccc = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    return ccc

def print_data_dimensions(x_train_rec, x_dev_rec, y_train_rec, y_dev_rec):
    print("shape of x_train_rec = " + str(x_train_rec.shape))  # review that the sizes of the data matrices are correct
    print("shape of x_dev_rec = " + str(x_dev_rec.shape))
    print("shape of y_train_rec = " + str(y_train_rec.shape))
    print("shape of y_dev_rec = " + str(y_dev_rec.shape))
    print("mean of x_train_rec = " + str(np.mean(x_train_rec, axis=0, dtype=np.float64)))  # review that the sizes of the data matrices are correct
    print("mean of x_dev_rec = " + str(np.mean(x_dev_rec, axis=0, dtype=np.float64)))
    print("mean of y_train_rec = " + str(np.mean(y_train_rec, dtype=np.float64)))
    print("mean of y_dev_rec = " + str(np.mean(y_dev_rec, dtype=np.float64)))
    print("std of x_train_rec = " + str(np.std(x_train_rec, axis=0, dtype=np.float64)))  # review that the sizes of the data matrices are correct
    print("std of x_dev_rec = " + str(np.std(x_dev_rec, axis=0, dtype=np.float64)))
    print("std of y_train_rec = " + str(np.std(y_train_rec, dtype=np.float64)))
    print("std of y_dev_rec = " + str(np.std(y_dev_rec, dtype=np.float64)))
    print("this is what y_train_rec looks like:")
    print(str(y_train_rec[0:5]))
    print("this is what y_dev_rec looks like:")
    print(str(y_dev_rec[0:5]))
    print("this is what x_train_rec looks like:")
    print(str(x_train_rec[0:5, 0:5]))
    print("this is what x_dev_rec looks like:")
    print(str(x_dev_rec[0:5, 0:5]))


def main():

    # target = 'arousal'
    target = 'valence'

    x_train_rec, x_dev_rec, y_train_rec, y_dev_rec = load_data(target=target)   # read in the data

    # print out data dimensions and some elements, for debugging purposes
    print_data_dimensions(x_train_rec=x_train_rec, x_dev_rec=x_dev_rec, y_train_rec=y_train_rec, y_dev_rec=y_dev_rec)

    # variables to track performance
    best_of_best_raw = [0.0, 0, 0, 0, 0, 0]           # list to hold information about best prediction
    best_of_best_scaled = [0.0, 0, 0, 0, 0, 0]        # format = [CCC, i, j, k, l, #epoch]
    best_of_best_smoothed = [0.0, 0, 0, 0, 0, 0]      #          [0    1  2  3  4    5   ]

    yh_best_raw = np.empty(len(y_dev_rec))
    yh_best_scaled = np.empty(len(y_dev_rec))
    yh_best_smoothed = np.empty(len(y_dev_rec) - 100)

    # path to save all plots (6 plots per neural net configuration) + hyperparameters + csv files with performance
    save_path = '/home/joris/Pictures/Runs/RECOLA/VALENCE/val6/'

    # defaults                   #######################################################################################################
    nr_epochs = 40               # model = Sequential()  , 1 input+hidden layer Dense() with input dimension 88 and <layer_size> nodes #
    learning_rate = 0.001        # + <nr_layers_list-1> EXTRA hidden layers Dense() with activ_function = <activ_function>             #
    activ_fctn = 'relu'          # + 1 output layer Dense() with ativ_function = 'linear'                                              #
                                 #######################################################################################################
    # best CCC = 0.557 @ 0.0.1.4 = 2 , 8, 5000, 0.003


    # grid search hyperparamters
    nr_layers_list = 3 #, 4]  # [1, 2, 3, 4]    -> 2 nr layers
    layer_size_list = [8, 16, 32, 64]  # , 16, 64]       # [8, 16, 32, 64] -> 4 layer sizes
    batch_list = [2500, 5000, 10000]
    drops = [0.5]   #, 0.0001, 0.001]   # [100, 300, 1000]  ->4 batches
    regul_list_kernel = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]    # [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1] -> 5 regs -> 7*5*3 = 105 experiments

    for i in range(0, len(layer_size_list)):
        for j in range(0, len(batch_list)):
            for k in range(0, len(drops)):
                for l in range(0, len(regul_list_kernel)):
                    ccc_max_raw, ccc_max_scaled, ccc_max_smoothed, best_raw, best_scaled, best_smoothed, yh_raw, yh_scaled, yh_smoothed = \
                    train_model(nr_epochs=nr_epochs, nr_hidden_layers=nr_layers_list, layer_size=layer_size_list[i],  btch_size=batch_list[j],
                        learning_rate=learning_rate, activ_fctn=activ_fctn, x_train=x_train_rec, x_dev=x_dev_rec,
                        y_train=y_train_rec, y_dev=y_dev_rec, drops=drops[k], regul_kernel=regul_list_kernel[l], i=i, j=j, k=k, l=l, save_path=save_path, plot=True)

                    if best_of_best_raw[0] < ccc_max_raw:
                        best_of_best_raw[0] = ccc_max_raw
                        best_of_best_raw[5] = best_raw
                        best_of_best_raw[1:4] = [i, j, k, l]
                        yh_best_raw = yh_raw

                    if best_of_best_scaled[0] < ccc_max_scaled:
                        best_of_best_scaled[0] = ccc_max_scaled
                        best_of_best_scaled[5] = best_scaled
                        best_of_best_scaled[1:4] = [i, j, k, l]
                        yh_best_scaled = yh_scaled

                    if best_of_best_smoothed[0] < ccc_max_smoothed:
                        best_of_best_smoothed[0] = ccc_max_smoothed
                        best_of_best_smoothed[5] = best_smoothed
                        best_of_best_smoothed[1:4] = [i, j, k, l]
                        yh_best_smoothed = yh_smoothed

    # save hyper-parameters in csv file
    with open(save_path + 'hyper_params.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        filewriter.writerow(['nr_layers', nr_layers_list])
        filewriter.writerow(['layer_size', layer_size_list])
        filewriter.writerow(['batch_size', batch_list])
        filewriter.writerow(['drops', drops])
        filewriter.writerow(['regul_list_kernel', regul_list_kernel])

    # now plot the best performing predictions in terms of CCC, for raw, scaled and raw+smoothed labels
    xas = np.arange(len(y_dev_rec))

    fig = plt.figure(1)
    plt.plot(xas, y_dev_rec)
    plt.plot(xas, yh_best_raw)
    plt.title('Max CCC raw=' + str(round(best_of_best_raw[0], 3)) + ' @' + str(best_of_best_raw[1]) + '.' + str(best_of_best_raw[2]) + '.' + str(best_of_best_raw[3])
              + '.' + str(best_of_best_raw[4]) + ' Epoch=' + str(best_of_best_raw[5]))
    fig.savefig(str(save_path + 'best_raw_' + str(best_of_best_raw[1]) + '.' + str(best_of_best_raw[2]) + '.' + str(best_of_best_raw[3])
              + '.' + str(best_of_best_raw[4]) + '.png'))

    fig = plt.figure(2)
    plt.plot(xas, y_dev_rec)
    plt.plot(xas, yh_best_scaled)
    plt.title('Max CCC scaled=' + str(round(best_of_best_scaled[0], 3)) + ' @' + str(best_of_best_scaled[1]) + '.' + str(best_of_best_scaled[2]) + '.' + str(best_of_best_scaled[3])
     + '.' + str(best_of_best_scaled[4]) + ' Epoch=' + str(best_of_best_scaled[5]))
    fig.savefig(str(save_path + 'best_scaled_' + str(best_of_best_raw[1]) + '.' + str(best_of_best_raw[2]) + '.' + str(best_of_best_raw[3])
              + '.' + str(best_of_best_raw[4])+ '.png'))

    fig = plt.figure(3)
    plt.plot(xas[:-100], y_dev_rec[100:])
    plt.plot(xas[:-100], yh_best_smoothed)
    plt.title('Max CCC scaled+smooth=' + str(round(best_of_best_smoothed[0], 3)) + ' @' + str(best_of_best_smoothed[1]) + '.' + str(best_of_best_smoothed[2]) + '.'
     + str(best_of_best_smoothed[3]) + '.' + str(best_of_best_smoothed[4]) + ' Epoch=' + str(best_of_best_smoothed[5]))
    fig.savefig(str(save_path + 'best_smooth_' + str(best_of_best_raw[1]) + '.' + str(best_of_best_raw[2]) + '.' + str(best_of_best_raw[3])
              + '.' + str(best_of_best_raw[4])+ '.png'))

    # shift best prediction 100 frames forward with respect to ground truth


    plt.show()

    return


if __name__ == '__main__':
    main()