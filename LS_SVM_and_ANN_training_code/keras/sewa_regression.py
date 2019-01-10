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

def load_mfcc(partition):

    mfcc = None

    path = "/home/joris/Documents/AVEC2018_CES/audio_features_mfcc/" + partition + "_"

    if partition == "Train_DE":
        tot = 34
    else:
        tot = 14

    for i in range(0, tot):
        f = read_csv(path + str(i + 1).zfill(2) + '.csv', delim=';', skip_header=True)

        rest = 10 - len(f[:, 0])%10    # the amount of feature vectors for each recording must be a multiple of 10

        if rest != 10:
            if rest != 0:
                xtra_rows = np.empty((rest, 39))
                f = np.concatenate((f, xtra_rows), axis=0)

        # now drop last 300 rows to compensate for annotation delay

        f = f[:-300, :]

        if i == 0:                # if this is the first file we set mfcc equal to f
            mfcc = f
        else:                     # else we concatenate f and mfcc along axis 0
            mfcc = np.concatenate((mfcc, f), axis=0)

    return mfcc

def load_features(partition):
    features = None

    path = "/home/joris/Documents/AVEC2018_CES/audio_features_mfcc/" + partition + "_"
    path2 = "/home/joris/Documents/AVEC2018_CES/audio_features_egemaps/" + partition + "_"
    path3 = "/home/joris/Documents/AVEC2018_CES/audio_features_xbow/" + partition + "_"

    if partition == "Train_DE":
        tot = 34
    else:
        tot = 14

    for i in range(0, tot):
        f = read_csv(path + str(i + 1).zfill(2) + '.csv', delim=';', skip_header=True)

        g = read_csv(path2 + str(i + 1).zfill(2) + '.csv', delim=';', skip_header=True)

        h = read_csv(path3 + str(i + 1).zfill(2) + '.csv', delim=';', skip_header=True)
        h2 = np.empty((len(h)*10, 100))
        n = 0
        for nk in range(0, len(h2)):
            if (nk % 10 == 0) and (nk != 0):
                n += 1
            h2[nk, :] = h[n, :]

        diff = len(f)-len(g)
        diff2 = len(f)-len(h2)

        if diff != 0:
            if diff > 0:
                xtra_rows = np.empty((diff, 23))
                for ix in range(0, diff):
                    xtra_rows[ix, :] = g[-1, :]
                g = np.concatenate((g, xtra_rows), axis=0)
            else:
                g = g[:-diff, :]
        f = np.concatenate((f, g), axis=1)

        if diff2 != 0:
            if diff2 > 0:
                xtr = np.empty((diff2, 100))
                for r in range(0, diff2):
                    xtr[r, :] = h2[-1, :]
                h2 = np.concatenate((h2, xtr), axis=0)
            else:
                h2 = h2[:-diff2, :]
        f = np.concatenate((f, h2), axis=1)

        # len(f) should always be a multiple of 10 to match with the labels
        extra = 10-len(f) % 10
        if extra != 10:
            extra_rows = np.empty((extra, 162))
            for j in range(0, extra):
                extra_rows[j, :] = f[-1, :]
            f = np.concatenate((f, extra_rows), axis=0)     # resize f so it's a multiple of 10
        if i == 0:
            features = f[:-300, :]                          # drop last 3 seconds
        else:
            features = np.concatenate((features, f[:-300, :]), axis=0)
    return features


def load_labels(partition):
    labels = None
    path = "/home/joris/Documents/AVEC2018_CES/labels/" + partition + "_"
    if partition == "Train_DE":
        tot = 34
    else:
        tot = 14

    for i in range(0, tot):
        f = read_csv(path + str(i + 1).zfill(2) + '.csv', delim=';', skip_header=False)
        if i == 0:
            labels = f[30:, :]                              # drop first 3 seconds
        else:
            labels = np.concatenate((labels, f[30:, :]), axis=0)
    return labels


def calc_ccc(x, y):                          # Function to calculate the CCC (=concordance correlation coefficient)

    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean(np.multiply((x - x_mean),(y - y_mean)))

    x_var = 1.0 / (len(x) - 1) * np.nansum((x - x_mean) ** 2)  # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    y_var = 1.0 / (len(y) - 1) * np.nansum((y - y_mean) ** 2)

    ccc = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    return ccc


# Helper functions
def get_num_lines(filename, skip_header=False):
    with open(filename, 'r') as file:
        c = 0
        if skip_header:
            c = -1
        for line in file:
            c += 1
    return c


def get_num_columns(filename, delim=';', skip_header=False):
    # Returns the number of columns in a csv file
    # First two columns must be 'instance name' and 'timestamp' and are not considered in the output
    with open(filename, 'r') as file:
        if skip_header:
            next(file)
        line = next(file)
        offset1 = line.find(delim) + 1
        offset2 = line[offset1:].find(delim) + 1 + offset1
        cols = np.fromstring(line[offset2:], dtype=float, sep=delim)
    return len(cols)


def read_csv(filename, delim=';', skip_header=False):
    # Returns the content of a csv file (delimiter delim, default: ';')
    # First two columns must be 'instance name' and 'timestamp' and are not considered in the output, header is skipped if skip_header=True
    num_lines = get_num_lines(filename, skip_header)
    data = np.empty((num_lines, get_num_columns(filename, delim, skip_header)), float)
    with open(filename, 'r') as file:
        if skip_header:
            next(file)
        c = 0
        for line in file:
            offset1 = line.find(delim) + 1
            offset2 = line[offset1:].find(delim) + 1 + offset1
            data[c, :] = np.fromstring(line[offset2:], dtype=float, sep=delim)
            c += 1
    return data


def load_data():

    # custom function which does all preprocessing as well

    y_orig_train = load_labels(partition='Train_DE')  # read in training labels
    y_train = np.empty((len(y_orig_train) * 10, 3))
    j = 0
    for i in range(0, len(y_train)):  # make label hop size 100ms --> 10ms
        if i % 10 == 0 and i != 0:  # to match feature hop size
            j += 1
        y_train[i, :] = y_orig_train[j, :]

    y_orig_dev = load_labels(partition='Devel_DE')  # read in training labels
    y_dev = np.empty((len(y_orig_dev) * 10, 3))
    j = 0
    for i in range(0, len(y_dev)):  # make label hop size 100ms --> 10ms
        if i % 10 == 0 and i != 0:  # to match feature hop size
            j += 1
        y_dev[i, :] = y_orig_dev[j, :]

    x_train = load_mfcc(partition='Train_DE')  # read in training labels
    x_dev = load_mfcc(partition='Devel_DE')

    # now the data must be normalized for develop we use training_mean and training_std
    mean_x_train = np.mean(x_train, axis=0, dtype=np.float64)
    mean_y_train = np.mean(y_train, axis=0, dtype=np.float64)
    std_x_train = np.std(x_train, axis=0, dtype=np.float64)
    std_y_train = np.std(y_train, axis=0, dtype=np.float64)

    for i in range(0, len(x_train[0])):
        x_train[:, i] = (x_train[:, i] - mean_x_train[i])/std_x_train[i]
        x_dev[:, i] = (x_dev[:, i] - mean_x_train[i]) / std_x_train[i]

    for i in range(0, len(y_train[0])):
        y_train[:, i] = (y_train[:, i] - mean_y_train[i]) / std_y_train[i]
        y_dev[:, i] = (y_dev[:, i] - mean_y_train[i]) / std_y_train[i]
    return x_train, x_dev, y_train, y_dev


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
    model.add(Dropout(drops, input_shape=(39,)))
    model.add(Dense(layer_size, activation=activ_fctn, kernel_regularizer=regularizers.l2(regul_kernel)))  # first hidden layer
    for idx in range(0, nr_hidden_layers-1):
        model.add(Dense(layer_size, activation=activ_fctn, kernel_regularizer=regularizers.l2(regul_kernel)))            # hidden layers

    model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(regul_kernel)))   # output layer
    model.summary()

    rmsprop = RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss=ccc_loss)

    performances = np.empty((nr_epochs, 6))
    history = []                   # Creating a empty list for holding the loss later

    ccc_max_raw = 0.0000
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
    target = 0  # 0 = arousal, 1 = valence, 2 = liking

    x_train_rec, x_dev_rec, y_train_rec, y_dev_rec = load_data()   # read in the data

    # print out data dimensions and some elements, for debugging purposes
    print_data_dimensions(x_train_rec=x_train_rec, x_dev_rec=x_dev_rec, y_train_rec=y_train_rec[:, 0], y_dev_rec=y_dev_rec[:, 0])

    # variables to track performance
    best_of_best_raw = [0.0, 0, 0, 0, 0, 0]           # list to hold information about best prediction
    best_of_best_scaled = [0.0, 0, 0, 0, 0, 0]        # format = [CCC, i, j, k, l, #epoch]
    best_of_best_smoothed = [0.0, 0, 0, 0, 0, 0]      #      1    [0    1  2  3  4    5   ]

    yh_best_raw = np.empty(len(y_dev_rec[:, 0]))
    yh_best_scaled = np.empty(len(y_dev_rec[:, 0]))
    yh_best_smoothed = np.empty(len(y_dev_rec[:, 0]) - 100)

    # path to save all plots (6 plots per neural net configuration) + hyperparameters + csv files with performance
    save_path = '/home/joris/Pictures/Runs/SEWA/AROUSAL/mfcc4/'

    # defaults                   #######################################################################################################
    nr_epochs = 20               # model = Sequential()  , 1 input+hidden layer Dense() with input dimension 88 and <layer_size> nodes #
    learning_rate = 0.001        # + <nr_layers_list-1> EXTRA hidden layers Dense() with activ_function = <activ_function>             #
    activ_fctn = 'relu'          # + 1 output layer Dense() with ativ_function = 'linear'                                              #
                                 #######################################################################################################
    # best CCC = 0.557 @ 0.0.1.4 = 2 , 8, 5000, 0.003


    # grid search hyperparamters
    nr_layers_list = 3
    layer_size_list = [8, 16, 32, 64]
    batch_list = [2500, 5000, 10000]
    drops = [0.05, 0.1, 0.5]
    regul_list_kernel = [0.00001, 0.0001, 0.001]

    for i in range(0, len(layer_size_list)):
        for j in range(0, len(batch_list)):
            for k in range(0, len(drops)):
                for l in range(0, len(regul_list_kernel)):
                    ccc_max_raw, ccc_max_scaled, ccc_max_smoothed, best_raw, best_scaled, best_smoothed, yh_raw, yh_scaled, yh_smoothed = \
                    train_model(nr_epochs=nr_epochs, nr_hidden_layers=nr_layers_list, layer_size=layer_size_list[i],  btch_size=batch_list[j],
                        learning_rate=learning_rate, activ_fctn=activ_fctn, x_train=x_train_rec, x_dev=x_dev_rec,
                        y_train=y_train_rec[:, target], y_dev=y_dev_rec[:, target], drops=drops[k], regul_kernel=regul_list_kernel[l],
                        i=i, j=j, k=k, l=l, save_path=save_path, plot=True)

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