import sys
import numpy as np
import pandas as pd
from scipy import signal

###################
## Solver for single layer network of three hidden nodes.
## Output node has no bias but hidden nodes have bias.
## We automatically accounted for this by adding an extra
## column of 1's to the input data.
## return [W, w] where the rows of W are the three hidden
## node weights and w are the output node weights
###################
def back_prop(traindata, trainlabels):
    ##############################
    ### Initialize all weights ###
    W = np.random.rand(3, cols)
    w = np.random.rand(3)
    ###################################################
    ### Calculate initial objective in variable obj ###
    obj = 0
    hidd_layers = np.matmul(traindata, np.transpose(W))
    sigmoid = lambda x: 1/(1+np.exp(-x))
    hidd_layers = np.array([sigmoid(xi) for xi in hidd_layers])
    out_layer =  np.matmul(hidd_layers,np.transpose(w))
    obj = np.sum(np.square(out_layer - trainlabels))
    ###############################
    ### Begin gradient descent ####

    stop = 0.001
    epochs = 1000
    eta = .001
    prevobj = np.inf
    i = 0
    while (prevobj - obj > stop and i < epochs):
        # Update previous objective
        prevobj= obj
        # Calculate gradient update dellw for output node w
        # dellw is the same dimension as w
        delw= (np.dot(hidd_layers[0,:],w)-trainlabels[0])*hidd_layers[0,:]
        for j in range(1,rows):
            delw += (np.dot(hidd_layers[j,:],np.transpose(w))-trainlabels[j])*hidd_layers[j,:]
        # Update w
        w= w - eta*delw
        # Calculate gradient update dellW for hidden layer weights W.
        # dellW has to be of same dimension as W.
        # Follow the steps below.

        # Our three hidden nodes are s, u, and v.
        # Let us first calculate dells. After that we do dellu and dellv.
        # Recall that dells = df/dz1 * (dz1/ds1, dz1,ds2)
        dels= np.sum(np.dot(hidd_layers[0,:],w)-trainlabels[0])*w[0]*hidd_layers[0,0]*(1-hidd_layers[0,0])*traindata[0,:]
        for j in range(1, rows):
            dels = dels+ np.sum(np.dot(hidd_layers[j, :], w) - trainlabels[j]) * w[0] * hidd_layers[j, 0] * (
                        1 - hidd_layers[j, 0]) * traindata[j, :]
        # Calculate dellu
        delu = np.sum(np.dot(hidd_layers[0, :], w) - trainlabels[0]) * w[1] * hidd_layers[0, 1] * (
                    1 - hidd_layers[0, 1]) * traindata[0, :]
        for j in range(1, rows):
            delu = delu + np.sum(np.dot(hidd_layers[j, :], w) - trainlabels[j]) * w[1] * hidd_layers[j, 1] * (
                    1 - hidd_layers[j, 1]) * traindata[j, :]
        # Calculate dellv
        delv = np.sum(np.dot(hidd_layers[0, :], w) - trainlabels[0]) * w[2] * hidd_layers[0, 2] * (
                    1 - hidd_layers[0, 2]) * traindata[0, :]
        for j in range(1, rows):
            delv = delv + np.sum(np.dot(hidd_layers[j, :], w) - trainlabels[j]) * w[2] * hidd_layers[j, 2] * (
                    1 - hidd_layers[j, 2]) * traindata[j, :]
        # Put dells, dellu, and dellv as rows of dellW
        delW= np.array([dels,delu,delv])
        # Update W
        W= W - eta*delW
        # Recalculate objective
        hidd_layers=np.matmul(traindata ,np.transpose(W))
        hidd_layers= np.array([sigmoid(xi) for xi in hidd_layers])
        out_layer = np.matmul(hidd_layers, np.transpose(w))
        obj = np.sum(np.square(out_layer - trainlabels))
        # Update i and print objective (comment print before final submission)
        i = i + 1
        print("i=", i, " Objective=", obj)

    return [W, w]


###################
## Stochastic gradient descent for same network as above
## return [W, w] where the rows of W are the three hidden
## node weights and w are the output node weights
###################
def sgd(traindata, trainlabels, batch_size):
    ##############################
    ### Initialize all weights ###
    pW = np.random.rand(3, cols)
    pw = np.random.rand(3)
    ###################################################
    ### Calculate initial objective in variable obj ###
    obj = 0
    mini_train_data=traindata[:batch_size,:]
    mini_trainlabels = trainlabels[:batch_size]
    mini_hidd_layers = np.matmul(mini_train_data, np.transpose(pW))
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    mini_hidd_layers = np.array([sigmoid(xi) for xi in mini_hidd_layers])
    mini_out_layer = np.matmul(mini_hidd_layers, np.transpose(pw))
    obj = np.sum(np.square(mini_out_layer - mini_trainlabels))
    ###############################
    ### Begin gradient descent ####

    stop = 0.001
    epochs = 1000
    eta = .1
    prevobj = np.inf
    i = 0
    while (i < epochs):
        # Update previous objective
        w= pw
        W= pW
        # Calculate gradient update dellw for output node w
        # dellw is the same dimension as w
        delw = (np.dot(mini_hidd_layers[0, :], w) - mini_trainlabels[0]) * mini_hidd_layers[0, :]
        for j in range(1, batch_size):
            delw += (np.dot(mini_hidd_layers[j, :], np.transpose(w)) - mini_trainlabels[j]) * mini_hidd_layers[j, :]
        # Update w
        w = w - eta * delw
        # Calculate gradient update dellW for hidden layer weights W.
        # dellW has to be of same dimension as W.
        # Follow the steps below.

        # Our three hidden nodes are s, u, and v.
        # Let us first calculate dells. After that we do dellu and dellv.
        # Recall that dells = df/dz1 * (dz1/ds1, dz1,ds2)
        dels = np.sum(np.dot(mini_hidd_layers[0, :], w) - mini_trainlabels[0]) * w[0] * mini_hidd_layers[0, 0] * (
                    1 - mini_hidd_layers[0, 0]) * mini_train_data[0, :]
        for j in range(1, batch_size):
            dels = dels + np.sum(np.dot(mini_hidd_layers[j, :], w) - mini_trainlabels[j]) * w[0] * mini_hidd_layers[j, 0] * (
                    1 - mini_hidd_layers[j, 0]) * mini_train_data[j, :]
        # Calculate dellu
        delu = np.sum(np.dot(mini_hidd_layers[0, :], w) - mini_trainlabels[0]) * w[1] * mini_hidd_layers[0, 1] * (
                1 - mini_hidd_layers[0, 1]) * mini_train_data[0, :]
        for j in range(1, batch_size):
            delu = delu + np.sum(np.dot(mini_hidd_layers[j, :], w) - mini_trainlabels[j]) * w[1] * mini_hidd_layers[j, 1] * (
                    1 - mini_hidd_layers[j, 1]) * mini_train_data[j, :]
        # Calculate dellv
        delv = np.sum(np.dot(mini_hidd_layers[0, :], w) - mini_trainlabels[0]) * w[2] * mini_hidd_layers[0, 2] * (
                1 - mini_hidd_layers[0, 2]) * mini_train_data[0, :]
        for j in range(1, batch_size):
            delv = delv + np.sum(np.dot(mini_hidd_layers[j, :], w) - mini_trainlabels[j]) * w[2] * mini_hidd_layers[j, 2] * (
                    1 - mini_hidd_layers[j, 2]) * mini_train_data[j, :]
        # Put dells, dellu, and dellv as rows of dellW
        delW = np.array([dels, delu, delv])
        # Update W
        W = W - eta * delW
        # Recalculate objective
        hidd_layers = np.matmul(traindata, np.transpose(W))
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        hidd_layers = np.array([sigmoid(xi) for xi in hidd_layers])
        mini_out_layer = np.matmul(hidd_layers, np.transpose(w))
        obj = np.sum(np.square(mini_out_layer - trainlabels))
        np.random.shuffle(traindata)
        mini_train_data = traindata[:batch_size, :]
        mini_trainlabels = trainlabels[:batch_size]
        if (obj < prevobj):
            prevobj = obj
            pw = w
            pW = W
        # Update i and print objective (comment print before final submission)
        i = i + 1
        print("i=", i, " Objective=", obj)

    return [pW, pw]


###################
## Back propagation gradient descent for a simple convolutional
## neural network containing one 2x2 convolution with global
## average pooling.
## Input to the network are 3x3 matrices.
## return the 2x2 convolutional kernel
###################
def convnet(traindata, trainlabels):
    stop = 0.001
    epochs = 1000
    eta = .1
    prevobj = np.inf
    i = 0
    train_len=len(trainlabels)

    c = np.ones((2, 2))

    obj = 0
    for i in range(0, train_len):
        hidd_layers = signal.convolve2d(traindata[i], c, mode='valid')
        for j in range(0, 2, 1):
            for k in range(0, 2, 1):
                hidd_layers[j][k] = sigmoid(hidd_layers[j][k])
        output_layer = (hidd_layers[0][0] + hidd_layers[0][1] + hidd_layers[1][0] + hidd_layers[1][1]) / 4
        obj += (output_layer - labels[i]) ** 2

    while (prevobj - obj > stop and i < epochs):

        prevobj = obj

        delc1 = 0
        delc2 = 0
        delc3 = 0
        delc4 = 0

        for i in range(0, train_len):
            hidd_layers = signal.convolve2d(traindata[i], c, mode="valid")
            for j in range(0, 2, 1):
                for k in range(0, 2, 1):
                    hidd_layers[j][k] = sigmoid(hidd_layers[j][k])


            dz1dc1 = hidd_layers[0][0]* (1- hidd_layers[0][0])* traindata[i][0][0]
            dz2dc1 = hidd_layers[0][1]* (1- hidd_layers[0][1])* traindata[i][0][1]
            dz3dc1 = hidd_layers[1][0]* (1- hidd_layers[1][0])* traindata[i][1][0]
            dz4dc1 = hidd_layers[1][1]* (1- hidd_layers[1][1])* traindata[i][1][1]

            dz1dc2 = hidd_layers[0][0]* (1- hidd_layers[0][0])* traindata[i][0][1]
            dz2dc2 = hidd_layers[0][1]* (1- hidd_layers[0][1])* traindata[i][0][2]
            dz3dc2 = hidd_layers[1][0]* (1- hidd_layers[1][0])* traindata[i][1][1]
            dz4dc2 = hidd_layers[1][1]* (1- hidd_layers[1][1])* traindata[i][1][2]

            dz1dc3 = hidd_layers[0][0]* (1- hidd_layers[0][0])* traindata[i][1][0]
            dz2dc3 = hidd_layers[0][1]* (1- hidd_layers[0][1])* traindata[i][1][1]
            dz3dc3 = hidd_layers[1][0]* (1- hidd_layers[1][0])* traindata[i][2][0]
            dz4dc3 = hidd_layers[1][1]* (1- hidd_layers[1][1])* traindata[i][2][1]

            dz1dc4 = hidd_layers[0][0]* (1- hidd_layers[0][0])* traindata[i][1][1]
            dz2dc4 = hidd_layers[0][1]* (1- hidd_layers[0][1])* traindata[i][1][2]
            dz3dc4 = hidd_layers[1][0]* (1- hidd_layers[1][0])* traindata[i][2][1]
            dz4dc4 = hidd_layers[1][1]* (1- hidd_layers[1][1])* traindata[i][2][2]

            f_rt = (hidd_layers[0][0]+ hidd_layers[0][1]+ hidd_layers[1][0]+ hidd_layers[1][1])/4 - labels[i]

            delc3 += (f_rt * (dz1dc3 + dz2dc3 + dz3dc3 + dz4dc3)) / 2
            delc2 += (f_rt * (dz1dc2 + dz2dc2 + dz3dc2 + dz4dc2)) / 2
            delc1 += (f_rt * (dz1dc1 + dz2dc1 + dz3dc1 + dz4dc1)) / 2
            delc4 += (f_rt * (dz1dc4 + dz2dc4 + dz3dc4 + dz4dc4)) / 2

        c[0][0] -= eta * delc1
        c[0][1] -= eta * delc2
        c[1][0] -= eta * delc3
        c[1][1] -= eta * delc4

        obj = 0
        for i in range(0, train_len):
            hidd_layers = signal.convolve2d(traindata[i], c, mode='valid')
            for j in range(0, 2, 1):
                for k in range(0, 2, 1):
                    hidd_layers[j][k] = sigmoid(hidd_layers[j][k])
            output_layer = (hidd_layers[0][0] + hidd_layers[0][1] + hidd_layers[1][0] + hidd_layers[1][1]) / 4
            obj += (output_layer - labels[i]) ** 2

    return c


###############
#### MAIN #####
###############

#######################################
### Read data for back_prop and sgd ###
#######################################

f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:, 1:]
trainlabels = data[:, 0]

onearray = np.ones((train.shape[0], 1))
train = np.append(train, onearray, axis=1)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:, 1:]
testlabels = data[:, 0]

onearray = np.ones((test.shape[0], 1))
test = np.append(test, onearray, axis=1)

rows = train.shape[0]
cols = train.shape[1]

hidden_nodes = 3

#########################
### Read data for convnet
#########################
traindir = sys.argv[3]
df = pd.read_csv(traindir + '/data.csv')  # load images' names and labels
names = df['Name'].values
labels = df['Label'].values

traindata = np.empty((len(labels), 3, 3), dtype=np.float32)
for i in range(0, len(labels)):
    image_matrix = np.loadtxt(traindir + '/' + names[i])
    traindata[i] = image_matrix

testdir = sys.argv[4]
df = pd.read_csv(testdir + '/data.csv')  # load images' names and labels
names2 = df['Name'].values
labels2 = df['Label'].values

testdata = np.empty((len(labels2), 3, 3), dtype=np.float32)
for i in range(0, len(labels2)):
    image_matrix = np.loadtxt(testdir + '/' + names2[i])
    testdata[i] = image_matrix

#######################
#### Train the networks
#######################

## Helper function
sigmoid = lambda x: 1 / (1 + np.exp(-x))

[W, w] = back_prop(train, trainlabels)

# A small batch size such as even 2 or 5 will also work
batch_size = 5
[W2, w2] = sgd(train, trainlabels, batch_size)

c = convnet(traindata, labels)

##################################
#### Classify test data as 1 or -1
##################################

OUT = open("back_prop_predictions", 'w')
bp_hidd_layers = np.matmul(test, np.transpose(W))
sigmoid = lambda x: 1 / (1 + np.exp(-x))
bp_hidd_layers = np.array([sigmoid(xi) for xi in bp_hidd_layers])
bp_out_layer=  np.matmul(bp_hidd_layers,np.transpose(w))
for i in range(bp_out_layer.shape[0]):
    if bp_out_layer[i]>0:
        bp_out_layer[i] = 1
    else:
        bp_out_layer[i] = -1
OUT.write(str(bp_out_layer))

OUT = open("sgd_predictions", 'w')
sgd_hidd_layers = np.matmul(test, np.transpose(W2))
sgd_hidd_layers = np.array([sigmoid(xi) for xi in sgd_hidd_layers])
sgd_out_layer=  np.matmul(sgd_hidd_layers,np.transpose(w2))
for i in range(sgd_out_layer.shape[0]):
    if sgd_out_layer[i]>0:
        sgd_out_layer[i] = 1
    else:
        sgd_out_layer[i] = -1
OUT.write(str(sgd_out_layer))
## For the convnet print the 2x2 kernel c in the
## first two lines. In the following lines print
## the test predictions.

OUT = open("convnet_output", 'w')
OUT.write(str(c)+'\n')
for i in range(0,len(labels2)):
    cnn_hid_layers = signal.convolve2d(testdata[i],c, mode='valid')
    for j in range(0,2,1):
        for k in range(0,2,1):
            cnn_hid_layers[j][k] = sigmoid(cnn_hid_layers[j][k])
    cnn_output_layer = (cnn_hid_layers[0][0] + cnn_hid_layers[0][1]+cnn_hid_layers[1][0]+cnn_hid_layers[1][1])/4
    if (cnn_output_layer < 0.5):
        OUT.write(str(-1)+'\n')
    else:
        OUT.write(str(1)+'\n')
