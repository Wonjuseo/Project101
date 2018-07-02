import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, Activation, LSTM
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn import datasets
def data_read():
    return datasets.fetch_mldata('MNIST original',data_home='.')
def data_preprocessing(n):
    return np.random.permutation(range(n))[:n]

class BiRNN():
    def __init__(self,n_hidden,n_time,n_in,n_out,weight_initializer):
        self.n_hidden = n_hidden
        self.n_time = n_time
        self.n_in = n_in
        self.n_out = n_out
        self.weight = weight_initializer

    def Make_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(self.n_hidden),input_shape=(n_time,n_in)))
        model.add(Dense(n_out,kernel_initializer=self.weight))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999),metrics=['accuracy'])
        return model

def plotting(epochs,val_acc):
    plt.rc('font',family='serif')
    fig = plt.figure()
    plt.plot(range(epochs),val_acc,label='acc',color='black')
    plt.show()
    plt.savefig('mnist_BiRNN.png')


if __name__ == "__main__":

    mnist = data_read()
    indicies = data_preprocessing(len(mnist.data))

    epochs = 20

    X = mnist.data[indicies]
    Y = mnist.target[indicies]
    Y = np.eye(10)[Y.astype(int)]
    X = X/255.0
    X = X-X.mean(axis=1).reshape(len(X),1)
    X = X.reshape(len(X),28,28)

    X, Y = shuffle(X,Y)
    split_size = int(len(X)/10*3)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=split_size)

    n_in = 28
    n_time = 28
    n_hidden = 128
    n_out = 10

    model = BiRNN(n_hidden,n_time,n_in,n_out,'glorot_uniform')
    model = model.Make_model()
    his = model.fit(X_train,Y_train,epochs= epochs,batch_size=128,validation_split=0.2)
    plotting(epochs,his.history['val_acc'])
    print("good")




