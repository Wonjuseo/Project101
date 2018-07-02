import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def n(digits=3):
    number = ''
    for i in range(np.random.randint(1,digits+1)):
        number += np.random.choice(list('0123456789'))
    return int(number)

def padding(chars, maxlen):
    return chars + ' '*(maxlen-len(chars))

N = 20000
N_train = int(N*0.9)
N_test = N-N_train

digits = 3
input_digits = digits*2+1
output_digits = digits+1

added = set()
questions = []
answers = []

while len(questions) < N:
    a, b = n(), n()

    pair = tuple(sorted((a,b)))
    if pair in added:
        continue


    question = '{}+{}'.format(a,b)
    question = padding(question,input_digits)
    answer = str(a+b)
    answer = padding(answer, output_digits)

    added.add(pair)
    questions.append(question)
    answers.append(answer)

chars = '0123456789+ '
char_indices = dict((c,i) for i, c in enumerate(chars))
indices_char = dict((i,c) for i, c in enumerate(chars))

X = np.zeros((len(questions),input_digits,len(chars)),dtype=np.integer)
Y = np.zeros((len(questions),digits+1,len(chars)),dtype=np.integer)

for i in range(N):
    for t, char in enumerate(questions[i]):
        X[i,t,char_indices[char]]=1
    for t, char in enumerate(answers[i]):
        Y[i,t,char_indices[char]]=1

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=N_train)

n_in = len(chars)
n_hidden =128
n_out = len(chars)

model = Sequential()
# Encoder
model.add(LSTM(n_hidden,input_shape=(input_digits,n_in)))
# Decoder
model.add(RepeatVector(output_digits))
model.add(LSTM(n_hidden,return_sequences=True))

model.add(TimeDistributed(Dense(n_out)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999),metrics=['accuracy'])

epochs = 200
batch_size = 128

for epoch in range(epochs):
    model.fit(X_train,Y_train,batch_size=batch_size,epochs=1,validation_data=(X_test,Y_test))

    for i in range(10):
        index = np.random.randint(0,N_test)
        question = X_test[np.array([index])]
        answer = Y_test[np.array([index])]
        prediction = model.predict_classes(question,verbose=0)

        question = question.argmax(axis=-1)
        answer = answer.argmax(axis=-1)

        q = ''.join(indices_char[i] for i in question[0])
        a = ''.join(indices_char[i] for i in answer[0])
        p = ''.join(indices_char[i] for i in prediction[0])

        print('-' * 10)
        print('Q:  ', q)
        print('A:  ', p)
        print('T/F:', end=' ')
        if a == p:
            print('T')
        else:
            print('F')
        print('-' * 10)
