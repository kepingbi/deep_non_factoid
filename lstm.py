import cPickle as pkl
import sys
import time
import yahoo
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
import recurrent as rnn
from seq import pad_sequences

max_features = 20000
maxlen = 100
# how to choose batch size?
batch_size = 32

#using word2vec


print("Load data")
(X_train, y_train), (X_test, y_test) = yahoo.load_data(test_split = 0.2)
maxlen = max( max([len(x) for x in X_train]), max([len(x) for x in X_test]))
print maxlen
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

#compare gated recu unit?
lstmf1 = LSTM(output_dim=64)
gru1 = GRU(output_dim=64)
gru2 = GRU(output_dim=64)
lstmb2 = LSTM(output_dim=64)
lstmf2 = LSTM(output_dim=64)
brnn1 = rnn.Bidirectional(forward=lstmf1,backward=gru1)
brnn2 = rnn.Bidirectional(forward=lstmf2,backward=gru2)
emb = Embedding(max_features,128,input_length=maxlen)

model = Sequential()
model.add(emb)
model.add(brnn1)
model.add(Dense(1))
model.add(Activation('sigmoid'))
# model.add(brnn2)
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# model.add(Activation('sigmoid'))
# model.add(Dropout(0.5))
# model.add(brnn2)
# model.add(rnn.GlobalPooling())
# model.add(Activation('sigmoid'))

print ("Build Model")
model.compile(loss='binary_crossentropy' , optimizer='adam' ,class_mode="binary")


print("Train...")
model.fit(X_train, y_train, batch_size = batch_size, nb_epoch=4,
          validation_data=(X_test,y_test), show_accuracy=True)

classes1 = model.predict(X_test)
classes2 = model.predict_classes(X_test)
print classes1
print classes2
