import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


# Load data
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
X_train, y_train = train
X_test, y_test = test

# Data Preprocessing
# Padding
X_train = pad_sequences(X_train, maxlen=100, value=0.)
X_test = pad_sequences(X_test, maxlen=100, value=0.)

# Labels to binary vectors
y_train = to_categorical(y_train, nb_classes=2)
y_test = to_categorical(y_test, nb_classes=2)

# Network architecture
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(X_train, y_train, validation_set=(X_test, y_test), show_metric=True,
            batch_size=32)

model.save("model.tflearn")

score = model.evaluate(X_test, y_test)
print("Accuracy = ", score[0]*100, "%")

# To loaded model:
# model.load("model.tflearn")

# To ccess Graph: Run the following command in terminal:
# $ tensorboard --logdir='/tmp/tflearn_logs'
