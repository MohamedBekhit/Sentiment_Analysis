from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import plot_model
from keras.models import model_from_json
import matplotlib.pyplot as plt

'''
Sentiment Analysis Model using RNN (LSTM) implementing with Keras.
Tainig and Test sets used are downloaded ratings from imdb.
'''


vocabulazry_size = 5000

# Load Dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulazry_size)
print("Dataset loaded\nNumber of training samples = ",
      len(X_train), ", Number of test samples = ", len(X_test))

word_to_id = imdb.get_word_index()
id_to_word = {i:word for word, i in word_to_id.items()}

# max review length = 2697, min review length = 14

# Padding
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


# RNN model design
embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulazry_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# Model training and evaluation
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

batch_size = 64
num_epochs = 3

X_val, y_val = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

history = model.fit(X_train2, y_train2, validation_data=(X_val, y_val),
          batch_size=batch_size, epochs=num_epochs)

model_json = model.to_json()
print("Saving model in JSON format")
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Model saved")

scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy: ', scores[1])

# Visualizations:
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_loss.png')
plt.show()





## LOAD AND CREATE MODEL FROM JSON FILE :
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # LOAD WEIGHTS :
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
