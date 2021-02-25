# library imports

"""
This script reads a csv file from previously extracted data and build a deap learning model
to generate songs.

Learning Outcomes:
1. tokenizing text
2. create sequences and pad sentences to be of one length
3. split the padded word sequences so that the last word of the sequence is the target
4. Use Embedding
5. Use LSTMs
6. generate text based on a sentence from user.
7. save output text and model

"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers


# styling plot
plt.style.use('fivethirtyeight')


lyrics = []  # to contain all songs
with open('data/songs_csv.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:  # reading the songs one by one
        lyrics.append(row['lyrics'])

print(f'There are {len(lyrics)} songs')

sentences = []  # contains individual sentences
total_words = 0  # to find the total number of words
word_count = {}
for song in lyrics:
    for sentence in song.split('\n'):
        sentences.append(sentence)
        total_words += len(sentence.split(' '))

for sentence in sentences:
    words = sentence.split(' ')
    for word in words:
        if word not in word_count:
            word_count[word] = 0
        else:
            word_count[word] += 1

print(f'Total words {total_words}')
print(f'There are: {len(sentences)} sentences')

#  sentence lengths
sentence_lengths = [len(sentence) for sentence in sentences]
max_sentence_length = max(sentence_lengths)
average_sentence_length = int(np.mean(sentence_lengths))
print(f'Maximum Sentence Length: {max_sentence_length}')
print(f'Average Sentence Length: {average_sentence_length}')

# Tokenization params
VOCAB_SIZE = 10000 # third of the total wards

# padding params
PAD_TYPE = 'pre'
TRUNC_TYPE = 'post'
MAX_LEN = average_sentence_length * 4  # to make the maximum length to be: 39 *4

# embedding param
EMBED_DIM = 120
# epochs
NUM_EPOCH = 20

# fitting tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
reverse_word_index = dict([(value, key) for key, value in word_index.items()])

input_sequences = []

for sentence in sentences:  # generate sequences for each sentence
    token_list = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(token_list)):  # generate sequences of len 2, 3, ...n
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
print(len(input_sequences))
# training data and labels split
padded = np.array(pad_sequences(input_sequences, maxlen=MAX_LEN, padding=PAD_TYPE, truncating=TRUNC_TYPE))
training_set = padded[:, :-1]  # all but the last word of a sequence
labels = padded[:, -1]  # the last word of each sequence

# using one hot encoder for labels
target_y = tf.keras.utils.to_categorical(labels, num_classes=VOCAB_SIZE)

# a callback class to stop upon reaching 98% accuracy
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy')) > 0.98:
            self.model.stop_training = True


callbacks = MyCallback()

# building the model
model = tf.keras.Sequential([
    # embedding layer with -1 for input_length as the last word is moved to labels
    layers.Embedding(VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN - 1),
    # a bidirectional lstm. return_sequences set to True as there will be another lstm layer
    layers.Bidirectional(layers.LSTM(150, return_sequences=True)),
    # 20% dropout layer
    layers.Dropout(0.2),
    # second lstm layer
    layers.LSTM(120),
    # DNN with regularizer
    layers.Dense(VOCAB_SIZE/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    # output layer
    layers.Dense(VOCAB_SIZE, activation='softmax')

])
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fitting the model
history = model.fit(training_set, target_y, steps_per_epoch=300, epochs=NUM_EPOCH, callbacks=[callbacks])

# user input for the first sentence of the song as well as the number of words to generate
seed_text = input('Enter the first song words: \n')
next_words = int(input('Enter the Number of words to be generated: \n'))

for _ in range(next_words):
    # using the same transformation as for preprocessing
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=MAX_LEN - 1, padding='pre')
    # get prediction class (words) in this case
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        # matching indices with words
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)

model.save('output/model')

with open('output/generated_song.txt', 'w') as file:
    file.write(seed_text)
# plotting
plt.plot(history.history['accuracy'])
plt.xlabel('Epochs')
plt.title('Accuracy Performance')
plt.ylabel('Accuracy')
plt.savefig('output/model_performance')
