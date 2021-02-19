import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

FILE_PATH = 'data/sarcasm.json'

# initiate lists that would contain labels, and headlines
labels = []
headlines = []
with open(FILE_PATH,'r') as f:
    data = json.loads(f.read())
    for item in data:
        labels.append(item['is_sarcastic'])
        headlines.append(item['headline'])

labels = np.array(labels) # convert labels into numpy array to be accepted into tensorflow
# show the length of data
print(f'There are {len(labels)} Labels')
print(f'There are {len(headlines)} Headlines')

# Output:
    # There are 26709 Labels
    # There are 26709 Headlines

# splitting the data into training and validation sets
TRAIN_SIZE = int(0.8 * len(headlines)) # 80% of data for training

# training data
train_sentences = headlines[:TRAIN_SIZE]
train_labels = labels[:TRAIN_SIZE]

# validation data
valid_sentences = headlines[TRAIN_SIZE:]
valid_labels = labels[TRAIN_SIZE:]


# initiate hyper params for Tokenizer and models
VOCAB_SIZE = 10000 # 10K words in the corpus
OOV_TOK = '<OOV>' # for words out of corpus

PAD_TYPE = 'post' # add padding at the end of sentence
TRUNC_TYPE = 'post' # truncate sentences longer than max_len from the end
MAX_LEN = 120 # all sentences will be of 120 length

DIM_EMBEDDING = 16 #

# Tokenizing
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
reverse_word_index = dict([(value,key) for key, value in word_index.items()])
# sequences
sequences = tokenizer.texts_to_sequences(train_sentences)
padded_training = pad_sequences(sequences,maxlen=MAX_LEN, padding=PAD_TYPE, truncating=TRUNC_TYPE)
print(f'Showing Sequences which have {len(sequences)}. The first sequence:')
print(sequences[0])
print(f'Padded with length of {len(padded_training[0])}\n{padded_training[0]}')
print(f'Decoded {[reverse_word_index[code] for code in sequences[0]]}')
print(f'The actual sentence is: \n\t{train_sentences[0]}')

# for tokenizing test sentences, we only need to create sequences and padding
# tokenizer will use the same tokens for testing as initiated for training

test_sequences = tokenizer.texts_to_sequences(valid_sentences)
padded_testing = pad_sequences(test_sequences,maxlen=MAX_LEN, padding=PAD_TYPE, truncating=TRUNC_TYPE)

# checking consistency of padded shape
print(f'Padded training: {padded_training.shape}')
print(f'Padded Testing: {padded_testing.shape}')


# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, DIM_EMBEDDING, input_length=MAX_LEN),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# show model layers and summary
print(model.summary())

# training the model
NUM_EPOCHS = 10
history = model.fit(x=padded_training,
                    y=train_labels,
                    validation_data=(padded_testing, valid_labels),
                    epochs = NUM_EPOCHS)

def plot_metrics(history,desc):
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8,15))
    # accuracy plots
    ax[0].plot(history.history['accuracy'], label='accuracy')
    ax[0].plot(history.history['val_accuracy'], label='val_accuracy')
    ax[0].legend(loc=5)
    ax[0].set_xlabel('Epochs')

    # loss plots
    ax[1].plot(history.history['loss'], label='loss')
    ax[1].plot(history.history['val_loss'], label='val_loss')
    ax[1].legend(loc=5)
    ax[1].set_xlabel('Epochs')

    fig.suptitle(f'Model Performance-{desc.title()}')
    fig.savefig(f'figures/{desc}')
    plt.show()


plot_metrics(history,'embedding_1l-Convnet_1l')

model.save('output/model2.h5')

