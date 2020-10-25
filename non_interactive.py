
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence


embedding_size = 128
vocab_size = 8254
max_len = 40


image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))


conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('model_weights.h5')

print("="*50)
print("MODEL LOADED")




###########################################################

vocab = np.load('vocab.npy', allow_pickle=True)

from keras.applications import ResNet50
resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')


resnet.save('resnet.h5')


import cv2
from keras.preprocessing.sequence import pad_sequences

img = cv2.imread('download.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224,224))
img = img.reshape(1,224,224,3)

incept = resnet.predict(img).reshape(1,2048)

print("++"*100)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


text_in = ['<start>']

count = 0
while count < 20:

	encoded = []
	for i in text_in:
		encoded.append(vocab[i])

	padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

	sampled_index = np.argmax(model.predict([incept, padded]))

	sampled_word = inv_vocab[sampled_index]

	if sampled_word != '<end>':
		print(sampled_word)

	text_in.append(sampled_word)

