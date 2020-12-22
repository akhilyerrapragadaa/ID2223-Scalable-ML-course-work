from os import listdir

import tensorflow as tf
from numpy import array, argmax
import re
import string

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from pickle import dump,load
from nltk.translate.bleu_score import corpus_bleu

imagesSet = "/home/nanda/Scalable ML/lab2_exercise/dataset/Flicker8k_Dataset"
allDescriptions = "/home/nanda/Scalable ML/lab2_exercise/dataset/Flickr8k.token.txt"
trainImages = "/home/nanda/Scalable ML/lab2_exercise/dataset/Flickr_8k.trainImages.txt"
allDesctxt = "/home/nanda/Scalable ML/lab2_exercise/dataset/descriptions.txt"
featurespkl = "/home/nanda/Scalable ML/lab2_exercise/dataset/features.pkl"
testImages = "/home/nanda/Scalable ML/lab2_exercise/dataset/Flickr_8k.testImages.txt"
savedModels = "/home/nanda/Scalable ML/lab2_exercise/models/model_9.h5"

print(tf.__version__)

PreTrainedCNNModel = InceptionV3(weights = 'imagenet') 

allFeat = dict()
imageDescriptions = dict()
imgDescPair = list()
allTrainDescriptions = list()
trainingImageDescriptions = dict()
Images = list()

# Pre trained CNN

def CNN():
	
# CNN for feature extraction

	for name in listdir(imagesSet):
		filename = imagesSet + '/' + name
		image = load_img(filename, target_size=(299, 299))
		image = img_to_array(image)
		print(name)
		image = image.reshape(
			(1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		feature = PreTrainedCNNModel.predict(image, verbose=0)
		imgId = name.split('.')[0]
		allFeat[imgId] = feature
	saveDesc(allFeat)
		
def saveDesc(allFeatures):  
	dump(allFeatures, open(featurespkl , 'wb'))
 


#Get Descriptions From Token Dataset

def loadallDesc():
	for line in open(allDescriptions, 'r').read().split('\n'):
		tokens = line.split("#")
		if len(line) < 2:
			continue
		imgId, imgDesc = tokens[0], tokens[1:]
		imgId = imgId.split('.')[0]
		imgDesc = ' '.join(imgDesc)
		if imgId not in imageDescriptions:
			imageDescriptions[imgId] = list()
		imageDescriptions[imgId].append(imgDesc)
	
	for key, value in imageDescriptions.items():
		for i in range(len(value)):
			eachDescription = value[i]
			eachDescription = eachDescription.split()
			eachDescription = [word.lower() for word in eachDescription]
			eachDescription = [w.translate(str.maketrans('', '', string.punctuation)) for w in eachDescription]
			eachDescription = [word for word in eachDescription if len(word)>1]
			eachDescription = [word for word in eachDescription if word.isalpha()]
			value[i] =  ' '.join(eachDescription)
 
	
	for key, value in imageDescriptions.items():
		for desc in value:
			imgDescPair.append(key + ' ' + desc)
   
	allRel = '\n'.join(imgDescPair)
	file = open(allDesctxt , 'w')
	file.write(allRel)
	file.close()
	

# Preprocessing Training Data

def fromAllDesc():
	for line in open(trainImages, 'r').read().split('\n'):
		if len(line) < 1:
			continue
		imgId = line.split('.')[0]
		Images.append(imgId)
	
	for line in  open(allDesctxt, 'r').read().split('\n'):
		tokens = line.split()
		imgId, imgDesc = tokens[0], tokens[1:]
		if imgId in set(Images):
			if imgId not in trainingImageDescriptions:
				trainingImageDescriptions[imgId] = list()
			finalDescription = 'startseq ' + ' '.join(imgDesc) + ' endseq'
			trainingImageDescriptions[imgId].append(finalDescription)
			allTrainDescriptions.append(finalDescription)
	return trainingImageDescriptions


def getFeatures():
	all_features = load(open(featurespkl, 'rb'))
	features = {k: all_features[k] for k in fromAllDesc()}
	return features


def getVocab():
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(allTrainDescriptions)
	return tokenizer


def getLength():
	return max(len(d.split()) for d in allTrainDescriptions)


def txtToseq(tokenizer, maxLen, value, feat):
	X1, X2, y = list(), list(), list()
	for desc in value:
		seq = tokenizer.texts_to_sequences([desc])[0]
		for i in range(1, len(seq)):
			inSeq, outSeq = seq[:i], seq[i]
			inSeq = pad_sequences([inSeq], maxlen=maxLen)[0]
			outSeq = to_categorical([outSeq], num_classes=vocabSize)[0]

			X1.append(feat)
			X2.append(inSeq)
			y.append(outSeq)
	return array(X1), array(X2), array(y)


def trainingData(descriptions, trainFeat, tokenizer, maxLen):
	while 1:
		for key, value in descriptions.items():
			feat = trainFeat[key][0]
			recvFeatVect, inSeqVect, outSeqVect = txtToseq(tokenizer, maxLen, value, feat)
			yield [recvFeatVect, inSeqVect], outSeqVect



# Step 1: Uncomment below to initiate pre training and extract features from all images
 
"""
CNN()
"""

# Step 2: load all training descriptions, training features and available vocabulary 

loadallDesc()
print('Loaded training descriptions from all desc ', len(fromAllDesc()))
tokenizer = getVocab()
vocabSize = len(tokenizer.word_index) + 1
print('Total Vocabulary from training descriptions ', vocabSize)
print('Longest sentence for RNN input layer ',  getLength())
trainFeat = getFeatures()

# Step 3: Model creation and training with descriptions and features

# Input for Image
inputs1 = layers.Input(shape=(47,1000))
fe1 = layers.Dropout(0.5)(inputs1)
fe2 = layers.Dense(256, activation='relu')(fe1)

# Input for text
inputs2 = layers.Input(shape=(getLength(),))
se1 = layers.Embedding(vocabSize, 256, mask_zero=True)(inputs2)
se2 = layers.Dropout(0.5)(se1)
se3 = layers.LSTM(256)(se2)

# Combine here
decoder1 = layers.add([fe2, se3])
decoder2 = layers.Dense(256, activation='relu')(decoder1)
outputs = layers.Dense(vocabSize, activation='softmax')(decoder2)
model = models.Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# Step 4: Uncomment below to start training

"""
generator = trainingData(fromAllDesc(), getFeatures(), getVocab(), getLength())
model.fit(generator, epochs=18, steps_per_epoch=60, verbose=1)
# Use Below to save trained models for different loss 
 
#model.save('/home/nanda/Scalable ML/lab2_exercise/models/model_' +' '+ '.h5')
"""

# Step 5: Test the trained model 

trainedModel = models.load_model(savedModels)

def getTestDesc(model, tokenizer, feat, maxLen):
	in_text = 'startseq'
	for i in range(maxLen):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=maxLen)
		outWord = model.predict([feat,sequence], verbose=0)
		outWord = argmax(outWord)
		for each, index in tokenizer.word_index.items():
			if index == outWord:
				word = each
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text
"""
for line in open(testImages, 'r').read().split('\n'):
	print(line)
	for name in listdir(imagesSet):
		if re.match(name, line):
			
			PreTrainedCNNModel.layers.pop()
			PreTrainedCNNModel = models.Model(inputs=PreTrainedCNNModel.inputs, outputs=PreTrainedCNNModel.layers[-1].output)
			image = load_img(imagesSet +'/' + line, PreTrainedCNNModel, target_size=(299, 299))
			image = img_to_array(image)
			image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
			image = preprocess_input(image)
			feature = PreTrainedCNNModel.predict(image, verbose=0)
   
			description = getTestDesc(trainedModel, tokenizer, feature, getLength())
			print(description)


"""
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()

	for key, value in descriptions.items():

		feat = getTestDesc(model, tokenizer, photos[key], max_length)

		references = [d.split() for d in value]

		#print("References: ",  references)

		#print("Predicted: ", yhat)

		actual.append(references)
		predicted.append(feat.split())

	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
	


evaluate_model(trainedModel, fromAllDesc(),getFeatures(), getVocab(), getLength())
