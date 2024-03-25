import os
import json
import pickle
import numpy as np

from PIL import Image

from network.network import Network
from layers.fc_layer import FCLayer
from layers.flatten_layer import FlattenLayer
from layers.activation_layer import ActivationLayer
from activations.activations import relu, relu_prime
from losses.losses import mse, mse_prime
from utils.utils import to_categorical


def generate_x_y_train():
	with open('dataset/train/hashtable_train.json') as fp:
		data = json.load(fp)
	hashtable_train = json.loads(data)


	x_train = []
	y_train = []
	for key, item in hashtable_train.items():
		img = Image.open(f'dataset/train/crops/{key}').convert('L')

		x_train.append(np.array(img, dtype=np.float32)/255.0)
		y_train.append(ord(item))

	return np.array(x_train), np.array(y_train)


def train(epochs=400, learning_rate=0.1):
	x_train, y_train = generate_x_y_train()
	
	x_train = x_train.reshape(x_train.shape[0], 1, 90*50)
	print('x shape:', x_train.shape)
	y_train = to_categorical(y_train, dtype=np.float32)
	print('y shape:', y_train.shape)

	net = Network()
	net.add(FlattenLayer(input_shape=(90, 50)))
	net.add(FCLayer(90*50, 512))
	net.add(ActivationLayer(relu, relu_prime))
	net.add(FCLayer(512, 256))
	net.add(ActivationLayer(relu, relu_prime))
	net.add(FCLayer(256, 128))
	net.add(ActivationLayer(relu, relu_prime))
	net.add(FCLayer(128, 122))

	net.use(mse, mse_prime)
	net.fit(x_train, y_train, epochs=epochs, learning_rate=learning_rate)

	return net


def save_model(network=None, model_name=None):
	if network is None or model_name is None:
		return

	with open(f'models/{model_name}', 'wb') as handle:
		pickle.dump(network, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	network = train(400, 0.1)
	save_model(network, 'model_400_epochs.pickle')
