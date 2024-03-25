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


def load_model(model_file):
	with open(f'models/{model_file}', 'rb') as f:
		object_file = pickle.load(f)

	return object_file


def tests(network=None):
	if network is None:
		return

	with open('dataset/tests/hashtable_tests.json') as fp:
		dict_images = json.load(fp)
	hashtable_tests = json.loads(dict_images)


	count_equals = 0
	for key, item in hashtable_tests.items():
		print('image name:', key)
		image = Image.open(f'dataset/tests/crops/{key}').convert('L')
		image_np_array = np.array(image, dtype=np.float32)/255.0
		print('image test shape:', image_np_array.shape)

		pred = network.predict(np.array([image_np_array]))

		idx = np.argmax(pred)
		print('prediction:', idx)
		idx_true = ord(item)
		print('true:', idx_true)

		if idx == idx_true: count_equals += 1

		# image.show()
		# input()
		# image.close()
	print('count equals:', count_equals)
	print('precision percentage:', count_equals/len(hashtable_tests.items()))

if __name__ == '__main__':
	model_object = load_model('model_400_epochs.pickle')
	tests(model_object)
