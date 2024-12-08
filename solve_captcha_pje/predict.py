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


def crop_image(image):
    # if not isinstance(image, Image):
    #     return 'Error on image type'
    
    width, height = image.size

    image_pieces = []
    step = 0
    count = 1
    while step < width:
        image_pieces.append(image.crop((step, 0, width-(250-step), height)))
        step += 50
        count += 1
    
    return image_pieces


def predict(image, model='model_400_epochs.pickle'):
    network = load_model(model)
    image_pieces = crop_image(image)

    captcha = ''

    for _image in image_pieces:
        image_np_array = np.array(_image.convert('L'), dtype=np.float32)/255.0
        pred = network.predict(np.array([image_np_array]))

        idx = np.argmax(pred)
        captcha += chr(idx)

    return captcha


if __name__ == '__main__':
    imagem_teste = Image.open('utils/captcha_teste.jpg').convert("RGBA")
    print(predict(imagem_teste, 'model_400_epochs.pickle'))
