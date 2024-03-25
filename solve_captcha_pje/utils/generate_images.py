import os
import json
import requests

from base64 import b64decode
from io import BytesIO
from PIL import Image
from uuid import uuid1 # isso faz sentido? possivelmente não, mas aqui é freestyle


def get_captcha(folder_captchas):
	headers = {
		'authority': 'juris.trt2.jus.br',
		'accept': 'application/json, text/plain, */*',
		'accept-language': 'en-US,en;q=0.9',
		'content-type': 'application/json',
		'referer': 'https://juris.trt2.jus.br/jurisprudencia/',
		'sec-ch-ua': '"Chromium";v="111", "Not(A:Brand";v="8"',
		'sec-ch-ua-mobile': '?0',
		'sec-ch-ua-platform': '"Linux"',
		'sec-fetch-dest': 'empty',
		'sec-fetch-mode': 'cors',
		'sec-fetch-site': 'same-origin',
		'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
	}

	response = requests.get('https://juris.trt2.jus.br/juris-backend/api/captcha', headers=headers)
	response_json = json.loads(response.content)
	token_captcha = response_json['tokenDesafio']
	img_bytes = b64decode(response_json['imagem'])
	img_bytesIO = BytesIO(img_bytes) # https://jdhao.github.io/2020/03/17/base64_opencv_pil_image_conversion/
	
	img = Image.open(img_bytesIO)
	img.save(f'{folder_captchas}{uuid1().hex}.png')


def crop_captcha(folder_captchas, img_filename, folder_crops):
	filename, extension = os.path.splitext(f'{img_filename}')

	img = Image.open(f'{folder_captchas}{img_filename}')
	width, height = img.size

	step = 0
	count = 1
	while step < width:
		img.crop((step, 0, width-(250-step), height)).save(f'{folder_crops}{filename}_{count}{extension}')
		step += 50
		count += 1


def main(option=None):
	if option is None:
		return

	for _ in range(10):
		get_captcha(f'data/{option}/captchas/')

	for filename in os.listdir(f'data/{option}/captchas/'):
		crop_captcha(f'data/{option}/captchas/', filename, f'data/{option}/crops/')


if __name__ == '__main__':
	main(option='train')
