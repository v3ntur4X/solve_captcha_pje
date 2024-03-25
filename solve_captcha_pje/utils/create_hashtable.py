import os
import json


# cgilopqz01
def create_hashtable(folder, file_json):
	with open(f'{folder}{file_json}') as fp:
		json_file = json.load(fp)
	hashtable_images = json.loads(json_file)

	for filename in os.listdir(f'{folder}crops/'):
		if hashtable_images.get(filename) is None:
			hashtable_images[filename] = ''


	dict_json = json.dumps(hashtable_images)
	with open(f'{folder}{file_json}', 'w') as fp:
		json.dump(dict_json, fp)


def main(option=None):
	if option is None:
		return

	create_hashtable(f'data/{option}/', f'hashtable_{option}.json')


if __name__ in '__main__':
	main('train')
