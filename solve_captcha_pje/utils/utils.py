import numpy as np


def to_categorical(y, num_classes=None, dtype=np.float32):
	if num_classes is None:
		num_classes = np.max(y) + 1
	categorical = np.zeros((len(y), num_classes), dtype=dtype)
	categorical[np.arange(len(y)), y] = 1
	return categorical