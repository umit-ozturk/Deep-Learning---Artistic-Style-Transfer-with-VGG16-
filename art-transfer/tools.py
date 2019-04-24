from constant import *
from keras import backend
import numpy as np

from PIL import Image


class Evaluator(object):
	def __init__(self, outputs):
		self.outputs = outputs
		self.f_outputs = backend.function([COMBINATION_IMAGE], outputs)
		self.loss_value = None
		self.grads_values = None

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x, self.f_outputs)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values


def get_content_image():
	""" Acıklama gir """
	return Image.open(IMAGE_FILE).resize((width, height))


def get_style_image():
	""" Acıklama gir """
	return Image.open(STYLE_IMAGE_FILE).resize((width, height))


def get_image_array(image):
	""" Acıklama gir """
	image_array = np.asarray(image, dtype='float32')
	image_array = np.expand_dims(image_array, axis=0)
	image_array[:, :, :, 0] -= 103.939
	image_array[:, :, :, 1] -= 116.779
	image_array[:, :, :, 2] -= 123.68
	image_array = image_array[:, :, :, ::-1]
	return image_array


def get_input_tensor(content_image, style_image):
	""" Acıklama gir """
	return backend.concatenate([content_image, style_image, COMBINATION_IMAGE], axis=0)


def get_result_image(x):
	""" Acıklama gir """
	x = x.reshape((height, width, 3))
	x = x[:, :, ::-1]
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	x = np.clip(x, 0, 255).astype('uint8')
	return x


def content_loss(content, combination):
	""" Acıklama gir """
	return backend.sum(backend.square(combination - content))


def total_variation_loss(x):
	""" Acıklama gir """
	a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
	b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
	return backend.sum(backend.pow(a + b, 1.25))


def gram_matrix(x):
	features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
	gram = backend.dot(features, backend.transpose(features))
	return gram


def style_loss(style, combination):
	s_gram = gram_matrix(style)
	c_gram = gram_matrix(combination)
	return backend.sum(backend.square(s_gram - c_gram)) / (4. * (CHANNELS ** 2) * (SIZE ** 2))


def eval_loss_and_grads(x, f_outputs):
	x = x.reshape((1, height, width, 3))
	outs = f_outputs([x])
	loss_value = outs[0]
	grad_values = outs[1].flatten().astype('float64')
	return loss_value, grad_values
