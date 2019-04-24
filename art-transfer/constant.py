from keras import backend


height = 512
width = 512

SIZE = height * width

EPOCH = 5

CONTENT_WEIGHT = 0.025
STYLE_WEIGHT = 5.0
TOTAL_VARIATION_WEIGHT = 1.0

COMBINATION_IMAGE = backend.placeholder((1, height, width, 3))

IMAGE_DIR = 'images/'
IMAGE_FILE = IMAGE_DIR + 'deneme4.jpg'
STYLE_IMAGE_DIR = IMAGE_DIR + 'styles/'
STYLE_IMAGE_FILE = STYLE_IMAGE_DIR + 'picasso2.jpg'
