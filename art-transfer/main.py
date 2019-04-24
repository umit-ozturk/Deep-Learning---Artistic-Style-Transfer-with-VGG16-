from tools import get_content_image, get_style_image, total_variation_loss, content_loss, get_image_array, \
    get_input_tensor, get_result_image, style_loss, Evaluator
from constant import *
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from PIL import Image
from time import time
import numpy as np


# Eğer ssl hatası alıyorsa
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


def main():
    content_image = get_content_image()
    content_array = get_image_array(content_image)
    content_image = backend.variable(content_array)

    style_image = get_style_image()
    style_array = get_image_array(style_image)
    style_image = backend.variable(style_array)

    input_tensor = get_input_tensor(content_image, style_image)

    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    layers = dict([layer.name, layer.output] for layer in model.layers)

    loss = backend.variable(0.)

    layer_features = layers['block2_conv2']
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss += CONTENT_WEIGHT * content_loss(content_image_features, combination_features)

    feature_layers = ['block1_conv2', 'block2_conv2',
                      'block3_conv3', 'block4_conv3',
                      'block5_conv3']

    for layer_name in feature_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features)
        loss += (STYLE_WEIGHT / len(feature_layers)) * sl

    loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(COMBINATION_IMAGE)

    grads = backend.gradients(loss, COMBINATION_IMAGE)

    outputs = [loss]
    outputs += grads

    evaluator = Evaluator(outputs)

    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128

    for i in range(EPOCH):
        print('Start of iteration', i)
        start_time = time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
        imsave('results/' + str(i) + '.png', Image.fromarray(get_result_image(x)))
    imsave('results/last.png', Image.fromarray(get_result_image(x)))


if __name__ == '__main__':
    main()
