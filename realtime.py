import cv2
import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    img = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


cap = cv2.VideoCapture(0)
style_image_url = 'https://media.vogue.fr/photos/5c8a55363d44a0083ccbef54/2:3/w_2560%2Cc_limit/GettyImages-625257378.jpg'  # @param {type:"string"}
style_img_size = (256, 256)  # Recommended to keep it at 256.
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 2, (512, 512))
counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Resize the frame to 256x256 pixels
    INPUT_SIZE = 512
    frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    content_image = frame.astype(np.float32)[np.newaxis, ...] / 255.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    upsized = cv2.resize(stylized_image[0].numpy(), (512, 512))
    upsized = cv2.cvtColor(upsized, cv2.COLOR_RGB2BGR)

    depthed_upscaled_frame = cv2.convertScaleAbs(upsized)
    out.write(depthed_upscaled_frame)

    cv2.imshow("Live Video", upsized)
    if cv2.waitKey(1) == ord('q') or counter > 10:
        break
    counter += 1

cap.release()
cv2.destroyAllWindows()
