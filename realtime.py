import cv2
import functools
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time


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
style_list = [
    "https://d3d00swyhr67nd.cloudfront.net/w800h800/collection/TATE/TATE/TATE_TATE_T05010_10-001.jpg",
    "https://cdn.britannica.com/17/196817-050-6A15DAC3/vegetables.jpg",
    "https://media.npr.org/assets/img/2012/04/30/scream_custom-9ef574d2014bd441879317ecf242ad060e34e743-s1100-c50.jpg",
    "https://cdn.cdaction.pl/images/2021/11/19/bde85c8d-7397-42ba-bcdd-e7d80c2789ac.jpeg?preset=medium"]
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)
FPS = 1
t = time.time()
ret, upsized = cap.read()
height, width, channels = upsized.shape
CAP_FPS = False
counter = 0
SIZE = 512
SWITCH_EVERY_FRAMES = 5
while True:
    if time.time() - t > 1 / FPS or not CAP_FPS:
        style_image_url = style_list[counter // SWITCH_EVERY_FRAMES % len(style_list)]
        style_img_size = (SIZE, SIZE)  # Recommended to keep it at 256.
        style_image = load_image(style_image_url, style_img_size)
        style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
        counter += 1
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to 256x256 pixels
        INPUT_SIZE = int(SIZE)
        frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        content_image = frame.astype(np.float32)[np.newaxis, ...] / 255.
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]
        stylized_image = stylized_image[0].numpy()
        upsized = cv2.resize(stylized_image, (width, height), interpolation=cv2.INTER_AREA)
        upsized = cv2.cvtColor(upsized, cv2.COLOR_RGB2BGR)

        t = time.time()
    cv2.imshow("Live Video", upsized)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
