import tensorflow as tf

BATCH_SIZE = 32
IMG_HEGIHT  = 224
IMG_WIDHT   = 224
IMG_CHANNEL = 3
BUFFER_SIZE = BATCH_SIZE*10

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image,channels=IMG_CHANNEL)
    
    input_image  = tf.image.resize(image, (IMG_HEGIHT, IMG_WIDHT))
    
    # Convert both images to float32 tensors
    input_image  = tf.cast(input_image, tf.float32)
    
    return input_image


def processing_image(input_image):
    input_image = tf.keras.applications.resnet50.preprocess_input(input_image)
    return input_image

def load_image_val(image_file):
    input_image = load(image_file)
    input_image = processing_image(input_image)

    return input_image