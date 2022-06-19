import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from retinaface import RetinaFace

class BuildRes50Unet():
    def __init__(self):
        
        self.encoder_blocks_name = ["conv1_relu", "conv2_block3_out", "conv3_block4_out",
                                    "conv4_block6_out"]
        self.bridge_block_name = "conv5_block3_out"
#         self.encoder_blocks_name = ["input_1", "conv1_relu", "conv2_block3_out", "conv3_block4_out"]
#         self.bridge_block_name = "conv4_block6_out"
    
    def conv_block(self, inputs, num_filters):
        x = layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        return x
    
    
    def upsample_concate_block(self, inputs, skip_connection, num_filters):
        x = layers.Conv2DTranspose(filters=num_filters, kernel_size=(2,2), strides=2, padding='same')(inputs)
        x = layers.Concatenate()([skip_connection, x])
        x = self.conv_block(x, num_filters)
        
        return x
    
    
    def build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        
        # encoder
        backbone = tf.keras.applications.ResNet50(include_top=False, input_tensor=inputs,
                                                  weights='imagenet')
        eb0 = backbone.get_layer(name=self.encoder_blocks_name[0]).output
        eb1 = backbone.get_layer(name=self.encoder_blocks_name[1]).output
        eb2 = backbone.get_layer(name=self.encoder_blocks_name[2]).output
        eb3 = backbone.get_layer(name=self.encoder_blocks_name[3]).output
        
        # bridge
        br = backbone.get_layer(name=self.bridge_block_name).output
        
        # decoder
        db3 = self.upsample_concate_block(inputs=br, skip_connection=eb3, num_filters=512)
        db2 = self.upsample_concate_block(inputs=db3, skip_connection=eb2, num_filters=256)
        db1 = self.upsample_concate_block(inputs=db2, skip_connection=eb1, num_filters=128)
        db0 = self.upsample_concate_block(inputs=db1, skip_connection=eb0, num_filters=64)
        
        # final output
        first_feature = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')(inputs)
        final_feature = self.upsample_concate_block(inputs=db0, skip_connection=first_feature, num_filters=64)
        outputs = layers.Conv2D(filters=3, kernel_size=(1,1), activation='sigmoid')(final_feature)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return  model


IMG_HEGIHT  = 224
IMG_WIDHT   = 224
IMG_CHANNEL = 3

def load_model():
    model = BuildRes50Unet()
    res50Unet = model.build_model(input_shape=(IMG_HEGIHT, IMG_WIDHT, IMG_CHANNEL))
    res50Unet.load_weights('model.h5')
    return res50Unet

