# Preprocessing
import os
import pathlib
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Deep Learning
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D, Dense, Reshape, Input, Flatten
from flask import Flask, flash, request, redirect, url_for, render_template
from skimage.io import imsave
from werkzeug.utils import secure_filename
from skimage.transform import resize
from PIL import Image, ImageChops


best_model_weights_path = "C:\\Users\\shris\\Downloads\\HCI_Model\\model\\cnn_model_last.h5"
app = Flask(__name__)
app.secret_key = "hello"
img_height = 128
img_width = 128
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = 'C:\\Users\\shris\\Downloads\\HCI_Model\\static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def trim(im):
    return im.resize((200,200))

def get_cnn_model():
    model = tf.keras.Sequential([        
        # CONV 1
        Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
        Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV2
        Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),

        #CONV3
        Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV4
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV5 (padding=2)
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV6 (padding=2)
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=(1,1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV7
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        BatchNormalization(),

        # CONV8
        Conv2DTranspose(filters=256, kernel_size=4, strides=(2,2), padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu'),
        Conv2D(filters=313, kernel_size=1, strides=(1,1), padding='valid'),
        # Softmax(axis=1), This layer was commented from the original model

        # OUTPUT
        Conv2D(filters=2, kernel_size=1, padding='valid', dilation_rate=1, strides=(1,1), use_bias=False),
        UpSampling2D(size=4, interpolation='bilinear'),
    ])
    
    # Show model summary
    model.build()
    print(model.summary())

    return model

def preprocess_lab(lab):
    with tf.name_scope('preprocess_lab'):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope('deprocess_lab'):
        #TODO This is axis=3 instead of axis=2 when deprocessing batch of images 
               # ( we process individual images but deprocess batches)
        #return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=2)

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def rgb_to_lab(srgb):
    # based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
    with tf.name_scope('rgb_to_lab'):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        with tf.name_scope('srgb_to_xyz'):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope('xyz_to_cielab'):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope('lab_to_rgb'):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])
        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope('cielab_to_xyz'):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope('xyz_to_srgb'):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

@tf.function
def get_image_lab(img_path, height=img_height, width=img_width):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width])
       
    image_lab = rgb_to_lab(img / 255.0)
    # L_chan, a_chan, b_chan = preprocess_lab(image_lab)
    # image_lab = deprocess_lab(L_chan, a_chan, b_chan)
    # image_lab = (image_lab) # + [0, 128, 128]) / [100, 255, 255]

    return image_lab

@tf.function
def get_l_ab_channels(img_path, height=img_height, width=img_width):
    image_lab = get_image_lab(img_path, height, width)
    
    image_l = tf.expand_dims(image_lab[:,:,0], -1)
    image_ab = image_lab[:,:,1:]

    return image_l, image_ab

def predict_and_show(image_path):
  image_to_predict_lab = get_l_ab_channels(image_path)

  # Use only L channel (grayscale) to predict
  image_to_predict = tf.expand_dims(image_to_predict_lab[0], 0)

  # Predict
  prediction = best_model.predict(image_to_predict,  verbose=1)[0]

  original_img = np.concatenate((image_to_predict_lab[0], image_to_predict_lab[1]), axis=2)
  original_img = lab_to_rgb(original_img).numpy()

  predicted_img = np.concatenate((image_to_predict[0], prediction), axis=2)
  predicted_img = lab_to_rgb(predicted_img).numpy()
  x = np.array(Image.fromarray((predicted_img * 255).astype(np.uint8)).resize((200, 200)).convert('RGB'))
  imsave("static\\output.jpg", x)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

best_model = get_cnn_model()
best_model.load_weights(best_model_weights_path)

@app.route('/', methods=['GET', 'POST'])
def colorize():
    if request.method == 'POST':
        try:
            url = request.form['url']
            if 'examples' in url:
                return render_template('index.html', res=1)
        except:
            print()
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            s = "C:\\Users\\shris\\Downloads\\HCI_Model\\static\\" + filename
            predict_and_show(s)
            trim(Image.open(s)).save(s)
            return render_template('index.html', res="static/output.jpg", og="static/" + filename)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
