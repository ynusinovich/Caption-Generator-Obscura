from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
from tensorflow import Graph
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import load_model, Model, model_from_json
from tensorflow.python.keras.backend import set_session
from werkzeug.utils import secure_filename
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


app = Flask(__name__, template_folder = 'templates')

tf_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 2)

app.config['UPLOAD_FOLDER'] = './static/uploads/'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = ['PNG', 'JPG', 'JPEG']

# restrict uploads
def allowed_file(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit('.', 1)[1]
    if ext.upper() in app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return True
    else:
        return False
# load vocab
def load_obj(name):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)
vocab = load_obj("vocab")
vocab_size = len(vocab) + 1
# load inception for image processing
graph1 = Graph()
with graph1.as_default():
    session1 = tf.compat.v1.Session(config = tf_config)
    with session1.as_default():
        with open('pre_model.json') as json_file:
            pre_model = model_from_json(json_file.read())
        pre_model.load_weights('pre_model_weights.h5')
# load vocabulary indices
index_to_word = {}
word_to_index = {}
index = 1
for word in vocab:
    word_to_index[word] = index
    index_to_word[index] = word
    index += 1
max_length = 40
# load model for combining text and images
graph2 = Graph()
with graph2.as_default():
    session2 = tf.compat.v1.Session(config = tf_config)
    with session2.as_default():
        with open('model.json') as json_file:
            model = model_from_json(json_file.read())
        model.load_weights('model_weights.h5')
# create caption generator
def captionGenerator(a_photo, a_randomness):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen = max_length)
        with graph2.as_default():
            set_session(session2)
            yhat = model.predict([a_photo, sequence], verbose = 0)
        yhat = np.random.choice((-yhat).argsort(axis = -1)[:, :a_randomness][0])
        word = index_to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

@app.route("/", methods = ['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('main.html')
    if request.method == 'POST':
        if request.files:
            an_image = request.files['filename']
            if an_image.filename == "":
                print("Image must have a filename.")
                return redirect(request.url)
            if not allowed_file(an_image.filename):
                print("That image extension is not allowed. Try a .jpeg, .jpg, or .png file.")
                return redirect(request.url)
            else:
                filename = secure_filename(an_image.filename)
                image_path = app.config['UPLOAD_FOLDER'] + filename
                an_image.save(image_path)
                print("Image is saved!")
                # Convert all the images to size 299 x 299 as expected by the InceptionV3 Model
                x = image.load_img(image_path, target_size = (299, 299))
                # Convert PIL image to numpy array of 3-dimensions
                x = image.img_to_array(x)
                # Add one more dimension
                x = np.expand_dims(x, axis = 0)
                # Preprocess images using preprocess_input() from inception module
                x = preprocess_input(x)
                with graph1.as_default():
                    set_session(session1)
                    x = pre_model.predict(x, verbose = 0)
                x = np.reshape(x, x.shape[1])
                x = x.reshape(1, 2048)
                random = int(request.form['randomness'])
                caption = captionGenerator(x, random)
                return render_template('main.html', original_randomness = random, original_path = image_path, result = caption)

if __name__ == "__main__":
    app.run(debug = True)
