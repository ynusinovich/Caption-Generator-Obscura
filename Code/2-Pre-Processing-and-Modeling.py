#!/usr/bin/env python
# coding: utf-8

# # Capstone Project:
# # Pre-Processing and Modeling

# ## Method References

# - https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8<br>
# - https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/<br>
# - https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
import os
import string
import pickle
import math
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Bidirectional, Embedding
from keras.layers.merge import add
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, optimizers

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ## Process Data for Modeling

# ### Generate Caption Sequence Dictionary and Vocabulary List

# In[2]:


# for first run

# caption_df = pd.read_csv("../Data/atlas_edits_clean.csv")
# caption_df.rename(columns = {"Unnamed: 0": "image_number"}, inplace = True)


# In[3]:


# for first run

# caption_dict = dict()
# for i in range(0,len(caption_df.index)):
#     caption_dict[caption_df.loc[i, "image_number"]] = caption_df.loc[i, "description"]


# In[4]:


# for first run

# table = str.maketrans('', '', string.punctuation)
# # characters to replace, characters to replace them with, characters to delete

# for image_number, description in caption_dict.items():
#     # tokenize
#     description = description.split()
#     # convert to lower case
#     description = [word.lower() for word in description]
#     # remove punctuation from each token
#     description = [word.translate(table) for word in description]
#     # remove hanging 's' and 'a'
#     description = [word for word in description if len(word) > 1]
#     # remove tokens with numbers in them
#     description = [word for word in description if word.isalpha()]
#     # store as string
#     description = ' '.join(description)
#     # save in dict
#     caption_dict[image_number] =  description


# In[5]:


# for first run

# # Create a list of all the training captions
# all_captions = []
# for image_number, description in caption_dict.items():
#     all_captions.append(description)

# # Consider only words which occur at least 3 times in the corpus
# word_count_threshold = 3
# word_counts = dict()
# num_sentences = 0
# for sent in all_captions:
#     num_sentences += 1
#     for w in sent.split(' '):
#         word_counts[w] = word_counts.get(w, 0) + 1 # add one to the count of the word

# vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

# print(f'Preprocessed words {len(vocab)} ')
# print(f'Number of sentences {num_sentences} ')


# In[6]:


# for first run

# def save_obj(obj, name):
#     with open('../Obj/' + name + '.pickle', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
# save_obj(vocab, "vocab")


# In[7]:


# for subsequent runs

def load_obj(name):
    with open('../Obj/' + name + '.pickle', 'rb') as f:
        return pickle.load(f)
    
vocab = load_obj("vocab")


# In[8]:


vocab_size = len(vocab) + 1


# In[9]:


# for first run

# for image_number, description in caption_dict.items():
#     tokens = description.split()
#     caption_dict[image_number] = 'startseq ' + ' '.join(tokens) + ' endseq'
#     print(f"Image {image_number} added to caption dictionary")


# In[10]:


# for first run

# def save_obj(obj, name):
#     with open('../Obj/' + name + '.pickle', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
# save_obj(caption_dict, "caption_dict")


# In[11]:


# for subsequent runs
    
caption_dict = load_obj("caption_dict")


# ### Run Images Through Inception V3 Model

# In[12]:


# Get the InceptionV3 model trained on imagenet data
model = InceptionV3(weights = 'imagenet')
# Remove the last layer (output softmax layer) from the InceptionV3
model_new = Model(model.input, model.layers[-2].output)


# In[13]:


# for first run

# pre_image_dict = dict()

# for filename in os.listdir("../Data/Atlas_Images/"):
#     if filename.endswith(".jpg"): 
#         image_path = os.path.join("../Data/Atlas_Images/", filename)
#         # Convert all the images to size 299 x 299 as expected by the InceptionV3 Model
#         img = image.load_img(image_path, target_size = (299, 299))
#         # Convert PIL image to numpy array of 3-dimensions
#         x = image.img_to_array(img)
#         # Add one more dimension
#         x = np.expand_dims(x, axis = 0)
#         # preprocess images using preprocess_input() from inception module
#         x = preprocess_input(x)
#         # add to pre-imagedict
#         image_number = int(filename.split(".")[0])
#         pre_image_dict[image_number] = x
#         print(f"Image {image_number} added to pre-image dictionary")


# In[14]:


# for first run

# save_obj(pre_image_dict, "pre_image_dict")


# In[15]:


# for first run

# image_dict = dict()

# for filename in os.listdir("../Data/Atlas_Images/"):
#     if filename.endswith(".jpg"): 
#         image_path = os.path.join("../Data/Atlas_Images/", filename)
#         image_number = int(filename.split(".")[0])
#         pre_image_dict[image_number] = x
#         feature = model_new.predict(x, verbose = 0)
#         feature = np.reshape(feature, feature.shape[1])
#         image_dict[image_number] = feature
#         print(f"Image {image_number} added to image dictionary")


# In[16]:


# for first run

# save_obj(image_dict, "image_dict")


# In[17]:


# for subsequent runs
    
image_dict = load_obj("image_dict")


# In[18]:


image_dict


# ### Create Word Index and Find Maximum Caption Length

# In[19]:


index_to_word = {}
word_to_index = {}
index = 1
for word in vocab:
    word_to_index[word] = index
    index_to_word[index] = word
    index += 1


# In[20]:


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(a_caption_dict):
    all_desc = list()
    for key in a_caption_dict.keys():
        all_desc.append(a_caption_dict[key])
    return all_desc

# calculate the length of the description with the most words
def max_length(a_caption_dict):
    lines = to_lines(a_caption_dict)
    return max(len(caption.split()) for caption in lines)

# determine the maximum sequence length
max_length = max_length(caption_dict)
print(f'Max Caption Length, in Words: {max_length}')


# ### Create Data Generator

# In[21]:


def data_generator(a_caption_dict, an_image_dict, a_word_to_index, a_max_length, a_num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n = 0
    # loop forever over images
    while 1:
        for image_number, caption in a_caption_dict.items():
            n += 1
            # retrieve the photo features
            photo_data = an_image_dict[image_number]
            # encode the sequence
            seq = [a_word_to_index[word] for word in caption.split(' ') if word in a_word_to_index]
            # split one sequence into multiple X, y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen = a_max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
                # store
                X1.append(photo_data)
                X2.append(in_seq)
                y.append(out_seq)
            # yield the batch data
            if n == a_num_photos_per_batch:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n = 0


# ### Embed Captions with Pre-Trained GloVe Vector

# In[22]:


# Load GloVe vectors
glove_dir = '../Data/'
embeddings_index = dict()
file = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding = "utf-8")
for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
file.close()


# In[23]:


embedding_dim = 200
# Get 200-dim dense matrix for each of the words in our vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in word_to_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[index] = embedding_vector


# ### Run Image Data Through Custom Model

# In[24]:


# image feature extractor model
inputs1 = Input(shape = (2048,))
fe1 = Dropout(0.05)(inputs1)
fe2 = Dense(256, activation = 'relu')(fe1)

# partial caption sequence model
inputs2 = Input(shape = (max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero = True)(inputs2)
se2 = Dropout(0.1)(se1)
se3 = LSTM(256)(se2)

# decoder (feed forward) model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(0.005))(decoder1)
outputs = Dense(vocab_size, activation = 'softmax')(decoder2)

# merge the two input models
model = Model(inputs = [inputs1, inputs2], outputs = outputs)


# In[25]:


model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False


# In[26]:


model.summary()


# In[27]:


opt = optimizers.Adam(learning_rate = 0.005)
model.compile(loss = 'categorical_crossentropy', optimizer = opt)


# ## Train Model

# In[ ]:


# run this as .py file

num_photos_per_batch = 16

generator = data_generator(caption_dict, image_dict, word_to_index, max_length, num_photos_per_batch)

early_stop = EarlyStopping(monitor = "loss", 
                           patience = 5,
                           min_delta = 0,
                           restore_best_weights = True)

model.fit_generator(generator, epochs = 100, verbose = 1,
                    steps_per_epoch = math.ceil(len(caption_dict)/num_photos_per_batch),
                    callbacks = [early_stop])


# In[ ]:


model.save("../Obj/final_model_2.h5")


# In[ ]:




