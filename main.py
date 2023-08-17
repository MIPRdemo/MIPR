import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
 
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth=True  
session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)
# KTF.set_session(session)

import numpy
import keras
from keras.utils.np_utils import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average,Lambda

from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers #keras2
from keras.utils import plot_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.optimizers import *

from Hypers import *
from Utils import *
from Preprocessing import *
from Generator import *
from Models import *
import nltk

data_root_path = "newsRec/MIND-small"
embedding_path = ""

news,news_index,category_dict,subcategory_dict,word_dict,content_dict,entity_dict = read_news(data_root_path)
news_title,news_vert,news_subvert,news_entity,news_content=get_doc_input(news,news_index,category_dict,subcategory_dict,word_dict,content_dict,entity_dict)

title_word_embedding_matrix, have_word = load_matrix(embedding_path,word_dict)
content_word_embedding_matrix, have_word = load_matrix(embedding_path,content_dict)

train_session = read_train_clickhistory(news_index,data_root_path,'train/behaviors.tsv')
train_user = parse_user(news_index,train_session)
train_sess, train_user_id, train_user_interest, train_entity_interest, mask, train_label = get_train_input(news_index,train_session)

news_fetcher = NewsFetcher(news_title,news_content,news_vert,news_subvert,news_entity)

test_session = read_test_clickhistory_noclk(news_index,data_root_path,'test/behaviors.tsv')
# test_session = read_test_clickhistory(news_index,data_root_path,'test/behaviors.tsv')
test_user = parse_user(news_index,test_session)
test_impressions, test_userids, test_user_interest, test_entity_interest = get_test_input(news_index,test_session)

train_generator = get_hir_train_generator(news_fetcher,train_user['click'],train_user_id,train_user_interest, train_entity_interest, mask, train_sess,train_label,32)
news_generator = get_hir_news_generator(news_fetcher,32)

model,news_encoder,user_encoder, user_encoder_inter, entity_encoder_inter = create_model(title_word_embedding_matrix,content_word_embedding_matrix,entity_dict,category_dict,subcategory_dict)
model.fit_generator(train_generator,epochs=3,verbose=1)

test_user_interest = np.array(test_user_interest)
# test_user_interest = user_encoder_inter.predict(test_user_interest,verbose=1)


news_scoring = news_encoder.predict_generator(news_generator,verbose=1)
test_user_generator = get_hir_user_generator(news_fetcher,test_user_interest,test_user['click'],32)
test_user_scoring = user_encoder.predict_generator(test_user_generator,verbose=1)
# dump_result(test_impressions,news_scoring,test_user_scoring)
AUC, MRR, nDCG5, nDCG10 = evaluate(test_impressions,news_scoring,test_user_scoring)
print(" AUC: {}\n MRR: {}\n nDCG@5: {}\n nDCG@10: {}".format(AUC, MRR, nDCG5, nDCG10))
