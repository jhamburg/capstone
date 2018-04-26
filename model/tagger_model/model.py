from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, concatenate, Concatenate, TimeDistributed, Dense, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras_contrib.layers import CRF
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint

import os
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_model(seq_input_len, n_word_input_nodes, n_tag_input_nodes, 
                 n_word_embedding_nodes, n_tag_embedding_nodes, n_RNN_nodes, 
                 n_dense_nodes, stateful=False, batch_size=None, 
                 recurrent_dropout=0.1, crf=False, drop_out=.2):
    """Create a model for both POS and NER tagging.  Can try CRF but if that doesn't work,
       can default back to softmax"""

    #Layers 1
    word_input = Input(batch_shape=(batch_size, seq_input_len), name='word_input_layer')
#     tag_input = Input(batch_shape=(batch_size, seq_input_len), name='tag_input_layer')

    #Layers 2
    #mask_zero will ignore 0 padding
    word_embeddings = Embedding(input_dim=n_word_input_nodes,
                                output_dim=n_word_embedding_nodes, 
                                mask_zero=True, name='word_embedding_layer')(word_input) 
    #Output shape = (batch_size, seq_input_len, n_word_embedding_nodes)
#     tag_embeddings = Embedding(input_dim=n_tag_input_nodes,
#                                output_dim=n_tag_embedding_nodes,
#                                mask_zero=True,
#                                name='tag_embedding_layer')(tag_input) 
    #Output shape = (batch_size, seq_input_len, n_tag_embedding_nodes)

    #Layer 3
#     merged_embeddings = Concatenate(axis=-1, name='concat_embedding_layer')([word_embeddings, tag_embeddings])
#     merged_embeddings = concatenate([word_embeddings, tag_embeddings], name='concat_embedding_layer')
    #Output shape =  (batch_size, seq_input_len, n_word_embedding_nodes + n_tag_embedding_nodes)

#     drop_out1 = Dropout(drop_out)(merged_embeddings)
#     drop_out1 = Dropout(drop_out)(word_embeddings)
    
    #Layer 4
    hidden_layer = Bidirectional(GRU(units=n_RNN_nodes, return_sequences=True, 
                                      recurrent_dropout=recurrent_dropout, dropout=drop_out,
                                     stateful=stateful, name='hidden_layer'))(word_embeddings)
    #Output shape = (batch_size, seq_input_len, n_hidden_nodes)

#     drop_out2 = TimeDistributed(Dropout(drop_out))(hidden_layer)
    
#     #Layer 5
    dense_layer = TimeDistributed(Dense(units=n_dense_nodes, activation='relu'), name='dense_layer')(hidden_layer)

#     drop_out3 = TimeDistributed(Dropout(drop_out))(dense_layer)
    
    #Layer 6
    if crf:
#         crf = CRF(units=n_tag_input_nodes, learn_mode = 'marginal',
        crf = CRF(units=n_tag_input_nodes,
                  sparse_target=True, name='output_layer')
        output_layer = crf(dense_layer)
        loss = crf.loss_function
        acc = crf.accuracy
    else:
        output_layer = TimeDistributed(Dense(units=n_tag_input_nodes, activation='softmax'), 
                                       name='output_layer')(dense_layer)
        loss = "sparse_categorical_crossentropy" 
        acc = 'acc'
    # Output shape = (batch_size, seq_input_len, n_tag_input_nodes)

    #Specify which layers are input and output, compile model with loss and optimization functions
#     model = Model(inputs=[word_input, tag_input], outputs=output_layer)
    model = Model(inputs=[word_input], outputs=output_layer)
    model.compile(loss=loss, optimizer="adam", metrics=[acc])

    return model 


def run_training_model(pad_words, pad_tags, y_dat, saveName, lexicon, 
                       batch_size=128, epochs=15, val_split=.2, 
                       print_summary=False, savePath='models', 
                       n_word_embedding_nodes=200,
                       n_tag_embedding_nodes=200,
                       n_RNN_nodes=300, 
                       n_dense_nodes=100,
                       crf=False):    
    """ Builds and fits a model"""
    
    model = create_model(seq_input_len=pad_words.shape[-1] - 1, #substract 1 from matrix length because of offset
                             n_word_input_nodes=len(lexicon.words_lexicon) + 1, #Add one for 0 padding
                             n_tag_input_nodes=len(lexicon.tags_lexicon) + 1, #Add one for 0 padding
                             n_word_embedding_nodes=n_word_embedding_nodes,
                             n_tag_embedding_nodes=n_tag_embedding_nodes,
                             n_RNN_nodes=n_RNN_nodes, 
                             n_dense_nodes=n_dense_nodes,
                             crf=crf)
        
    if print_summary:
        model.summary()

    filepath = os.path.join(savePath, saveName + '.hdf5')
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)
    callbacks_list = [checkpoint]

    '''Train the model'''

    # output matrix (y) has extra 3rd dimension added because sparse cross-entropy 
    # function requires one label per row
#     model.fit(x=[pad_words[:,1:], pad_tags[:,:-1]], 
    model.fit(x=[pad_words[:,1:]], 
              y=pad_tags[:, 1:, None], batch_size=batch_size, 
              epochs=epochs, validation_split=val_split,
              callbacks=callbacks_list)

    return model


def create_test_model(loadName, lexicon, loadPath='models',
                      n_word_embedding_nodes=200,
                      n_tag_embedding_nodes=200,
                      n_RNN_nodes=300, 
                      n_dense_nodes=100,
                      crf=False):
    """ Loads a model to predict new data"""
    
    model = create_model(seq_input_len=1,
                         n_word_input_nodes=len(lexicon.words_lexicon) + 1, #Add one for 0 padding
                         n_tag_input_nodes=len(lexicon.tags_lexicon) + 1, #Add one for 0 padding
                         n_word_embedding_nodes=n_word_embedding_nodes,
                         n_tag_embedding_nodes=n_tag_embedding_nodes,
                         n_RNN_nodes=n_RNN_nodes, 
                         n_dense_nodes=n_dense_nodes,
                         stateful=True, batch_size=1,
                         crf=crf)

    model.load_weights(os.path.join(loadPath, loadName + '.hdf5'))
    return model


def predict_new_tag(predictor_model, test, lexicon):
    """Predict tags for new data"""
    pred_tags = []
    for _, sent in test.iterrows():
        tok_sent = sent['sents']
        sent_idxs = sent['sent_indx']
        sent_pred_tags = []
#         prev_tag = 1  #initialize predicted tag sequence with padding
        prev_tag = 0  #initialize predicted tag sequence with padding
        for cur_word in sent_idxs:
            # cur_word and prev_tag are just integers, but the model expects an input array
            # with the shape (batch_size, seq_input_len), so prepend two dimensions to these values
            p_next_tag = predictor_model.predict(x=[np.array(cur_word)[None, None]])[0]            
#             p_next_tag = predictor_model.predict(x=[np.array(cur_word)[None, None],
#                                                     np.array(prev_tag)[None, None]])[0]
            prev_tag = np.argmax(p_next_tag, axis=-1)[0]
            sent_pred_tags.append(prev_tag)
        predictor_model.reset_states()

        #Map tags back to string labels
        sent_pred_tags = [lexicon.indx_to_tags_dict[tag] for tag in sent_pred_tags]
        pred_tags.append(sent_pred_tags) #filter padding 

    return pred_tags


def evaluate_model(pred_tags, test, print_sample=False):
    """Evaluate predictions against a test set"""
    
    test = test.copy()
    test['predicted_tags'] = pred_tags
    
    if print_sample:
        for _, sent in test.sample(n=10).iterrows():
            print("SENTENCE:\t{}".format("\t".join(sent['sents'])))
            print("PREDICTED:\t{}".format("\t".join(sent['predicted_tags'])))
            print("GOLD:\t\t{}".format("\t".join(sent['tags'])))
            print("CORRECT:\t{}".format("\t".join([str(x) for x in np.array(sent['tags']) == np.array(sent['predicted_tags'])])), "\n\n")

    
    all_gold_tags = [tag for sent_tags in test['tags'] for tag in sent_tags]
    all_pred_tags = [tag for sent_tags in test['predicted_tags'] for tag in sent_tags]
    accuracy = accuracy_score(y_true=all_gold_tags, y_pred=all_pred_tags)
    precision = precision_score(y_true=all_gold_tags, y_pred=all_pred_tags, average='weighted')
    recall = recall_score(y_true=all_gold_tags, y_pred=all_pred_tags, average='weighted')
    f1 = f1_score(y_true=all_gold_tags, y_pred=all_pred_tags, average='weighted')

    print("ACCURACY: {:.3f}".format(accuracy))
    print("PRECISION: {:.3f}".format(precision))
    print("RECALL: {:.3f}".format(recall))
    print("F1: {:.3f}".format(f1))
    