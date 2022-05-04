# source: https://towardsdatascience.com/hugging-face-transformers-fine-tuning-distilbert-for-binary-classification-tasks-490f1d192379

import os

from DataLoader import DataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import DistilBertConfig, DistilBertTokenizerFast, TFDistilBertModel

EPOCHS = 100
EPOCHS_SMALL = 20
BATCH_SIZE = 32
DISTILBERT_DROPOUT = 0.2
DISTILBERT_ATT_DROPOUT = 0.2
LAYER_DROPOUT = 0.2
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
RANDOM_STATE = 42


def batch_encode(tokenizer, texts, batch_size=256, max_length=MAX_LENGTH):
    """
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed
    into a pre-trained transformer model.

    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """
    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='longest',
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False)
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


def initialize_base_model():
    config = DistilBertConfig(dropout=DISTILBERT_DROPOUT,
                              attention_dropout=DISTILBERT_ATT_DROPOUT,
                              output_hidden_states=True)

    # Bare, pre-trained DistilBERT model without a specific classification head
    distilBERT = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

    # Make DistilBERT layers untrainable
    for layer in distilBERT.layers:
        layer.trainable = False

    return distilBERT


def build_model(transformer, max_length=MAX_LENGTH):
    """
    Template for building a model off of the BERT or DistilBERT architecture
    for a binary classification task.

    Input:
      - transformer:  a base Hugging Face transformer model object (BERT or DistilBERT)
                      with no added classification head attached.
      - max_length:   integer controlling the maximum number of encoded tokens
                      in a given sequence.

    Output:
      - model:        a compiled tf.keras.Model with added classification layers
                      on top of the base pre-trained model architecture.
    """
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=RANDOM_STATE)

    input_ids_layer = tf.keras.layers.Input(shape=(max_length, ), name='input_ids', dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(max_length, ),
                                                  name='input_attention',
                                                  dtype='int32')

    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]
    cls_token = last_hidden_state[:, 0, :]

    output = tf.keras.layers.Dense(1,
                                   activation='sigmoid',
                                   kernel_initializer=weight_initializer,
                                   kernel_constraint=None,
                                   bias_initializer='zeros')(cls_token)

    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)
    model.compile(tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def single_distilbert_test(X, y, epochs=EPOCHS):
    X = np.array(X)
    y = np.array(y)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                          y,
                                                          test_size=0.2,
                                                          random_state=RANDOM_STATE)
    # X_valid, X_test, y_valid, y_test = train_test_split(X_valid,
    #                                                     y_valid,
    #                                                     test_size=0.5,
    #                                                     random_state=RANDOM_STATE)

    X_train_ids, X_train_attention = batch_encode(tokenizer, X_train.tolist())
    X_valid_ids, X_valid_attention = batch_encode(tokenizer, X_valid.tolist())
    # X_test_ids, X_test_attention = batch_encode(tokenizer, X_test.tolist())

    model = build_model(initialize_base_model())
    num_steps = len(X_train) // BATCH_SIZE
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(x=[X_train_ids, X_train_attention],
                        y=y_train,
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        steps_per_epoch=num_steps,
                        validation_data=([X_valid_ids, X_valid_attention], y_valid),
                        callbacks=[early_stopping],
                        verbose=2)

    return history.history['val_accuracy'][-1]


def single_distilbert_consistency_test(X_orig, y_orig, X_aug, y_aug, epochs=EPOCHS):
    X_orig = np.array(X_orig)
    y_orig = np.array(y_orig)
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    X_orig_ids, X_orig_attention = batch_encode(tokenizer, X_orig.tolist())
    X_aug_ids, X_aug_attention = batch_encode(tokenizer, X_aug.tolist())

    model = build_model(initialize_base_model())
    num_steps = len(X_orig) // BATCH_SIZE
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)
    history = model.fit(x=[X_orig_ids, X_orig_attention],
                        y=y_orig,
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        steps_per_epoch=num_steps,
                        validation_data=([X_aug_ids, X_aug_attention], y_aug),
                        callbacks=[early_stopping],
                        verbose=2)

    return history.history['val_accuracy'][-1]


def run_distilbert_tests():
    dl = DataLoader()
    sizes = [50, 100, 500, 1000, 5000]
    file_name = 'distilbert_scores.csv'

    da_methods = {'eda': dl.import_from_eda, 'unaltered': dl.import_unaltered_reddit}

    if os.path.exists(file_name):
        df = pd.read_csv(file_name, index_col=0)
    else:
        df = pd.DataFrame(columns=da_methods.keys())

    for size in sizes:
        if size not in df.index:
            df.loc[size] = np.nan

        for method_name in da_methods:
            da_method = da_methods[method_name]

            if method_name not in df.columns:
                df.insert(loc=0, column=method_name, value=np.nan)

            if np.isnan(df.loc[size][method_name]):
                X, y = da_method(size=size)
                epochs = EPOCHS_SMALL if size > 1000 else EPOCHS

                df.loc[size][method_name] = single_distilbert_test(X, y, epochs=epochs)

            print(df)
            df.to_csv(file_name)

def run_distilbert_tests_dir():
    dl = DataLoader()
    sizes = [50]
    file_name = 'distilbert_scores_many.csv'
    da_methods = {'eda': dl.import_from_eda_dir, 'unaltered': dl.import_unaltered_reddit_dir}

    dat = []
    for size in sizes:
        row = []
        for method_name in da_methods:
            da_method = da_methods[method_name]
            col = []
            for X, y in da_method(size=size):
                epochs = EPOCHS_SMALL if size > 1000 else EPOCHS
                col.append(single_distilbert_test(X, y,epochs=epochs))
            row.append(col)
        dat.append(row)

    df = pd.DataFrame(dat, columns=["eda_means", "unaltered_means"])
    df.index = sizes
    df.to_csv(file_name)
    return df


def run_distilbert_consistency_tests():
    dl = DataLoader()
    sizes = [50, 100, 500, 1000, 5000]
    file_name = 'distilbert_consistency_scores.csv'

    da_methods = {'unaltered': dl.import_unaltered_reddit, 'eda': dl.import_from_eda}

    if os.path.exists(file_name):
        df = pd.read_csv(file_name, index_col=0)
    else:
        df = pd.DataFrame(columns=da_methods.keys())

    for size in sizes:
        X_orig, y_orig = dl.import_unaltered_reddit(size=size)

        if size not in df.index:
            df.loc[size] = np.nan

        for method_name in da_methods:
            da_method = da_methods[method_name]

            if method_name not in df.columns:
                df.insert(loc=0, column=method_name, value=np.nan)

            if np.isnan(df.loc[size][method_name]):
                X_aug, y_aug = da_method(size=size)
                epochs = EPOCHS_SMALL if size > 1000 else EPOCHS

                df.loc[size][method_name] = single_distilbert_consistency_test(X_orig,
                                                                               y_orig,
                                                                               X_aug,
                                                                               y_aug,
                                                                               epochs=epochs)

            print(df)
            df.to_csv(file_name)

def run_distilbert_consistency_tests_dir():
    dl = DataLoader()
    sizes = [50]
    file_name = 'distilbert_consistency_scores_many.csv'
    da_methods = {'eda': dl.import_from_eda_dir, 'unaltered': dl.import_unaltered_reddit_dir}

    dat = []
    for size in sizes:
        origs = list(dl.import_unaltered_reddit_dir(size))
        row = []
        for method_name in da_methods:
            da_method = da_methods[method_name]
            augs = list(da_method(size=size))
            epochs = EPOCHS_SMALL if size > 1000 else EPOCHS
            col= [single_distilbert_consistency_test(
                    orig[0], orig[1], aug[0], aug[1],epochs=epochs) for orig, aug in zip(origs,augs)]
            row.append(col)
        dat.append(row)

    df = pd.DataFrame(dat, columns=["eda_consistencies", "unaltered_consistencies"])
    df.index = sizes
    df.to_csv(file_name)
    return df

if __name__ == '__main__':
    run_distilbert_tests_dir()
    run_distilbert_consistency_tests_dir()