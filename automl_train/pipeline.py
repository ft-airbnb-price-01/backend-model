from tensorflow.train import cosine_decay, AdamOptimizer
from tensorflow.contrib.opt import AdamWOptimizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, CuDNNLSTM, GRU, CuDNNGRU, concatenate, Dense, BatchNormalization, Dropout, AlphaDropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
import csv
import sys
import warnings
from datetime import datetime
from math import floor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)


def build_model(encoders):
    """Builds and compiles the model from scratch.

    # Arguments
        encoders: dict of encoders (used to set size of text/categorical inputs)

    # Returns
        model: A compiled model which can be used to train or predict.
    """

    # property_type
    input_property_type_size = len(encoders['property_type_encoder'].classes_)
    input_property_type = Input(shape=(
        input_property_type_size if input_property_type_size != 2 else 1,), name="input_property_type")

    # room_type
    input_room_type_size = len(encoders['room_type_encoder'].classes_)
    input_room_type = Input(shape=(
        input_room_type_size if input_room_type_size != 2 else 1,), name="input_room_type")

    # accommodates
    input_accommodates_size = len(encoders['accommodates_encoder'].classes_)
    input_accommodates = Input(shape=(
        input_accommodates_size if input_accommodates_size != 2 else 1,), name="input_accommodates")

    # bathrooms
    input_bathrooms_size = len(encoders['bathrooms_encoder'].classes_)
    input_bathrooms = Input(shape=(
        input_bathrooms_size if input_bathrooms_size != 2 else 1,), name="input_bathrooms")

    # bed_type
    input_bed_type_size = len(encoders['bed_type_encoder'].classes_)
    input_bed_type = Input(shape=(
        input_bed_type_size if input_bed_type_size != 2 else 1,), name="input_bed_type")

    # cancellation_policy
    input_cancellation_policy_size = len(
        encoders['cancellation_policy_encoder'].classes_)
    input_cancellation_policy = Input(shape=(
        input_cancellation_policy_size if input_cancellation_policy_size != 2 else 1,), name="input_cancellation_policy")

    # cleaning_fee
    input_cleaning_fee_size = len(encoders['cleaning_fee_encoder'].classes_)
    input_cleaning_fee = Input(shape=(
        input_cleaning_fee_size if input_cleaning_fee_size != 2 else 1,), name="input_cleaning_fee")

    # city
    input_city_size = len(encoders['city_encoder'].classes_)
    input_city = Input(
        shape=(input_city_size if input_city_size != 2 else 1,), name="input_city")

    # host_identity_verified
    input_host_identity_verified_size = len(
        encoders['host_identity_verified_encoder'].classes_)
    input_host_identity_verified = Input(shape=(
        input_host_identity_verified_size if input_host_identity_verified_size != 2 else 1,), name="input_host_identity_verified")

    # host_since
    input_host_since = Input(shape=(1,), name="input_host_since")

    # instant_bookable
    input_instant_bookable_size = len(
        encoders['instant_bookable_encoder'].classes_)
    input_instant_bookable = Input(shape=(
        input_instant_bookable_size if input_instant_bookable_size != 2 else 1,), name="input_instant_bookable")

    # review_scores_rating
    input_review_scores_rating = Input(
        shape=(1,), name="input_review_scores_rating")

    # zipcode
    input_zipcode_size = len(encoders['zipcode_encoder'].classes_)
    input_zipcode = Input(shape=(
        input_zipcode_size if input_zipcode_size != 2 else 1,), name="input_zipcode")

    # bedrooms
    input_bedrooms_size = len(encoders['bedrooms_encoder'].classes_)
    input_bedrooms = Input(shape=(
        input_bedrooms_size if input_bedrooms_size != 2 else 1,), name="input_bedrooms")

    # beds
    input_beds_size = len(encoders['beds_encoder'].classes_)
    input_beds = Input(
        shape=(input_beds_size if input_beds_size != 2 else 1,), name="input_beds")

    # Combine all the inputs into a single layer
    concat = concatenate([
        input_property_type,
        input_room_type,
        input_accommodates,
        input_bathrooms,
        input_bed_type,
        input_cancellation_policy,
        input_cleaning_fee,
        input_city,
        input_host_identity_verified,
        input_host_since,
        input_instant_bookable,
        input_review_scores_rating,
        input_zipcode,
        input_bedrooms,
        input_beds
    ], name="concat")

    # Multilayer Perceptron (MLP) to find interactions between all inputs
    hidden = Dense(128, activation='selu', name='hidden_1',
                   kernel_regularizer=None)(concat)
    hidden = AlphaDropout(0.2, name="dropout_1")(hidden)

    for i in range(4-1):
        hidden = Dense(128, activation="selu", name="hidden_{}".format(
            i+2), kernel_regularizer=None)(hidden)
        hidden = AlphaDropout(0.2, name="dropout_{}".format(i+2))(hidden)
    output = Dense(1, name="output", kernel_regularizer=l2(1e-3))(hidden)

    # Build and compile the model.
    model = Model(inputs=[
        input_property_type,
        input_room_type,
        input_accommodates,
        input_bathrooms,
        input_bed_type,
        input_cancellation_policy,
        input_cleaning_fee,
        input_city,
        input_host_identity_verified,
        input_host_since,
        input_instant_bookable,
        input_review_scores_rating,
        input_zipcode,
        input_bedrooms,
        input_beds
    ],
        outputs=[output])
    model.compile(loss="msle",
                  optimizer=AdamWOptimizer(learning_rate=0.0001,
                                           weight_decay=0.025))

    return model


def build_encoders(df):
    """Builds encoders for fields to be used when
    processing data for the model.

    All encoder specifications are stored in locally
    in /encoders as .json files.

    # Arguments
        df: A pandas DataFrame containing the data.
    """

    # property_type
    property_type_counts = df['property_type'].value_counts()
    property_type_perc = max(floor(0.5 * property_type_counts.size), 1)
    property_type_top = np.array(
        property_type_counts.index[0:property_type_perc], dtype=object)
    property_type_encoder = LabelBinarizer()
    property_type_encoder.fit(property_type_top)

    with open(os.path.join('encoders', 'property_type_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(property_type_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # room_type
    room_type_counts = df['room_type'].value_counts()
    room_type_perc = max(floor(0.5 * room_type_counts.size), 1)
    room_type_top = np.array(
        room_type_counts.index[0:room_type_perc], dtype=object)
    room_type_encoder = LabelBinarizer()
    room_type_encoder.fit(room_type_top)

    with open(os.path.join('encoders', 'room_type_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(room_type_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # accommodates
    accommodates_counts = df['accommodates'].value_counts()
    accommodates_perc = max(floor(0.5 * accommodates_counts.size), 1)
    accommodates_top = np.array(
        accommodates_counts.index[0:accommodates_perc], dtype=object)
    accommodates_encoder = LabelBinarizer()
    accommodates_encoder.fit(accommodates_top)

    with open(os.path.join('encoders', 'accommodates_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(accommodates_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # bathrooms
    bathrooms_counts = df['bathrooms'].value_counts()
    bathrooms_perc = max(floor(0.5 * bathrooms_counts.size), 1)
    bathrooms_top = np.array(
        bathrooms_counts.index[0:bathrooms_perc], dtype=object)
    bathrooms_encoder = LabelBinarizer()
    bathrooms_encoder.fit(bathrooms_top)

    with open(os.path.join('encoders', 'bathrooms_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bathrooms_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # bed_type
    bed_type_counts = df['bed_type'].value_counts()
    bed_type_perc = max(floor(0.5 * bed_type_counts.size), 1)
    bed_type_top = np.array(
        bed_type_counts.index[0:bed_type_perc], dtype=object)
    bed_type_encoder = LabelBinarizer()
    bed_type_encoder.fit(bed_type_top)

    with open(os.path.join('encoders', 'bed_type_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bed_type_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # cancellation_policy
    cancellation_policy_counts = df['cancellation_policy'].value_counts()
    cancellation_policy_perc = max(
        floor(0.5 * cancellation_policy_counts.size), 1)
    cancellation_policy_top = np.array(
        cancellation_policy_counts.index[0:cancellation_policy_perc], dtype=object)
    cancellation_policy_encoder = LabelBinarizer()
    cancellation_policy_encoder.fit(cancellation_policy_top)

    with open(os.path.join('encoders', 'cancellation_policy_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(cancellation_policy_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # cleaning_fee
    cleaning_fee_counts = df['cleaning_fee'].value_counts()
    cleaning_fee_perc = max(floor(0.5 * cleaning_fee_counts.size), 1)
    cleaning_fee_top = np.array(
        cleaning_fee_counts.index[0:cleaning_fee_perc], dtype=object)
    cleaning_fee_encoder = LabelBinarizer()
    cleaning_fee_encoder.fit(cleaning_fee_top)

    with open(os.path.join('encoders', 'cleaning_fee_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(cleaning_fee_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # city
    city_counts = df['city'].value_counts()
    city_perc = max(floor(0.5 * city_counts.size), 1)
    city_top = np.array(city_counts.index[0:city_perc], dtype=object)
    city_encoder = LabelBinarizer()
    city_encoder.fit(city_top)

    with open(os.path.join('encoders', 'city_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(city_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # host_identity_verified
    host_identity_verified_counts = df['host_identity_verified'].value_counts()
    host_identity_verified_perc = max(
        floor(0.5 * host_identity_verified_counts.size), 1)
    host_identity_verified_top = np.array(
        host_identity_verified_counts.index[0:host_identity_verified_perc], dtype=object)
    host_identity_verified_encoder = LabelBinarizer()
    host_identity_verified_encoder.fit(host_identity_verified_top)

    with open(os.path.join('encoders', 'host_identity_verified_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(host_identity_verified_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # host_since
    host_since_enc = df['host_since']
    host_since_encoder = StandardScaler()
    host_since_encoder_attrs = ['mean_', 'var_', 'scale_']
    host_since_encoder.fit(df['host_since'].values.reshape(-1, 1))

    host_since_encoder_dict = {attr: getattr(host_since_encoder, attr).tolist()
                               for attr in host_since_encoder_attrs}

    with open(os.path.join('encoders', 'host_since_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(host_since_encoder_dict, outfile, ensure_ascii=False)

    # instant_bookable
    instant_bookable_counts = df['instant_bookable'].value_counts()
    instant_bookable_perc = max(floor(0.5 * instant_bookable_counts.size), 1)
    instant_bookable_top = np.array(
        instant_bookable_counts.index[0:instant_bookable_perc], dtype=object)
    instant_bookable_encoder = LabelBinarizer()
    instant_bookable_encoder.fit(instant_bookable_top)

    with open(os.path.join('encoders', 'instant_bookable_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(instant_bookable_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # review_scores_rating
    review_scores_rating_enc = df['review_scores_rating']
    review_scores_rating_encoder = StandardScaler()
    review_scores_rating_encoder_attrs = ['mean_', 'var_', 'scale_']
    review_scores_rating_encoder.fit(
        df['review_scores_rating'].values.reshape(-1, 1))

    review_scores_rating_encoder_dict = {attr: getattr(review_scores_rating_encoder, attr).tolist()
                                         for attr in review_scores_rating_encoder_attrs}

    with open(os.path.join('encoders', 'review_scores_rating_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(review_scores_rating_encoder_dict,
                  outfile, ensure_ascii=False)

    # zipcode
    zipcode_counts = df['zipcode'].value_counts()
    zipcode_perc = max(floor(0.5 * zipcode_counts.size), 1)
    zipcode_top = np.array(zipcode_counts.index[0:zipcode_perc], dtype=object)
    zipcode_encoder = LabelBinarizer()
    zipcode_encoder.fit(zipcode_top)

    with open(os.path.join('encoders', 'zipcode_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(zipcode_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # bedrooms
    bedrooms_counts = df['bedrooms'].value_counts()
    bedrooms_perc = max(floor(0.5 * bedrooms_counts.size), 1)
    bedrooms_top = np.array(
        bedrooms_counts.index[0:bedrooms_perc], dtype=object)
    bedrooms_encoder = LabelBinarizer()
    bedrooms_encoder.fit(bedrooms_top)

    with open(os.path.join('encoders', 'bedrooms_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bedrooms_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # beds
    beds_counts = df['beds'].value_counts()
    beds_perc = max(floor(0.5 * beds_counts.size), 1)
    beds_top = np.array(beds_counts.index[0:beds_perc], dtype=object)
    beds_encoder = LabelBinarizer()
    beds_encoder.fit(beds_top)

    with open(os.path.join('encoders', 'beds_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(beds_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Target Field: price


def load_encoders():
    """Loads the encoders built during `build_encoders`.

    # Returns
        encoders: A dict of encoder objects/specs.
    """

    encoders = {}

    # property_type
    property_type_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'property_type_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        property_type_encoder.classes_ = json.load(infile)
    encoders['property_type_encoder'] = property_type_encoder

    # room_type
    room_type_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'room_type_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        room_type_encoder.classes_ = json.load(infile)
    encoders['room_type_encoder'] = room_type_encoder

    # accommodates
    accommodates_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'accommodates_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        accommodates_encoder.classes_ = json.load(infile)
    encoders['accommodates_encoder'] = accommodates_encoder

    # bathrooms
    bathrooms_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bathrooms_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bathrooms_encoder.classes_ = json.load(infile)
    encoders['bathrooms_encoder'] = bathrooms_encoder

    # bed_type
    bed_type_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bed_type_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bed_type_encoder.classes_ = json.load(infile)
    encoders['bed_type_encoder'] = bed_type_encoder

    # cancellation_policy
    cancellation_policy_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'cancellation_policy_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        cancellation_policy_encoder.classes_ = json.load(infile)
    encoders['cancellation_policy_encoder'] = cancellation_policy_encoder

    # cleaning_fee
    cleaning_fee_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'cleaning_fee_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        cleaning_fee_encoder.classes_ = json.load(infile)
    encoders['cleaning_fee_encoder'] = cleaning_fee_encoder

    # city
    city_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'city_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        city_encoder.classes_ = json.load(infile)
    encoders['city_encoder'] = city_encoder

    # host_identity_verified
    host_identity_verified_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'host_identity_verified_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        host_identity_verified_encoder.classes_ = json.load(infile)
    encoders['host_identity_verified_encoder'] = host_identity_verified_encoder

    # host_since
    host_since_encoder = StandardScaler()
    host_since_encoder_attrs = ['mean_', 'var_', 'scale_']

    with open(os.path.join('encoders', 'host_since_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        host_since_attrs = json.load(infile)

    for attr, value in host_since_attrs.items():
        setattr(host_since_encoder, attr, value)
    encoders['host_since_encoder'] = host_since_encoder

    # instant_bookable
    instant_bookable_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'instant_bookable_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        instant_bookable_encoder.classes_ = json.load(infile)
    encoders['instant_bookable_encoder'] = instant_bookable_encoder

    # review_scores_rating
    review_scores_rating_encoder = StandardScaler()
    review_scores_rating_encoder_attrs = ['mean_', 'var_', 'scale_']

    with open(os.path.join('encoders', 'review_scores_rating_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        review_scores_rating_attrs = json.load(infile)

    for attr, value in review_scores_rating_attrs.items():
        setattr(review_scores_rating_encoder, attr, value)
    encoders['review_scores_rating_encoder'] = review_scores_rating_encoder

    # zipcode
    zipcode_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'zipcode_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        zipcode_encoder.classes_ = json.load(infile)
    encoders['zipcode_encoder'] = zipcode_encoder

    # bedrooms
    bedrooms_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bedrooms_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bedrooms_encoder.classes_ = json.load(infile)
    encoders['bedrooms_encoder'] = bedrooms_encoder

    # beds
    beds_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'beds_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        beds_encoder.classes_ = json.load(infile)
    encoders['beds_encoder'] = beds_encoder

    # Target Field: price

    return encoders


def process_data(df, encoders, process_target=True):
    """Processes an input DataFrame into a format
    sutable for model prediction.

    This function loads the encoder specifications created in
    `build_encoders`.

    # Arguments
        df: a DataFrame containing the source data
        encoders: a dict of encoders to process the data.
        process_target: boolean to determine if the target should be encoded.

    # Returns
        A tuple: A list containing all the processed fields to be fed
        into the model, and the processed target field.
    """

    # property_type
    property_type_enc = df['property_type'].values
    property_type_enc = encoders['property_type_encoder'].transform(
        property_type_enc)

    # room_type
    room_type_enc = df['room_type'].values
    room_type_enc = encoders['room_type_encoder'].transform(room_type_enc)

    # accommodates
    accommodates_enc = df['accommodates'].values
    accommodates_enc = encoders['accommodates_encoder'].transform(
        accommodates_enc)

    # bathrooms
    bathrooms_enc = df['bathrooms'].values
    bathrooms_enc = encoders['bathrooms_encoder'].transform(bathrooms_enc)

    # bed_type
    bed_type_enc = df['bed_type'].values
    bed_type_enc = encoders['bed_type_encoder'].transform(bed_type_enc)

    # cancellation_policy
    cancellation_policy_enc = df['cancellation_policy'].values
    cancellation_policy_enc = encoders['cancellation_policy_encoder'].transform(
        cancellation_policy_enc)

    # cleaning_fee
    cleaning_fee_enc = df['cleaning_fee'].values
    cleaning_fee_enc = encoders['cleaning_fee_encoder'].transform(
        cleaning_fee_enc)

    # city
    city_enc = df['city'].values
    city_enc = encoders['city_encoder'].transform(city_enc)

    # host_identity_verified
    host_identity_verified_enc = df['host_identity_verified'].values
    host_identity_verified_enc = encoders['host_identity_verified_encoder'].transform(
        host_identity_verified_enc)

    # host_since
    host_since_enc = df['host_since'].values.reshape(-1, 1)
    host_since_enc = encoders['host_since_encoder'].transform(host_since_enc)

    # instant_bookable
    instant_bookable_enc = df['instant_bookable'].values
    instant_bookable_enc = encoders['instant_bookable_encoder'].transform(
        instant_bookable_enc)

    # review_scores_rating
    review_scores_rating_enc = df['review_scores_rating'].values.reshape(-1, 1)
    review_scores_rating_enc = encoders['review_scores_rating_encoder'].transform(
        review_scores_rating_enc)

    # zipcode
    zipcode_enc = df['zipcode'].values
    zipcode_enc = encoders['zipcode_encoder'].transform(zipcode_enc)

    # bedrooms
    bedrooms_enc = df['bedrooms'].values
    bedrooms_enc = encoders['bedrooms_encoder'].transform(bedrooms_enc)

    # beds
    beds_enc = df['beds'].values
    beds_enc = encoders['beds_encoder'].transform(beds_enc)

    data_enc = [property_type_enc,
                room_type_enc,
                accommodates_enc,
                bathrooms_enc,
                bed_type_enc,
                cancellation_policy_enc,
                cleaning_fee_enc,
                city_enc,
                host_identity_verified_enc,
                host_since_enc,
                instant_bookable_enc,
                review_scores_rating_enc,
                zipcode_enc,
                bedrooms_enc,
                beds_enc
                ]

    if process_target:
        # Target Field: price
        price_enc = df['price'].values

        return (data_enc, price_enc)

    return data_enc


def model_predict(df, model, encoders):
    """Generates predictions for a trained model.

    # Arguments
        df: A pandas DataFrame containing the source data.
        model: A compiled model.
        encoders: a dict of encoders to process the data.

    # Returns
        A numpy array of predictions.
    """

    data_enc = process_data(df, encoders, process_target=False)

    headers = ['price']
    predictions = pd.DataFrame(model.predict(data_enc), columns=headers)

    return predictions


def model_train(df, encoders, args, model=None):
    """Trains a model, and saves the data locally.

    # Arguments
        df: A pandas DataFrame containing the source data.
        encoders: a dict of encoders to process the data.
        args: a dict of arguments passed through the command line
        model: A compiled model (for TensorFlow, None otherwise).
    """
    X, y = process_data(df, encoders)

    split = ShuffleSplit(n_splits=1, train_size=args.split,
                         test_size=None, random_state=123)

    for train_indices, val_indices in split.split(np.zeros(y.shape[0]), y):
        X_train = [field[train_indices, ] for field in X]
        X_val = [field[val_indices, ] for field in X]
        y_train = y[train_indices, ]
        y_val = y[val_indices, ]

    meta = meta_callback(args, X_val, y_val)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs,
              callbacks=[meta],
              batch_size=256)


class meta_callback(Callback):
    """Keras Callback used during model training to save current weights
    and metrics after each training epoch.

    Metrics metadata is saved in the /metadata folder.
    """

    def __init__(self, args, X_val, y_val):
        self.f = open(os.path.join('metadata', 'results.csv'), 'w')
        self.w = csv.writer(self.f)
        self.w.writerow(['epoch', 'time_completed'] + ['mse', 'mae', 'r_2'])
        self.in_automl = args.context == 'automl-gs'
        self.X_val = X_val
        self.y_val = y_val

    def on_train_end(self, logs={}):
        self.f.close()
        self.model.save_weights('model_weights.hdf5')

    def on_epoch_end(self, epoch, logs={}):
        y_true = self.y_val
        y_pred = self.model.predict(self.X_val)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r_2 = r2_score(y_true, y_pred)

        metrics = [mse, mae, r_2]
        time_completed = "{:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())
        self.w.writerow([epoch+1, time_completed] + metrics)

        # Only run while using automl-gs, which tells it an epoch is finished
        # and data is recorded.
        if self.in_automl:
            sys.stdout.flush()
            print("\nEPOCH_END")
