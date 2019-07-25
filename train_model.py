"""
Train VGG19
import os
os.system("pip install -U efficientnet")
"""
import argparse
import configparser
import datetime
import os

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from efficientnet import EfficientNetB5, preprocess_input
from keras.applications.densenet import DenseNet201
from keras.applications.densenet import \
    preprocess_input as densenet_preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils.data_utils import get_file
from PIL import Image, ImageFile
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


###################################################################################################
# Arguments for setting parameters while running array batch job
###################################################################################################

OPTIMISER_MODE = "Adam"

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-c", "--config_file", type=str, required=True,
                    help="Configuration file path")
PARSER.add_argument("-e", "--epochs", type=int, default=None,
                    help="Epochs")

ARGS = vars(PARSER.parse_args())

CONFIG_FILE = ARGS["config_file"]
CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_FILE)

# Set verbosity
NAME = CONFIG["general"]["name"]
VERBOSITY = int(CONFIG["general"]["verbosity"])

# Set model configuration
PRETRAINED_WEIGHTS = CONFIG["model"].get(
    "pretrained_weights_path", fallback=None)
LOSS = CONFIG["model"]["loss"]
METRICS = CONFIG["model"]["metrics"].split(",")

# Dataset folder information
TRAINING_CSV = "../input/training-labels.csv"
DATASET_FOLDER = "../input/train/output_combined2"
TESTSET_FOLDER = "../input/test/Test"

CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

OUTPUT_FOLDER = "../output"
CURRENT_OUTPUT = os.path.join(OUTPUT_FOLDER, f"{NAME}_{CURRENT_TIMESTAMP}")
LOGS_FOLDER = os.path.join(CURRENT_OUTPUT, "./logs")
WEIGHTS_FOLDER = os.path.join(CURRENT_OUTPUT, "./weights")
REPORTS_FOLDER = os.path.join(CURRENT_OUTPUT, "./reports")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CURRENT_OUTPUT, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

CLASSIFICATION_REPORT_FILE = os.path.join(
    REPORTS_FOLDER, "classification_report.txt")
CONFUSION_MATRIX_FILE = os.path.join(REPORTS_FOLDER, "confusion_matrix.txt")
COHEN_KAPPA_SCORE_FILE = os.path.join(REPORTS_FOLDER, "cohen_kappa_metric.txt")


# Image augmentation parameters
IMAGE_AUGMENTATION = CONFIG["image_augmentation"]
HEIGHT = int(IMAGE_AUGMENTATION["height"])
WIDTH = int(IMAGE_AUGMENTATION["width"])
DEPTH = int(IMAGE_AUGMENTATION["depth"])
SHIFT = float(IMAGE_AUGMENTATION["shift"])
ROTATION = float(IMAGE_AUGMENTATION["rotation"])
VAL_AUG_FACTOR = float(
    IMAGE_AUGMENTATION["validation_data_augmentation_factor"])

# Hyperparameters
HYPERPARAMETERS = CONFIG["hyperparameters"]

# Set epochs from args if set else from config file
EPOCHS = ARGS["epochs"] if ARGS["epochs"] else int(HYPERPARAMETERS["epochs"])

BATCH_SIZE = int(HYPERPARAMETERS["batch_size"])
LEARNING_RATE = float(HYPERPARAMETERS["learning_rate"])
DROP_EVERY = float(HYPERPARAMETERS["learning_rate_decay_after_x_epoch"])
DROP_FACTOR = float(HYPERPARAMETERS["decay_rate"])
MOMENTUM = float(HYPERPARAMETERS["momentum"])

# Image generator information
TRAIN_TEST_VAL_SPLIT = CONFIG["train_test_val_split"]

DEBUG_MODE = "DEBUG" in CONFIG

TEST_SPLIT = float(TRAIN_TEST_VAL_SPLIT["test_split"])
VALIDATION_SPLIT = float(TRAIN_TEST_VAL_SPLIT["validation_split"])


##################################################################################################
# Switch models
###################################################################################################
BASE_MODEL = None
preprocessing_function = None

if NAME == "VGG19":
    HEIGHT, WIDTH = 224, 224
    BASE_MODEL = VGG19(include_top=False, weights="imagenet",
                       input_shape=(HEIGHT, WIDTH, DEPTH))
    preprocessing_function = vgg19_preprocess_input
elif NAME == "VGG16":
    HEIGHT, WIDTH = 224, 224
    BASE_MODEL = VGG16(include_top=False, weights="imagenet",
                       input_shape=(HEIGHT, WIDTH, DEPTH))
    preprocessing_function = vgg16_preprocess_input
elif NAME == "Densenet":
    HEIGHT, WIDTH = 128, 128
    BASE_MODEL = DenseNet201(include_top=False, weights="imagenet",
                             input_shape=(HEIGHT, WIDTH, DEPTH))
    preprocessing_function = densenet_preprocess_input
elif NAME == "efficientnet":
    HEIGHT, WIDTH = 128, 128
    BASE_MODEL = EfficientNetB5(include_top=False, weights="imagenet",
                                input_shape=(HEIGHT, WIDTH, DEPTH))
    preprocessing_function = preprocess_input
else:
    HEIGHT, WIDTH = 224, 224
    BASE_MODEL = VGG19(include_top=False, weights="imagenet",
                       input_shape=(HEIGHT, WIDTH, DEPTH))
    preprocessing_function = vgg19_preprocess_input


##################################################################################################
# Read details from CSV
###################################################################################################

DATASET = pd.read_csv(TRAINING_CSV, dtype=str)

# DEBUG_MODE SET
DATASET = DATASET[:BATCH_SIZE*12] if DEBUG_MODE else DATASET

TRAIN_VALIDATION, TEST = train_test_split(DATASET, test_size=TEST_SPLIT)
TRAIN, VALIDATION = train_test_split(
    TRAIN_VALIDATION, test_size=VALIDATION_SPLIT)


# KAGGLE TESTSET

TESTSET_ARRAY = [[filename, "0"]
                 for filename in os.listdir(TESTSET_FOLDER)]
TESTSET = pd.DataFrame(TESTSET_ARRAY, columns=["Id", "Expected"])

TESTSET = TESTSET[:int(BATCH_SIZE*1.5)] if DEBUG_MODE else TESTSET

###################################################################################################
#  Create data generator to augment images for training and validation
###################################################################################################

TRAINING_DATA_GENERATOR = ImageDataGenerator(rotation_range=ROTATION,
                                             width_shift_range=SHIFT,
                                             height_shift_range=SHIFT,
                                             preprocessing_function=preprocessing_function)

VALIDATION_DATA_GENERATOR = ImageDataGenerator(rotation_range=ROTATION *
                                               (1+VAL_AUG_FACTOR),
                                               width_shift_range=SHIFT *
                                               (1+VAL_AUG_FACTOR),
                                               height_shift_range=SHIFT *
                                               (1+VAL_AUG_FACTOR),
                                               preprocessing_function=preprocessing_function)

TEST_DATA_GENERATOR = ImageDataGenerator(
    preprocessing_function=preprocessing_function)

TESTSET_DATA_GENERATOR = ImageDataGenerator(
    preprocessing_function=preprocessing_function)

COLOR_MODE = "grayscale" if DEPTH == 1 else "rgb"

print("[INFO] Creating training data generator")
TRAINING_DATA = TRAINING_DATA_GENERATOR.flow_from_dataframe(dataframe=TRAIN,
                                                            directory=DATASET_FOLDER,
                                                            x_col="Filename",
                                                            y_col="Drscore",
                                                            class_mode="categorical",
                                                            color_mode=COLOR_MODE,
                                                            target_size=(
                                                                WIDTH, HEIGHT),
                                                            batch_size=BATCH_SIZE)

print("[INFO] Creating validation data generator")
VALIDATION_DATA = VALIDATION_DATA_GENERATOR.flow_from_dataframe(dataframe=VALIDATION,
                                                                directory=DATASET_FOLDER,
                                                                x_col="Filename",
                                                                y_col="Drscore",
                                                                class_mode="categorical",
                                                                color_mode=COLOR_MODE,
                                                                target_size=(
                                                                    WIDTH, HEIGHT),
                                                                batch_size=BATCH_SIZE)

print("[INFO] Creating test data generator")
TEST_DATA = TEST_DATA_GENERATOR.flow_from_dataframe(dataframe=TEST,
                                                    directory=DATASET_FOLDER,
                                                    x_col="Filename",
                                                    y_col="Drscore",
                                                    class_mode="categorical",
                                                    target_size=(
                                                        WIDTH, HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)

print("[INFO] Creating testset data generator")
TESTSET_DATA = TESTSET_DATA_GENERATOR.flow_from_dataframe(dataframe=TESTSET,
                                                          directory=TESTSET_FOLDER,
                                                          x_col="Id",
                                                          y_col="Expected",
                                                          class_mode="categorical",
                                                          target_size=(
                                                              WIDTH, HEIGHT),
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=False)

NUM_OF_TRAINING_SAMPLES = 64 if DEBUG_MODE else len(TRAIN)
NUM_OF_VALIDATION_SAMPLES = len(VALIDATION)
NUM_OF_TEST_SAMPLES = len(TEST)//BATCH_SIZE+1
NUM_OF_TESTSET_SAMPLES = len(TESTSET)//BATCH_SIZE+1
CLASSES = 5


###################################################################################################
# Cohen Kappa metrics
###################################################################################################


def cohen_kappa(y_true, y_pred):
    y_true_classes = tf.argmax(y_true, 1)
    y_pred_classes = tf.argmax(y_pred, 1)
    return tf.contrib.metrics.cohen_kappa(y_true_classes, y_pred_classes, CLASSES)[1]


###################################################################################################
# Compile MetaMorph model
###################################################################################################


MODEL = Sequential()
MODEL.add(BASE_MODEL)
MODEL.add(GlobalAveragePooling2D(name="v_GAP2D"))
MODEL.add(Dense(1024, activation='relu', name="v_1024Dense"))
MODEL.add(Dropout(0.5, name="v_0.5Dropout"))
MODEL.add(Dense(512, activation='relu',  name="v_512Dense"))
MODEL.add(Dense(CLASSES, activation='softmax', name="v_Softmax"))

OPTIMISER = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)

if OPTIMISER_MODE == "Adam":
    OPTIMISER = Adam(lr=LEARNING_RATE, beta_1=0.9,
                     beta_2=0.999, epsilon=0.1, decay=0.0)

MODEL.compile(loss=LOSS, optimizer=OPTIMISER, metrics=[*METRICS, cohen_kappa])

K.get_session().run(tf.local_variables_initializer())

if PRETRAINED_WEIGHTS:
    MODEL.load_weights(PRETRAINED_WEIGHTS)

###################################################################################################
# Define callbacks
###################################################################################################


# Checkpoint model callback
WEIGHT_NAME = os.path.join(
    WEIGHTS_FOLDER, "./weights.{epoch:02d}-{val_cohen_kappa:.2f}.hdf5")
CHECKPOINT = ModelCheckpoint(WEIGHT_NAME, monitor="val_cohen_kappa", mode="max",
                             save_best_only=True, save_weights_only=True, verbose=1)

EARLY_STOP = EarlyStopping(monitor='val_cohen_kappa',
                           min_delta=0.001,
                           patience=8,
                           mode='max',
                           verbose=1)

TENSORBOARD = TensorBoard(log_dir=LOGS_FOLDER,
                          histogram_freq=0,
                          # write_batch_performance=True,
                          write_graph=True,
                          write_images=True)

CALLBACKS = [EARLY_STOP, CHECKPOINT, TENSORBOARD]

###################################################################################################
# Train the model
###################################################################################################

print("[INFO] Training the model....")
HISTORY = MODEL.fit_generator(generator=TRAINING_DATA,
                              steps_per_epoch=NUM_OF_TRAINING_SAMPLES//BATCH_SIZE,
                              epochs=EPOCHS,
                              callbacks=CALLBACKS,
                              validation_data=VALIDATION_DATA,
                              validation_steps=NUM_OF_VALIDATION_SAMPLES//BATCH_SIZE,
                              verbose=VERBOSITY)


###################################################################################################
# Storing the model to output
###################################################################################################

print("[INFO] Storing trained model....")
MODEL.save("./trained_model.hdf5")
MODEL.save_weights("./trained_weights.hdf5")

###################################################################################################
# Evaluate the model and store the report and history log
###################################################################################################

print("[INFO] Evaluating the model....")
TEST_PREDICTIONS = MODEL.predict_generator(generator=TEST_DATA,
                                           steps=NUM_OF_TEST_SAMPLES,
                                           verbose=VERBOSITY)
TEST_Y_PREDICTIONS = np.argmax(TEST_PREDICTIONS, axis=1)

LABELS = [0, 1, 2, 3, 4]

CLASSIFICATION_REPORT = classification_report(
    y_true=TEST_DATA.classes, y_pred=TEST_Y_PREDICTIONS, labels=LABELS)
with open(CLASSIFICATION_REPORT_FILE, "w") as report:
    print(CLASSIFICATION_REPORT, file=report)


CONFUSION_MATRIX = confusion_matrix(
    y_true=TEST_DATA.classes, y_pred=TEST_Y_PREDICTIONS, labels=LABELS)
with open(CONFUSION_MATRIX_FILE, "w") as report:
    print(CONFUSION_MATRIX, file=report)


COHEN_KAPPA_SCORE = cohen_kappa_score(
    y1=TEST_DATA.classes, y2=TEST_Y_PREDICTIONS, labels=LABELS)
with open(COHEN_KAPPA_SCORE_FILE, "w") as report:
    print(COHEN_KAPPA_SCORE, file=report)

PREDICTIONS = MODEL.predict_generator(generator=TESTSET_DATA,
                                      steps=NUM_OF_TESTSET_SAMPLES,
                                      verbose=VERBOSITY)
Y_PREDICTIONS = np.argmax(PREDICTIONS, axis=1)

TESTSET["Expected"] = Y_PREDICTIONS
TESTSET.to_csv("Submission.csv", index=False)

ACCURACY = HISTORY.history["acc"][-1] * 100
VALIDATION_ACCURACY = HISTORY.history["val_acc"][-1] * 100
LOSS = HISTORY.history["loss"][-1]
COHEN_KAPPA = HISTORY.history["cohen_kappa"][-1]
VALIDATION_LOSS = HISTORY.history["val_loss"][-1]
VALIDATION_COHEN_KAPPA = HISTORY.history["val_cohen_kappa"][-1]

REPORT = [
    80*"#",
    "\n",
    "REPORT".center(80),
    f'Training with {NAME} config'.center(80),
    "\n",
    80*"#",
    "\n",
    "DETAILS",
    "\n",
    f'Accuracy: {ACCURACY:.2f}',
    f'Validation accuracy: {VALIDATION_ACCURACY:.2f}',
    f'Loss: {LOSS:.4f}',
    f'Validation Loss: {VALIDATION_LOSS:.4f}',
    f'COHEN_KAPPA: {COHEN_KAPPA:.4f}',
    f'VALIDATION_COHEN_KAPPA: {VALIDATION_COHEN_KAPPA:.4f}',
    "\n",
    80*"#",
]

for line in REPORT:
    print(line)

FILENAME = f'REPORT.txt'

print("[INFO] Storing the evaluation results....")
with open(FILENAME, "w") as eval_result:
    eval_result.write("\n".join(REPORT))
