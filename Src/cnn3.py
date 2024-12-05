import tensorflow as tf
import tf_keras
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tf_keras.preprocessing import image
from tf_keras.preprocessing.image import ImageDataGenerator, img_to_array
from tf_keras.applications import MobileNet, MobileNetV2, imagenet_utils
from tf_keras.utils import to_categorical

from tf_keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, MaxPooling2D
from tf_keras.models import Sequential, Model
from tf_keras.utils import to_categorical
from tf_keras.losses import binary_crossentropy
from tf_keras.optimizers import Adam
from tf_keras.optimizers.schedules import ExponentialDecay
from tf_keras.regularizers import l2
from tf_keras.callbacks import EarlyStopping

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import class_weight
from sklearn.preprocessing import MultiLabelBinarizer

def prepare_images(img_path, dataframe):
    # preprocess and flatten images first
    data_arr = []
    target_arr = []
    img_names_arr = []

    print("Loading images ...")

    for index, row in dataframe.iterrows():
        img_name = row['Image ID']
        categories = row.drop('Image ID').values.astype(int)  # getting all labels

        # load and process the img
        img = image.load_img(img_path + img_name, target_size=(224, 224))
        img_array = image.img_to_array(img)
        data_arr.append(img_array)
        target_arr.append(categories)
        img_names_arr.append(img_name)

    data = np.array(data_arr)
    data = np.array(data, dtype="float") / 255.0
    target = np.array(target_arr)

    print("Loaded images")

    return data, target, img_names_arr


def get_img_gen(df):
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.05,
        samplewise_center=True,
        samplewise_std_normalization=True,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='reflect')

    return datagen


def develop_model(input_shape, class_number):
    base_model = MobileNet(include_top=False, input_tensor=Input(shape=input_shape))

    '''
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    # tried softmax
    model.add(Dense(class_number, activation='sigmoid'))
    model.summary()
    '''
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(class_number, activation="sigmoid"))
    model.summary()

    return model


def all_labels(path, label_list):
    # need to account for multi-label nature
    # read in .csv file and produce dataframe with each disease associated to the img
    # creating a dataframe for multi-label classification
    disease_names = ['Atelectasis',
                     'Cardiomegaly',
                     'Effusion',
                     'Infiltration',
                     'Mass',
                     'Nodule',
                     'Pneumonia',
                     'Pneumothorax',
                     'Consolidation',
                     'Edema',
                     'Emphysema',
                     'Fibrosis',
                     'Pleural Thickening',
                     'Hernia',
                     'Other']

    def gather_diseases(row):
        diseases = [disease for disease in disease_names if row[disease] == 'YES']
        return ','.join(diseases) if diseases else 'Normal'

    label_list['Diseases'] = label_list.apply(gather_diseases, axis=1)

    df = label_list[['Image ID', 'Diseases']]
    df['Diseases'] = df['Diseases'].apply(lambda x: x.split(','))

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['Diseases'])

    labels_df = pd.DataFrame(labels, columns=mlb.classes_)
    labels_df.insert(0, 'Image ID', df['Image ID'])

    disease_labels = mlb.classes_

    x, y, img_names = prepare_images(path + 'labeled/', labels_df)

    # SPLIT THE DATA
    print("Splitting data into training, testing, and validation sets ...")
    x_train, x_temp, y_train, y_temp, img_train, img_temp = train_test_split(x, y, img_names, random_state=42,
                                                                             test_size=0.40)
    x_val, x_test, y_val, y_test, img_val, img_test = train_test_split(x_temp, y_temp, img_temp, random_state=42,
                                                                       test_size=0.50)
    print("Split data into training, testing, and validation sets")

    class_weights = get_weights(y_train)

    mlb.fit_transform(y_train)
    mlb.fit_transform(y_test)
    mlb.fit_transform(y_val)

    # COMPILE THE MODEL
    print("Compiling the model ...")
    model = develop_model((224, 224, 3), 16)

    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    print("Compiled the model")

    datagen = get_img_gen(df)

    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # TRAIN THE MODEL
    print("Training the model ...")
    h = model.fit(datagen.flow(x_train, y_train,
                                          batch_size=32,
                                          seed=27,
                                          shuffle=False),
                  steps_per_epoch=len(x_train) // 32,
                  validation_data=(x_val, y_val),
                  validation_steps=len(x_val) // 32,
                  epochs=60)
    print("Trained the model")

    # EVALUATE THE MODEL
    # have model predict preprocessed images (test set)
    print("Evaluating the model ...")
    predictions = model.predict(x_test, batch_size=32)
    print("Evaluated the model")

    # apply threshold to convert probabilities to binary predictions (0 or 1)
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)  # Binary classification (0 or 1)

    #print(predicted_labels)

    # y_test should already be binary (0 or 1) for each label
    correct_labels = y_test

    # calculate binary accuracy (you can also use `accuracy_score` for multi-label)
    accuracy = accuracy_score(correct_labels.flatten(), predicted_labels.flatten())

    print(f"The model is {accuracy * 100:.2f}% accurate\n")

    accuracy = accuracy_score(correct_labels.flatten(), predicted_labels.flatten())

    print(f"\nThe model is {accuracy * 100:.2f}% accurate\n")

    # plot training loss & accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 60), h.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 60), h.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 60), h.history["binary_accuracy"], label="accuracy")
    plt.plot(np.arange(0, 60), h.history["val_binary_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()

    '''
    print("Sample correct labels:", correct_labels[:5])
    print("\nSample predictions:", predictions[:5])
    print("\nSample threshold adjusted predictions:", predicted_labels[:5])
    '''

    # prepare results for csv file
    results = []
    for img_name, correct_label, predicted_label in zip(img_test, correct_labels, predicted_labels):
        # correct_label and predicted_label should be vectors
        if isinstance(correct_label, np.ndarray) or isinstance(correct_label, list):
            actual = [disease_labels[i] for i in range(len(correct_label)) if correct_label[i] == 1]
        else:
            actual = [disease_labels[correct_label]]  # single-label case

        if isinstance(predicted_label, np.ndarray) or isinstance(predicted_label, list):
            predicted = [disease_labels[i] for i in range(len(predicted_label)) if predicted_label[i] == 1]
        else:
            predicted = [disease_labels[predicted_label]]  # single-label case

        # append to results
        results.append({
            "Image ID": img_name,
            "True Label": ', '.join(actual),  # joining the list of labels into a string
            "Predicted Label": ', '.join(predicted)  # joining the list of labels into a string
        })

    # create dataframe from results
    df_results = pd.DataFrame(results)

    # output dataframe to .csv file
    df_results.to_csv(path + "multiple_labels.csv", index=False)

def get_weights(y_train):
    n_classes = y_train.shape[1]
    class_weights = {}

    for i in range(n_classes):
        class_weights[i] = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=y_train[:, i]
        )

    return class_weights


if __name__ == "__main__":
    # load csv file
    path = "/Users/wendy/Desktop/school/uml/2024-2025 — junior year/machine learning/lung_diseases/"
    label_list = pd.read_csv(path + 'labels.csv')

    all_labels(path, label_list)
