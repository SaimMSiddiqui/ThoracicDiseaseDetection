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
from tf_keras.callbacks import EarlyStopping

from tf_keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, MaxPooling2D
from tf_keras.models import Sequential, Model
from tf_keras.utils import to_categorical

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
        category = row['Class']

        # load and process the img
        img = image.load_img(img_path + img_name, target_size=(224, 224))
        img_array = image.img_to_array(img)
        # img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        data_arr.append(img_array)
        target_arr.append(category) # TODO: change to name instead of index
        img_names_arr.append(img_name)

    data = np.array(data_arr)
    data = np.array(data, dtype="float") / 255.0
    target = np.array(target_arr)

    print("Loaded images")

    return data, target, img_names_arr


def get_img_gen(home_path, df):
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

    train_generator_df = datagen.flow_from_dataframe(dataframe=df,
                                                     directory=home_path+'/labeled/',
                                                     x_col="Image ID",
                                                     y_col="Class",
                                                     class_mode="binary",
                                                     target_size=(200,200),
                                                     batch_size=1,
                                                     rescale=1.0/255,
                                                     seed=2020)

    # plot the images
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,15))

    for i in range(4):
        # convert to unsigned ints for plotting
        img = next(train_generator_df)[0].astype('uint8')

        # change size from (1, 200, 200, 3) to (200, 200, 3) to plot the img
        img = np.squeeze(img)

        # plot raw pixel data
        ax[i].imshow(img)
        ax[i].axis('off')

    plt.show()

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
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    model.summary()

    return model


def abnormal_or_not(path, label_list):
    # creating a dataframe for abnormal and normal classification only
    category_names = ['Normal', 'Abnormal']
    df = label_list[['Image ID', 'Abnormal']]
    df.rename(columns={'Abnormal': 'Class'}, inplace=True)
    df['Class'] = df['Class'].replace({'YES': '1', 'NO': '0'})

    df = pd.DataFrame(df)

    datagen = get_img_gen(path, df)

    x, y, img_names = prepare_images(path + 'labeled/', df)

    print("Splitting data into training, testing, and validation sets ...")

    x_train, x_temp, y_train, y_temp, img_train, img_temp = train_test_split(x, y, img_names, random_state=42,
                                                                             test_size=0.40)
    x_val, x_test, y_val, y_test, img_val, img_test = train_test_split(x_temp, y_temp, img_temp, random_state=42,
                                                                       test_size=0.50)

    print("Split data into training, testing, and validation sets")

    # convert labels to "one-hot encoding"
    y_train_flat = np.array(y_train, dtype=int)
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # COMPILE THE MODEL
    print("Compiling the model ...")
    model = develop_model((224, 224, 3), 2)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
    print("Compiled the model")

    classes = np.array([0,1])
    # addressing class imbalance (more abnormal lungs than normal lungs)
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train_flat)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    # TRAIN THE MODEL
    print("Training the model ...")
    h = model.fit(datagen.flow(x_train, y_train,
                                          batch_size=32,
                                          seed=27,
                                          shuffle=False),
                  steps_per_epoch=len(x_train) // 32,
                  validation_data=(x_val, y_val),
                  validation_steps=len(x_val) // 32,
                  epochs=60,
                  class_weight=class_weight_dict,
                  verbose=1)
    print("Trained the model")


    # EVALUATE THE MODEL
    # have model predict preprocessed images (test set)
    print("Evaluating the model ...")
    predictions = model.predict(x_test, batch_size=32)
    print("Evaluated the model")

    # for each img in test set, need to find index of the label with corresponding largest predicted prob
    predictions = np.argmax(predictions, axis=1)

    # grab actual label for comparison purposes
    correct_labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(predictions, correct_labels)
    print(f"The model is {accuracy*100}% accurate\n")

    # prepare results for csv file
    results = []
    for img_name, correct_label, predicted_label in zip(img_test, correct_labels, predictions):
        # grab img name, correct, predicted
        actual = list(category_names)[correct_label]
        predicted = list(category_names)[predicted_label]
        # append to results
        results.append({"Image ID": img_name, "True Label": actual, "Predicted Label": predicted})

    # create dataframe from results
    df_results = pd.DataFrame(results)

    # output dataframe to .csv file
    df_results.to_csv(path + "1.csv", index=False)

    # display classification report
    print("Classification Report:")
    print(classification_report(y_test.argmax(axis=1), predictions, target_names=category_names))

    # plot training loss & accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 60), h.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 60), h.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 60), h.history["accuracy"], label="accuracy")
    plt.plot(np.arange(0, 60), h.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()

def all_labels(path, label_list):
    # need to account for multi-label nature
    # read in .csv file and produce dataframe with each disease associated to the img
    # creating a dataframe for multi-label classification

    copy = label_list
    disease_names = ['Atelectasis',
                     'Cardiomegaly',
                     'Effusion',
                     'Mass',
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

    df = label_list['Image ID', 'Diseases']
    print(df)
    # convert




if __name__ == "__main__":
    # load csv file
    path = "/Users/wendy/Desktop/school/uml/2024-2025 — junior year/machine learning/lung_diseases/"
    label_list = pd.read_csv(path + 'labels.csv')

    abnormal_or_not(path, label_list)
    #all_labels(path, label_list)
