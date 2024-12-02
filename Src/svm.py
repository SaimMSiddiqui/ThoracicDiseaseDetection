import pandas as pd
import os
import seaborn as sns
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def plot_data(abnormal_lungs, normal_lungs, diseases_dict):
    # graph to represent abnormal and normal lungs
    labels = ['Normal', 'Abnormal']
    counts = [len(normal_lungs), len(abnormal_lungs)]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['green', 'red'])
    plt.xlabel('Category')
    plt.ylabel('Number of X-rays')
    plt.title('Number of Normal and Abnormal Lung X-rays')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # graph to represent all disease data
    disease_labels = diseases_dict.keys()
    disease_count = [len(disease) for disease in diseases_dict.values()]
    print(disease_count)
    plt.figure(figsize=(12, 6))
    plt.barh(disease_labels, disease_count)
    plt.xlabel('Number of X-rays')
    plt.ylabel('Diseases')
    plt.title('Dataset Disease Distribution')
    max_count = max(disease_count)
    plt.xticks(range(0, max_count+10, 10))
    plt.tight_layout()
    plt.show()


def load_and_split_data(path, dictionary, categories):
    # preprocess and flatten images first
    flat_data_arr= [] # input array
    target_arr = [] # output array

    path = os.path.join(path, 'labeled/')
    for category, image_names in dictionary.items():
        print(f"Loading category: {category} ...")
        for img_name in image_names:
            img_array = imread(os.path.join(path, img_name))
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(categories.index(category))
        print(f"Loaded category: {category} successfully")

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)

    # create dataframe and split the data
    df = pd.DataFrame(flat_data)
    df['Target'] = target

    x = df.iloc[:, :-1] # input data
    y = df.iloc[:, -1] # output data

    # training, testing, validation sets: (60%, 20%, 20%)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.40, random_state=77, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=88, stratify=y_temp)

    '''
    # normalizing the features: important for almost all ML algorithms -- this didn't make it more accurate
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    '''

    print("Split data into training, testing, and validation sets")

    return x_train, x_val, x_test, y_train, y_val, y_test

def training_and_evaluation(x_train, x_val, x_test, y_train, y_val, y_test):
    path = "/Users/wendy/Desktop/school/uml/2024-2025 — junior year/machine learning/lung_diseases/"
    # parameters for GridSearchCV
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
    # support vector classifier
    svc = svm.SVC(probability=True)
    # creating model with GridSearchCV
    model = GridSearchCV(svc, param_grid)
    # training the model with the training data
    print("Training the model ...")
    model.fit(x_train, y_train)
    print("Training completed")

    # get accuracy
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)

    print(f"The model is {accuracy*100}% accurate\n")

    # get results, classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(training_labels.keys())))

    # create heatmap/confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # mask = np.eye(cm.shape[0], dtype=bool)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Reds', xticklabels=list(training_labels.keys()), yticklabels=list(training_labels.keys()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.show()

if __name__ == "__main__":
    # load csv file
    path = "Path/to/folder/"
    label_list = pd.read_csv(path + 'labels.csv')

    # filter images based on being abnormal lungs or not
    abnormal_lungs = label_list[label_list['Abnormal'] == 'YES']['Image ID'].tolist()
    normal_lungs = label_list[label_list['Abnormal'] == 'NO']['Image ID'].tolist()

    training_labels = {
        'Abnormal': abnormal_lungs,
        'Normal': normal_lungs
    }

    '''
    # dictionary with disease and image names
    diseases_dict = {}
    disease_columns = label_list.columns[4:]  # read through columns excluding IDs and 'Abnormal'

    for disease in disease_columns:
        disease_files = label_list[label_list[disease] == 'YES']['Image ID'].tolist()
        diseases_dict[disease] = disease_files

    # add x-rays with no apparent disease as well
    diseases_dict['None'] = label_list[label_list['Abnormal'] == 'NO']['Image ID'].tolist()

    # plot the data
    plot_data(abnormal_lungs, normal_lungs, diseases_dict)
    '''

    # load images, convert to a dataframe, and separate training and testing data
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_split_data(path, training_labels, list(training_labels.keys()))
    training_and_evaluation(x_train, x_val, x_test, y_train, y_val, y_test)

    ''' Showed to be computationally expensive to train model with specific diseases
    # filter images based on being pneumonia or nodule
    mass = label_list[label_list['Mass'] == 'YES']['Image ID'].tolist()
    nodule = label_list[label_list['Nodule'] == 'NO']['Image ID'].tolist()

    training_labels = {
        'Mass': mass,
        'Nodule': nodule
    }
    print(list(training_labels.keys()))

    # train again
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_split_data(path, training_labels, list(training_labels.keys()))
    training_and_evaluation(x_train, x_val, x_test, y_train, y_val, y_test)
    '''