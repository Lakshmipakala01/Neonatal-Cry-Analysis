from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display
import cv2


from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.preprocessing import LabelEncoder,label_binarize

accuracy = []
precision = []
recall = []
fscore = []

model_folder = "Model"

def Upload_Dataset():
    global filename,categories,num_classes
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    num_classes = len(categories)
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load audio file
    
    # Feature extraction
    features = []
    
    # 1. MFCC (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc_mean = np.mean(mfcc, axis=1)
    features.extend(mfcc_mean)
      
    return np.array(features)
    
def load_dataset(X_file,Y_file,model_folder,dataset_path):
    X = []
    y = []
    for subfolder in os.listdir(dataset_path):          
        class_label = subfolder
        subfolder_path = os.path.join(dataset_path, subfolder)
        for file in os.listdir(subfolder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(subfolder_path, file)
                print(file_path)
                features = extract_features(file_path)
                X.append(features)
                y.append(class_label)
    np.save(X_file, X)
    np.save(Y_file, y)
    return np.array(X), np.array(y)

def Preprocess_Dataset():
    global X,Y,filename
    filepath = Path(filename)
    last_folder=filepath.name
 
    
    X_file = os.path.join(model_folder, "X.txt.npy")
    Y_file = os.path.join(model_folder, "Y.txt.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
    else:
        X, Y = load_dataset(X_file,Y_file,model_folder,last_folder)
    
#    desired_samples = 3000
 #   X, Y = resample(X, Y, n_samples=desired_samples, random_state=42)
    
    text.insert(END, "Preprocessing and MFCC Feature Extraction completed on Dataset: " + str(filename) + "\n\n")
    text.insert(END, "Input MFCC Feature Set Size: " + str(X.shape) + "\n\n")


def Train_Test_Splitting():
    global X,Y
    global x_train,y_train,x_test,y_test

    # Create a count plot


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
# Display information about the dataset
    text.delete('1.0', END)
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to train: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to test: " + str(x_test.shape[0]) + "\n\n")

def Calculate_Metrics(algorithm, predict, y_test,y_score):
    global categories
    if not os.path.exists('results'):
        os.makedirs('results')
        
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure() 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.savefig(f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png")
    plt.show()     
    print(y_score)

    if y_score is not None:
        y_test_bin = label_binarize(y_test, classes=range(len(categories)))
        fpr, tpr, roc_auc = dict(), dict(), dict()

        for i in range(len(categories)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            

        plt.figure(figsize=(10, 8))
        for i in range(len(categories)):
            plt.plot(fpr[i], tpr[i], label=f'Class {categories[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.title(f"{algorithm} ROC Curves (One-vs-Rest)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
        plt.show()
        

def existing_classifier():
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)

    mlmodel = SVC(probability=True)  # enable probability estimation
    mlmodel.fit(x_train, y_train)
    
    y_pred = mlmodel.predict(x_test)
    if hasattr(mlmodel, "predict_proba"):
        y_score = mlmodel.predict_proba(x_test)
    else:
        y_score = mlmodel.decision_function(x_test)
    
    Calculate_Metrics("Existing SVM", y_pred, y_test, y_score=y_score)


def existing_classifier1():
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)

    mlmodel = KNeighborsClassifier(n_neighbors=500)
    mlmodel.fit(x_train, y_train)
    
    y_pred = mlmodel.predict(x_test)
    if hasattr(mlmodel, "predict_proba"):
        y_score = mlmodel.predict_proba(x_test)
    else:
        y_score = None  # No alternative available
    
    Calculate_Metrics("Existing KNN", y_pred, y_test, y_score=y_score)


def existing_classifier2():
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)

    mlmodel = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, min_samples_split=8)
    mlmodel.fit(x_train, y_train)
    
    y_pred = mlmodel.predict(x_test)
    y_score = mlmodel.predict_proba(x_test)  # DTC supports predict_proba
    
    Calculate_Metrics("Existing DTC", y_pred, y_test, y_score=y_score)


def existing_classifier4():
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)

    mlmodel = AdaBoostClassifier()
    mlmodel.fit(x_train, y_train)
    
    y_pred = mlmodel.predict(x_test)
    y_score = mlmodel.predict_proba(x_test)  # AdaBoost supports predict_proba
    
    Calculate_Metrics("Existing AdaBoost Classifier", y_pred, y_test, y_score=y_score)


def existing_classifier5():
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)

    mlmodel = LinearDiscriminantAnalysis()
    mlmodel.fit(x_train, y_train)
    
    y_pred = mlmodel.predict(x_test)
    y_score = mlmodel.predict_proba(x_test)  # LDA supports predict_proba
    
    Calculate_Metrics("Existing Linear Discriminant Analysis", y_pred, y_test, y_score=y_score)
    

def cnn_classifier():
    global x_train, y_train, x_test, y_test, mlmodel, text,num_classes
    text.delete('1.0', END)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Save the LabelEncoder
    encoder_path = "model/label_encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    model_path = "model/cnn_model.h5"
    history_path = "model/cnn_model_history.pkl"

    if os.path.exists(model_path):
        model = load_model(model_path)

     
    else:
        # Reshape data for CNN input
        x_train_reshaped = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test_reshaped = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        y_train_cat = to_categorical(y_train_encoded, num_classes=num_classes)
        y_test_cat = to_categorical(y_test_encoded, num_classes=num_classes)

        # Build CNN model
        model = Sequential()
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model
        history = model.fit(
            x_train_reshaped, y_train_cat,
            epochs=100, batch_size=32, verbose=2,
            validation_split=0.2
        )

        # Save model and history
        model.save(model_path)
        with open(history_path, "wb") as f:
            pickle.dump(history.history, f)

    # Predict and evaluate
    y_pred = np.argmax(model.predict(x_test.reshape((x_test.shape[0], x_test.shape[1], 1))), axis=1)
    # Load training history
    history_path = "model/cnn_model_history.pkl"
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    # Plot training & validation accuracy
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot training & validation loss
    plt.subplot(1,2,2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('CNN Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    y_score = model.predict(x_test.reshape((x_test.shape[0], x_test.shape[1], 1)))  # returns probabilities

    Calculate_Metrics("CNN Classifier", y_pred, y_test_encoded, y_score=y_score)


 

def Prediction():
   # File selection dialog
    filename = filedialog.askopenfilename(initialdir="testvoice")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')

    # Feature extraction
    features = extract_features(filename)
    features = features.reshape(1, -1)
    
    # Load trained CNN model
    model_path = "model/cnn_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        text.insert(END, "CNN model not found.\n")
        return

    # Load LabelEncoder
    encoder_path = "model/label_encoder.pkl"
    if os.path.exists(encoder_path):
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)
    else:
        text.insert(END, "Label encoder not found.\n")
        return

    # Reshape features for CNN prediction
    features = features.reshape((1, features.shape[1], 1))

    # Predict using CNN
    cnn_features = model.predict(features)
    predicted_category_idx = np.argmax(cnn_features, axis=1)
    
    # Decode using LabelEncoder
    predicted_category = le.inverse_transform(predicted_category_idx)[0]
    text.insert(END, f"Predicted Outcome From Test Audio is: {predicted_category}\n\n")

    # Load audio file
    y, sr = librosa.load(filename, sr=None)

    # Create a waveplot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Save plot to an image buffer
    plt.savefig("waveplot.png")
    plt.close()

    # Load the saved waveplot image
    waveplot_img = cv2.imread("waveplot.png")
    if waveplot_img is not None:
        # Add the predicted category text on the waveplot
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)  # Position to place the text
        font_scale = 1
        font_color = (0, 0, 255)  # Red color in BGR
        thickness = 2
        line_type = cv2.LINE_AA

        cv2.putText(waveplot_img, f"Predicted: {predicted_category}", position, font, font_scale, font_color, thickness, line_type)

        # Display the waveplot with annotation
        cv2.imshow("Waveplot with Prediction", waveplot_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def close():
    main.destroy()


import tkinter as tk
from PIL import Image, ImageTk


main = tkinter.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

# ------------------ Load and place background ------------------
bg_image = Image.open("background.png")  
bg_image = bg_image.resize((screen_width, screen_height))
bg_photo = ImageTk.PhotoImage(bg_image)

background_label = tk.Label(main, image=bg_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


font = ('times', 18, 'bold')
title = Label(main, text="Prediction of Baby Cry using CNN")
title.config(bg='white', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(relx=0.5, y=20, anchor='n')  # Centered at top
title.pack()

font1 = ('times', 13, 'bold')
Button1 = Button(main, text="Upload Dataset", command=Upload_Dataset)
Button1.place(x=20,y=100)
Button1.config(font=font1)

Button1 = Button(main, text="Preprocess Dataset", command=Preprocess_Dataset)
Button1.place(x=20,y=150)
Button1.config(font=font1)

Button1 = Button(main, text="Train Test Splitting", command=Train_Test_Splitting)
Button1.place(x=20,y=200)
Button1.config(font=font1) 


Button1 = Button(main, text="Support Vector Machine", command=existing_classifier)
Button1.place(x=20,y=250)
Button1.config(font=font1)

Button1 = Button(main, text="K Nearest Neighbour", command=existing_classifier1)
Button1.place(x=20,y=300)
Button1.config(font=font1)

Button1 = Button(main, text="Decision Tree Classifier", command=existing_classifier2)
Button1.place(x=20,y=350)
Button1.config(font=font1)


Button1 = Button(main, text="AdaBoost Classifier", command=existing_classifier4)
Button1.place(x=20,y=400)
Button1.config(font=font1)

Button1 = Button(main, text="Linear Discriminant Analysis", command=existing_classifier5)
Button1.place(x=20,y=450)
Button1.config(font=font1)

Button1 = Button(main, text="Proposed CNN", command=cnn_classifier)
Button1.place(x=20,y=500)
Button1.config(font=font1)

Button1 = Button(main, text="Prediction", command=Prediction)
Button1.place(x=20,y=550)
Button1.config(font=font1)


Button1 = Button(main, text="Exit", command=close)
Button1.place(x=20,y=600)
Button1.config(font=font1)



                            
font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=70)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=300,y=100)
text.config(font=font1)
main.config(bg='SeaGreen1')

main.mainloop()