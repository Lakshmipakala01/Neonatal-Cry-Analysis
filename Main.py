import os
import pickle
import zipfile
import tempfile
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# TensorFlow is used only for the proposed CNN.
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

APP_TITLE = "Neonatal Cry Analysis"
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title=APP_TITLE, page_icon="👶", layout="wide")

st.title("👶 Neonatal Cry Analysis")
st.caption("MFCC feature extraction and machine-learning classification of infant cry audio.")

# Keep app data available across Streamlit reruns.
for key, default in {
    "dataset_dir": None,
    "X": None,
    "y": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "trained_models": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def extract_features(file_path):
    """Extract 10 mean MFCC values from an audio file."""
    audio, sample_rate = librosa.load(str(file_path), sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10)
    return np.mean(mfcc, axis=1)


def extract_dataset(dataset_dir):
    """Read WAV files from class subfolders: dataset/class_name/audio.wav."""
    features, labels = [], []
    dataset_dir = Path(dataset_dir)

    class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(
            "No class folders were found. The ZIP must contain folders such as "
            "'hunger/', 'discomfort/', and WAV files inside each folder."
        )

    skipped = []
    for class_dir in class_dirs:
        wav_files = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
        for audio_path in wav_files:
            try:
                features.append(extract_features(audio_path))
                labels.append(class_dir.name)
            except Exception as exc:
                skipped.append(f"{audio_path.name}: {exc}")

    if not features:
        raise ValueError("No readable .wav files were found in the class folders.")

    if skipped:
        st.warning(f"Skipped {len(skipped)} unreadable audio file(s).")

    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def split_dataset(test_size=0.2):
    X = st.session_state.X
    y = st.session_state.y
    if X is None or y is None:
        raise ValueError("Upload and preprocess a dataset first.")

    # Stratification keeps class proportions where each class has enough examples.
    counts = pd.Series(y).value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    return X_train, X_test, y_train, y_test


def show_metrics(model_name, y_true, y_pred, y_score=None, score_classes=None):
    """Display classification metrics and plots in Streamlit."""
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    st.subheader(f"{model_name} — evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.2f}%")
    m2.metric("Precision", f"{precision:.2f}%")
    m3.metric("Recall", f"{recall:.2f}%")
    m4.metric("F1-score", f"{f1:.2f}%")

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{model_name} confusion matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    with st.expander("Classification report"):
        st.text(report)

    # ROC curves are plotted only when valid probability scores are available.
    if y_score is not None and score_classes is not None:
        try:
            classes = list(score_classes)
            y_true_bin = label_binarize(y_true, classes=classes)
            if y_true_bin.ndim == 2 and y_true_bin.shape[1] == y_score.shape[1]:
                fig, ax = plt.subplots(figsize=(8, 5))
                plotted = False
                for i, class_name in enumerate(classes):
                    # ROC is undefined if the test set contains only one outcome for this class.
                    if len(np.unique(y_true_bin[:, i])) < 2:
                        continue
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")
                    plotted = True
                ax.plot([0, 1], [0, 1], "k--", label="Random")
                ax.set_title(f"{model_name} ROC curves (one-vs-rest)")
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                ax.legend(loc="lower right")
                ax.grid(True)
                plt.tight_layout()
                if plotted:
                    st.pyplot(fig)
                else:
                    st.info("ROC curves cannot be calculated because the test set lacks examples from one or more classes.")
                plt.close(fig)
        except Exception as exc:
            st.info(f"ROC curve not available for this split: {exc}")


def train_sklearn_model(model_name):
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    if X_train is None:
        st.error("Split the dataset into training and testing sets first.")
        return

    if model_name == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, min_samples_split=4, random_state=42)
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier(random_state=42)
    elif model_name == "Linear Discriminant Analysis":
        model = LinearDiscriminantAnalysis()
    else:
        st.error("Unknown model.")
        return

    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        score_classes = list(model.classes_) if y_score is not None else None

    st.session_state.trained_models[model_name] = model
    with open(MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.pkl", "wb") as file:
        pickle.dump(model, file)
    show_metrics(model_name, y_test, y_pred, y_score, score_classes)


def train_cnn():
    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow could not be imported in this environment. Check requirements.txt and Streamlit Cloud logs.")
        return
    if st.session_state.X_train is None:
        st.error("Split the dataset first.")
        return

    X_train = st.session_state.X_train[..., np.newaxis]
    X_test = st.session_state.X_test[..., np.newaxis]
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    # Handle test labels that are absent from training split.
    known_mask = np.isin(y_test, encoder.classes_)
    if not known_mask.all():
        st.warning("Some test classes were absent from the training split; those rows will be excluded from CNN evaluation.")
        X_test = X_test[known_mask]
        y_test = y_test[known_mask]
    if len(y_test) == 0:
        st.error("No test examples belong to classes present in the training set.")
        return

    y_test_encoded = encoder.transform(y_test)
    class_count = len(encoder.classes_)
    y_train_cat = to_categorical(y_train_encoded, num_classes=class_count)

    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation="relu"),
        Dropout(0.2),
        Dense(class_count, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    with st.spinner("Training CNN. This may take a few minutes..."):
        history = model.fit(
            X_train, y_train_cat, epochs=100, batch_size=32,
            validation_split=0.2, callbacks=[early_stop], verbose=0
        )

    model.save(MODEL_DIR / "cnn_model.keras")
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as file:
        pickle.dump(encoder, file)
    with open(MODEL_DIR / "cnn_history.pkl", "wb") as file:
        pickle.dump(history.history, file)

    probabilities = model.predict(X_test, verbose=0)
    predicted_indices = np.argmax(probabilities, axis=1)
    y_pred = encoder.inverse_transform(predicted_indices)
    st.session_state.trained_models["CNN"] = model
    show_metrics("CNN", y_test, y_pred, probabilities, list(encoder.classes_))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("accuracy", []), label="Training accuracy")
    axes[0].plot(history.history.get("val_accuracy", []), label="Validation accuracy")
    axes[0].set_title("CNN accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history.get("loss", []), label="Training loss")
    axes[1].plot(history.history.get("val_loss", []), label="Validation loss")
    axes[1].set_title("CNN loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Sidebar: dataset upload and workflow.
st.sidebar.header("1. Dataset")
st.sidebar.write("Upload a ZIP file with one folder per cry class, containing `.wav` files.")
zip_file = st.sidebar.file_uploader("Upload dataset ZIP", type=["zip"])

if zip_file is not None:
    if st.sidebar.button("Extract and load dataset"):
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="cry_dataset_"))
            with zipfile.ZipFile(BytesIO(zip_file.getvalue())) as archive:
                # Basic ZIP path safety check.
                for member in archive.namelist():
                    target = (temp_dir / member).resolve()
                    if not str(target).startswith(str(temp_dir.resolve()) + os.sep) and target != temp_dir.resolve():
                        raise ValueError("The ZIP contains an unsafe file path.")
                archive.extractall(temp_dir)

            # Support ZIP files where all classes are inside one top-level directory.
            candidates = [p for p in temp_dir.iterdir() if p.is_dir()]
            dataset_root = temp_dir
            if len(candidates) == 1:
                child_dirs = [p for p in candidates[0].iterdir() if p.is_dir()]
                if child_dirs:
                    dataset_root = candidates[0]

            X, y = extract_dataset(dataset_root)
            st.session_state.dataset_dir = str(dataset_root)
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.X_train = None
            st.session_state.X_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.trained_models = {}
            st.sidebar.success("Dataset loaded.")
        except Exception as exc:
            st.sidebar.error(f"Could not load dataset: {exc}")

if st.session_state.X is not None:
    X = st.session_state.X
    y = st.session_state.y
    st.subheader("Dataset overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Audio files processed", len(y))
    col2.metric("MFCC features per file", X.shape[1])
    col3.metric("Cry classes", len(np.unique(y)))
    st.write("Class distribution")
    counts = pd.Series(y).value_counts().rename_axis("Class").reset_index(name="Audio files")
    st.dataframe(counts, use_container_width=True)

    st.sidebar.header("2. Preprocessing")
    if st.sidebar.button("Prepare train/test split"):
        try:
            X_train, X_test, y_train, y_test = split_dataset()
            st.sidebar.success("Train/test split completed.")
        except Exception as exc:
            st.sidebar.error(str(exc))

    if st.session_state.X_train is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Training records", len(st.session_state.X_train))
        c2.metric("Testing records", len(st.session_state.X_test))
        c3.metric("Feature count", st.session_state.X_train.shape[1])

        st.sidebar.header("3. Train models")
        model_choice = st.sidebar.selectbox(
            "Choose a classifier",
            ["SVM", "KNN", "Decision Tree", "AdaBoost", "Linear Discriminant Analysis", "CNN"]
        )
        if st.sidebar.button("Train and evaluate"):
            if model_choice == "CNN":
                train_cnn()
            else:
                train_sklearn_model(model_choice)

st.divider()
st.subheader("Predict a cry category")
st.write("Upload a `.wav` file. Train the CNN above first, or deploy a previously trained CNN and label encoder in the `model/` folder.")
audio_file = st.file_uploader("Choose test audio", type=["wav"], key="prediction_audio")

if audio_file is not None:
    st.audio(audio_file)
    if st.button("Predict cry category"):
        model_path = MODEL_DIR / "cnn_model.keras"
        encoder_path = MODEL_DIR / "label_encoder.pkl"
        if not model_path.exists() or not encoder_path.exists():
            st.error("CNN model or label encoder not found. Train the CNN first.")
        elif not TENSORFLOW_AVAILABLE:
            st.error("TensorFlow is unavailable in this environment.")
        else:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_audio.write(audio_file.getvalue())
                    audio_path = temp_audio.name
                features = extract_features(audio_path).reshape(1, 10, 1)
                model = load_model(model_path)
                with open(encoder_path, "rb") as file:
                    encoder = pickle.load(file)
                probabilities = model.predict(features, verbose=0)[0]
                predicted_index = int(np.argmax(probabilities))
                predicted_label = encoder.inverse_transform([predicted_index])[0]
                st.success(f"Predicted cry category: **{predicted_label}**")
                st.write(f"Model confidence: {probabilities[predicted_index] * 100:.2f}%")
                st.caption("This is a model prediction, not a medical diagnosis.")

                audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
                fig, ax = plt.subplots(figsize=(10, 3))
                librosa.display.waveshow(audio, sr=sample_rate, ax=ax)
                ax.set_title(f"Audio waveform — predicted: {predicted_label}")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            finally:
                try:
                    os.remove(audio_path)
                except (NameError, OSError):
                    pass

st.caption("Note: model performance depends on dataset size, class balance, recording quality, and correct labels.")
