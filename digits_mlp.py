"""
HandWritting Digits Classification Using MLP (Scikit-learn)

This script demonstrates a complete, clean, and principled pipeline for:
- Loading the sklearn digits dataset
- Proper preprocessing and normalization
- Training a Multi-Layer Perceptron (MLP) classifier
- Evaluating the model using accuracy and confusion matrix
- Predicting a digit from an external image

The implementation is designed to be educational, reproducible,
and scientifically defensible.
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from skimage.filters import threshold_otsu

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_digits():
    """
    Load the sklearn digits dataset and preprocess it.

    Steps:
    - Load digits dataset (8x8 grayscale images)
    - Flatten images into 1D feature vectors
    - Normalize pixel values to range [0, 1]

    Returns
    -------
    X : np.ndarray
        Normalized feature matrix of shape (n_samples, 64)
    y : np.ndarray
        Target labels
    scaler : MinMaxScaler
        Fitted scaler for future transformations
    digits : Bunch
        Original digits dataset object
    """
    digits = datasets.load_digits()

    X = digits.images.reshape(len(digits.images), -1)
    y = digits.target

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X = scaler.fit_transform(X)

    return X, y, scaler, digits


def plot_multi(start_index, digits, nplots=16, grid_size=(4, 4)):
    """
    Plot multiple digit images from the sklearn digits dataset.

    Parameters
    ----------
    start_index : int
        Starting index in the dataset.
    digits : Bunch
        Sklearn digits dataset containing images and targets.
    nplots : int, optional
        Number of images to plot (default is 16).
    grid_size : tuple[int, int], optional
        Grid size for subplots (rows, cols), default is (4, 4).

    Returns
    -------
    None
    """
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    if start_index + nplots > len(digits.images):
        raise IndexError("Requested range exceeds dataset size")

    fig = plt.figure(figsize=(5, 5))

    for j in range(nplots):
        ax = fig.add_subplot(grid_size[0], grid_size[1], j + 1)

        image = digits.images[start_index + j]
        label = int(digits.target[start_index + j])

        ax.imshow(image, cmap="binary")
        ax.set_title(str(label))
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def build_mlp_model():
    """
    Build and configure the MLP classifier.

    Architecture rationale:
    - Two hidden layers with moderate width
    - ReLU activation for stable gradients
    - Adam optimizer for adaptive learning rate
    - Early stopping to prevent overfitting

    Returns
    -------
    model : MLPClassifier
        Configured but untrained MLP model
    """
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=42,
        verbose=False
    )
    return model


def plot_training_loss(model):
    """
    Plot the training loss curve of the MLP.

    Parameters
    ----------
    model : MLPClassifier
        Trained MLP model
    """
    plt.figure(figsize=(6, 4))
    plt.plot(model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.grid(True)
    plt.show()


def evaluate_model(model, X_test, y_test, digits):
    """
    Evaluate the trained model on the test set.

    Metrics:
    - Accuracy
    - Confusion matrix

    Parameters
    ----------
    model : MLPClassifier
        Trained model
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        True test labels
    digits : Bunch
        Digits dataset (for label names)
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=digits.target_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def preprocess_external_image(image_path, scaler):
    """
    Preprocess an external image for digit prediction.

    Steps:
    - Load image in grayscale
    - Resize to 8x8 (digits dataset resolution)
    - Apply Otsu thresholding for binarization
    - Flatten and normalize using training scaler

    Parameters
    ----------
    image_path : str
        Path to the image file
    scaler : MinMaxScaler
        Fitted scaler from training data

    Returns
    -------
    img_flat : np.ndarray
        Preprocessed image of shape (1, 64)
    """
    img = imread(image_path, as_gray=True)
    img = resize(img, (8, 8), anti_aliasing=True)

    thresh = threshold_otsu(img)
    img = img < thresh

    img = img.astype(float).reshape(1, -1)
    img = scaler.transform(img)

    return img


def show_test_sample_with_prediction(model, X_test, y_test, index):
    """
    Display a test image along with its true label and predicted label.

    Parameters
    ----------
    model : MLPClassifier
        Trained classification model
    X_test : np.ndarray
        Test features of shape (n_samples, 64)
    y_test : np.ndarray
        True labels of shape (n_samples,)
    index : int
        Index of the test sample to visualize
    """
    if X_test.ndim != 2 or X_test.shape[1] != 64:
        raise ValueError("X_test must have shape (n_samples, 64)")

    if not (0 <= index < len(X_test)):
        raise IndexError("Index out of range for X_test")

    image = X_test[index].reshape(8, 8)
    true_label = int(y_test[index])
    predicted_label = int(model.predict(X_test[index].reshape(1, -1))[0])

    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="binary")
    plt.title(f"True: {true_label} | Predicted: {predicted_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_external_image_prediction(model, image_path, scaler):
    """
    Display an external digit image along with the model's predicted label.

    Parameters
    ----------
    model : MLPClassifier
        Trained MLP classifier
    image_path : str
        Path to the external image
    scaler : MinMaxScaler
        Fitted scaler used during training
    """
    # Load image
    img = imread(image_path, as_gray=True)
    img_resized = resize(img, (8, 8), anti_aliasing=True)

    # Otsu threshold for binarization
    threshold = threshold_otsu(img_resized)
    img_binary = img_resized < threshold

    # Flatten and scale
    img_flat = img_binary.astype(float).reshape(1, -1)
    img_scaled = scaler.transform(img_flat)

    # Predict
    prediction = int(model.predict(img_scaled)[0])

    # Plot
    plt.figure(figsize=(3, 3))
    plt.imshow(img_binary, cmap="binary")
    plt.title(f"Predicted: {prediction}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_wrong_predicts(model, X_test, y_test, n_rows=2, n_cols=3):
    """
    Display a grid of test samples where the model predicted incorrectly.
    """
    y_pred = model.predict(X_test)
    wrong_indices = np.where(y_pred != y_test)[0]

    n_to_show = min(len(wrong_indices), n_rows * n_cols)
    if n_to_show == 0:
        print("No misclassified samples to show.")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = np.array(axes).flatten()

    for i in range(n_to_show):
        idx = int(wrong_indices[i])
        img = X_test[idx].reshape(8, 8)
        true_label = int(y_test[idx])
        predicted_label = int(y_pred[idx])

        ax = axes[i]
        ax.imshow(img, cmap="binary")
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}")
        ax.axis("off")

    for j in range(n_to_show, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main execution pipeline:
    - Load and preprocess data
    - Split into train and test sets
    - Train MLP model
    - Evaluate performance
    - Predict digit from external image
    - Show wrong predictions
    """
    X, y, scaler, digits = load_and_preprocess_digits()
    plot_multi(0, digits=digits)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = build_mlp_model()
    model.fit(X_train, y_train)

    plot_training_loss(model)
    evaluate_model(model, X_test, y_test, digits)

    # Example external image prediction
    # Ensure the image contains a centered handwritten digit
    image_path = "2.jpg"
    try:
        show_external_image_prediction(model, image_path, scaler)
        img = preprocess_external_image(image_path, scaler)
        prediction = model.predict(img)
        print(f"Predicted digit for external image: {int(prediction[0])}")
    except FileNotFoundError:
        print("External image not found. Skipping external prediction.")

    show_wrong_predicts(model, X_test, y_test, n_rows=4, n_cols=3)


if __name__ == "__main__":
    main()
