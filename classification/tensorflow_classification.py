from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from data_generation.generated_data import load_new_timeseries_data, load_original_timeseries_data 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Get the current learning rate from the optimizer
        current_lr = tf.keras.backend.eval(self.model.optimizer.lr)
        logs['lr'] = current_lr
        print(f"Epoch {epoch+1}: Learning Rate = {current_lr}")

def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.2))


def main():
    og_X, og_y = load_original_timeseries_data(['mixed'], as_numpy=True)
    new_X, new_y = load_new_timeseries_data(as_numpy=True)
    X = np.concatenate((og_X, new_X))
    y = np.concatenate((og_y, new_y))

    X_norm = normalize(X, norm="l2") # Reduces impact of different scales in the input data
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.1)

    dl_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(12,)),
        tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.004) # Found 0.004 to be most optimal from lr_scheduler

    # Compile the model
    dl_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', # Suitable for integer labels
        metrics=['accuracy']
    )

    print(dl_model.summary())

    epochs = 100
    # Train and test the model
    history = dl_model.fit(X_train, y_train, epochs=epochs, batch_size=6, validation_split=0.2, 
                           #callbacks=[lr_schedule]
                           )
    predictions = dl_model.predict(X_test)
    predicted_classes = tf.argmax(predictions, axis=1).numpy()
    print(f"Predicted Classes: {predicted_classes}")
    print(f"Actual Classes: {y_test}")

    loss, accuracy = dl_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.2f}")
    print(f"Test Accuracy: {accuracy:.2f}")

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1) # Epoch numbers

    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['learning_rate'], train_loss)
    # plt.title('Learning Rate vs. Loss')
    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

