from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_generation.generated_data import load_new_timeseries_data, load_original_timeseries_data

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



# Normalize each series individually to [0, 1]
def normalize_series(series):
    min_val = np.min(series)
    max_val = np.max(series)
    return (series - min_val) / (max_val - min_val + 1e-8), min_val, max_val


def create_sequences(series, input_len=10, output_len=2):
    X, y = [], []
    for start in range(len(series) - input_len - output_len + 1):
        end = start + input_len
        X.append(series[start:end])
        y.append(series[end:end + output_len])
    return np.array(X), np.array(y)


def plot_prediction_vs_actual(pred, true, x_input, scaler=None, denormalize=True, title="Forecast"):
    """
    Plots a forecast (predicted vs actual) given the predicted, true, and input values.

    Parameters:
    - pred: predicted future values (array of shape (output_len,))
    - true: ground truth future values (same shape as pred)
    - x_input: input sequence used for forecasting (shape: (input_len,))
    - scaler: (min, max) tuple for denormalization (optional)
    - denormalize: whether to denormalize values
    - title: plot title
    """
    if denormalize and scaler is not None:
        min_v, max_v = scaler
        x_input = x_input * (max_v - min_v) + min_v
        pred = pred * (max_v - min_v) + min_v
        true = true * (max_v - min_v) + min_v

    # Build time axis
    input_len = len(x_input)
    output_len = len(pred)
    total_len = input_len + output_len
    time = np.arange(total_len)

    # Combine for plotting
    full_actual = np.concatenate([x_input, true])
    full_pred = np.concatenate([x_input, pred])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(time, full_actual, label="Actual", marker='o')
    plt.plot(time, full_pred, label="Predicted", marker='x')
    plt.axvline(input_len - 1, color='gray', linestyle='--')  # mark prediction start
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    ### y data is not for forecasting, should update it
    #df = pd.read_csv("data_generation/time_series_data.csv")
    #df = df.drop("series_id", axis='columns')
    # data = df.values.astype(np.float32)

    og_X, og_y = load_original_timeseries_data(['mixed'], as_numpy=True)
    new_X, new_y = load_new_timeseries_data(as_numpy=True)
    list_data = np.concatenate((og_X, new_X))
    data = np.asarray(list_data)

    normalized_data = []
    scalers = []

    for s in data:
        norm, min_v, max_v = normalize_series(s)
        normalized_data.append(norm)
        scalers.append((min_v, max_v))

    normalized_data = np.array(normalized_data)

    X_all, y_all = [], []
    for series in normalized_data:
        X_seq, y_seq = create_sequences(series)
        X_all.append(X_seq)
        y_all.append(y_seq)

    X_all = np.concatenate(X_all, axis=0)  # shape: (n_samples, input_len)
    y_all = np.concatenate(y_all, axis=0)  # shape: (n_samples, output_len)

    # Add channel dimension for TensorFlow
    X_all = np.expand_dims(X_all, axis=-1)  # shape: (n_samples, input_len, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(10, 1)),
        tf.keras.layers.Conv1D(filters=5, # Number of kernels
                               kernel_size=3, # Size of each kernel
                               padding='causal', # Pads the start if needed, so the model doesn't a future time step to predict itself
                               activation='relu'),
        tf.keras.layers.LSTM(24, return_sequences=False), # Sequence-to-vector
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2)  # 2-step prediction
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    print(model.summary())

    epochs = 50
    # Train and test the model
    history = model.fit(X_all, y_all, epochs=epochs, batch_size=8, validation_split=0.2, 
                        # callbacks=[lr_schedule]
                        )


    # Example: predict the next 4 steps from the last 12 values of the first series
    # sample_input = normalized_data[0][-16:-4]  # pick a valid 12-step slice
    # sample_input = np.expand_dims(sample_input, axis=(0, -1))  # shape: (1, 12, 1)

    # prediction = model.predict(sample_input)
    # print(f"Normalized Prediction: {prediction}")
    
    # min_v, max_v = scalers[0]
    # pred_real_scale = prediction * (max_v - min_v) + min_v
    # print(f"Un-normalized Prediction: {pred_real_scale}")

    # Choose a sample
    sample_index = 24
    x_input = X_all[sample_index].squeeze()
    true = y_all[sample_index]
    pred = model.predict(np.expand_dims(X_all[sample_index], axis=0), verbose=0)[0]

    # Plot it
    plot_prediction_vs_actual(pred, true, x_input, scaler=scalers[0], title="Series 0 Forecast")

    # predictions = model.predict(X_test)
    # predicted_classes = tf.argmax(predictions, axis=1).numpy()
    # print(f"Predicted Classes: {predicted_classes}")
    # print(f"Actual Classes: {y_test}")

    # loss, accuracy = dl_model.evaluate(X_test, y_test)
    # print(f"Test Loss: {loss:.2f}")
    # print(f"Test Accuracy: {accuracy:.2f}")

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1) # Epoch numbers

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['learning_rate'], train_loss)
    plt.title('Learning Rate vs. Loss')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.grid(True)

    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, train_loss, 'r', label='Training Loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)

    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, train_accuracy, 'r', label='Training Accuracy')
    # plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.grid(True)
    
    plt.show()

