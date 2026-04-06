import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

def make_early_stopping(patience = 3, monitor = "val_loss"):
    return keras.callbacks.EarlyStopping(
    monitor=monitor,
    patience= patience,
    restore_best_weights = True
)

def build_baseline_model(input_dim, learning_rate = 0.0005):
    model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation = "relu"), 
    layers.Dense(32, activation = "relu"), 
    layers.Dense(1)
   ])
    model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
    loss= "mse",
    metrics = ["mae"]
    )

    return model

def train_baseline_model(
    X_train,
    y_train,
    learning_rate = 0.0005,
    epochs = 20,
    batch_size= 512,
    validation_split= 0.1,
    patience = 3): 
    model = build_baseline_model(
        input_dim= X_train.shape[1],
        learning_rate= learning_rate
    )
    early_stop = make_early_stopping(patience=patience)

    history= model.fit(
    X_train,
    y_train,
    validation_split= validation_split,
    epochs = epochs,
    batch_size= batch_size,
    callbacks = [early_stop],
    verbose= 1
    )
    return model, history 

def build_lstm_model(input_shape, learning_rate=0.001):
    model = keras.Sequential ([
        layers.Input (shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(
        optimizer =keras.optimizers.Adam(learning_rate = learning_rate),
        loss = "mse",
        metrics = ["mae"]
    )
    return model

def train_lstm_model(
    X_train_seq,
    y_train_seq,
    learning_rate = 0.0001,
    epochs = 10,
    batch_size= 128,
    validation_split= 0.1,
    patience = 3): 
    model = build_lstm_model(
        input_shape= (X_train_seq.shape[1], X_train_seq.shape[2]),
        learning_rate= learning_rate
    )
    early_stop = make_early_stopping(patience=patience)

    history= model.fit(
    X_train_seq,
    y_train_seq,
    validation_split= validation_split,
    epochs = epochs,
    batch_size= batch_size,
    callbacks = [early_stop],
    verbose= 1
    )
    return model, history 
