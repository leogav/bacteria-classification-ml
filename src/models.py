import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===============================
# Modelos Deep Learning
# ===============================

def create_lstm_model(input_shape, num_classes):
    """Modelo LSTM para classificação."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_rnn_model(input_shape, num_classes):
    """Modelo RNN simples para classificação."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.SimpleRNN(64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ===============================
# Modelos Clássicos
# ===============================

def create_knn_model(n_neighbors=5):
    """Modelo KNN."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def create_decision_tree_model(max_depth=None, random_state=42):
    """Modelo Árvore de Decisão."""
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

def create_random_forest_model(n_estimators=100, max_depth=None, random_state=42):
    """Modelo Random Forest."""
    return RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  random_state=random_state)

def create_xgboost_model(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42):
    """Modelo XGBoost."""
    return XGBClassifier(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         max_depth=max_depth,
                         random_state=random_state,
                         use_label_encoder=False,
                         eval_metric="mlogloss")

