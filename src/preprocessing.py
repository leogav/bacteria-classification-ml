import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def drop_missing(df):
    """Remove linhas com valores nulos."""
    return df.dropna(axis=0, how='any')

def encode_labels(y):
    """Transforma rótulos categóricos em valores numéricos."""
    encoder = LabelEncoder()
    return encoder.fit_transform(y), encoder

def interpolate_points(data, num_interpolations=3):
    """
    Interpola pontos para aumentar o conjunto de dados.
    """
    interpolated_data = []
    for i in range(len(data) - 1):
        for j in range(1, num_interpolations + 1):
            ratio = j / (num_interpolations + 1)
            point = (1 - ratio) * data[i] + ratio * data[i + 1]
            interpolated_data.append(point)
    return np.array(interpolated_data)

def normalize_data(X_train, X_test):
    """Normaliza dados com StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
