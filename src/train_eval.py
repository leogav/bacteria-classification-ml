import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.preprocessing import interpolate_points, normalize_data
from src.models import create_lstm_model, create_rnn_model

# ===============================
# Treino e Avaliação - Deep Learning
# ===============================

def train_evaluate_lstm_rnn(X, y, model_type="lstm", num_folds=5, num_interpolations=3,
                            epochs=150, batch_size=64):
    """
    Treina e avalia modelos LSTM ou RNN usando validação cruzada estratificada.
    model_type: "lstm" ou "rnn"
    """

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    accuracies, f1_scores, recall_scores, precision_scores = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train_orig, X_test = X[train_idx], X[test_idx]
        y_train_orig, y_test = y[train_idx], y[test_idx]

        # Interpolação
        interpolated_X, interpolated_y = [], []
        for label in np.unique(y_train_orig):
            mask = (y_train_orig == label)
            data_points = X_train_orig[mask]
            interp_data = interpolate_points(data_points, num_interpolations)
            interpolated_X.extend(interp_data)
            interpolated_y.extend([label] * len(interp_data))

        X_train = np.concatenate([X_train_orig, interpolated_X])
        y_train = np.concatenate([y_train_orig, interpolated_y])

        # Normalização
        X_train, X_test, _ = normalize_data(X_train, X_test)

        # Ajuste para formato RNN/LSTM: (samples, timesteps, features)
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

        # Modelo
        if model_type == "lstm":
            model = create_lstm_model(input_shape=(X_train.shape[1], 1),
                                      num_classes=len(np.unique(y)))
        else:
            model = create_rnn_model(input_shape=(X_train.shape[1], 1),
                                     num_classes=len(np.unique(y)))

        # Treinamento
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Avaliação
        preds = np.argmax(model.predict(X_test), axis=1)
        accuracies.append(accuracy_score(y_test, preds))
        f1_scores.append(f1_score(y_test, preds, average='weighted'))
        recall_scores.append(recall_score(y_test, preds, average='weighted'))
        precision_scores.append(precision_score(y_test, preds, average='weighted'))

        print(f"Fold {fold}: Acc={accuracies[-1]:.4f}, F1={f1_scores[-1]:.4f}, "
              f"Recall={recall_scores[-1]:.4f}, Precision={precision_scores[-1]:.4f}")

    print("\nMédias:")
    print(f"Acurácia: {np.mean(accuracies):.4f}")
    print(f"F1-Score: {np.mean(f1_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f}")
    print(f"Precisão: {np.mean(precision_scores):.4f}")


# ===============================
# Treino e Avaliação - Modelos Clássicos
# ===============================

def train_evaluate_classic(X, y, model, num_folds=5, num_interpolations=3):
    """
    Treina e avalia modelos clássicos (KNN, DT, RF, XGBoost) usando validação cruzada estratificada.
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    accuracies, f1_scores, recall_scores, precision_scores = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train_orig, X_test = X[train_idx], X[test_idx]
        y_train_orig, y_test = y[train_idx], y[test_idx]

        # Interpolação
        interpolated_X, interpolated_y = [], []
        for label in np.unique(y_train_orig):
            mask = (y_train_orig == label)
            data_points = X_train_orig[mask]
            interp_data = interpolate_points(data_points, num_interpolations)
            interpolated_X.extend(interp_data)
            interpolated_y.extend([label] * len(interp_data))

        X_train = np.concatenate([X_train_orig, interpolated_X])
        y_train = np.concatenate([y_train_orig, interpolated_y])

        # Pipeline com escalonamento + SMOTE + modelo
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Métricas
        accuracies.append(accuracy_score(y_test, preds))
        f1_scores.append(f1_score(y_test, preds, average='weighted'))
        recall_scores.append(recall_score(y_test, preds, average='weighted'))
        precision_scores.append(precision_score(y_test, preds, average='weighted'))

        print(f"Fold {fold}: Acc={accuracies[-1]:.4f}, F1={f1_scores[-1]:.4f}, "
              f"Recall={recall_scores[-1]:.4f}, Precision={precision_scores[-1]:.4f}")

    print("\nMédias:")
    print(f"Acurácia: {np.mean(accuracies):.4f}")
    print(f"F1-Score: {np.mean(f1_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f}")
    print(f"Precisão: {np.mean(precision_scores):.4f}")

