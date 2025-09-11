# bacteria-classification-ml
Aplicação de Machine Learning e Deep Learning para identificação e classificação de bactérias em dados biomédicos. Inclui modelos LSTM, RNN, KNN, Árvore de Decisão, Random Forest e XGBoost.

# Bacteria Classification using Machine Learning

Este projeto faz parte do meu **Mestrado em Computação Aplicada (USP)** e tem como objetivo a **classificação de bactérias** a partir de dados obtidos por espectroscopia de impedância elétrica.

## Modelos implementados
- LSTM (Long Short-Term Memory)
- RNN simples
- KNN (K-Nearest Neighbors)
- Árvore de Decisão
- Random Forest
- XGBoost

## Tecnologias utilizadas
- Python
- TensorFlow / Keras
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Imbalanced-learn (SMOTE)

## Estrutura
- `notebooks/` → versões exploratórias dos modelos.
- `src/preprocessing.py` → funções para limpeza, interpolação e preparação dos dados.
- `src/models.py` → definições de diferentes modelos (LSTM, RNN, KNN, etc.).
- `src/train_eval.py` → treino, validação cruzada e métricas de desempenho.

## Resultados
- LSTM alcançou **X% de acurácia média** em validação cruzada.
- Comparação com KNN, Random Forest e XGBoost.

## Como executar
1. Clone este repositório:
   ```bash
   git clone https://github.com/leogav/bacteria-classification-ml.git
   cd bacteria-classification-ml
   
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt

3. Abra os notebooks na pasta notebooks/ no Jupyter Notebook ou VSCode.
