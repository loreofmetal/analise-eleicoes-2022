import os
import pickle

"""
Utilitários para carregar métricas dos modelos de regressão
"""

def load_modelo_inicial_metrics():
    """
    Função para carregar as métricas do modelo inicial nos outros notebooks.

    Returns:
        dict: Dicionário com todas as métricas do modelo inicial
    """
    metrics_file = os.path.join("..", "data", "metrics", "modelo_inicial_metrics.pkl")

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {metrics_file}")

    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)

    return metrics

def load_regional_model_metrics():
    """
    Função para carregar as métricas do modelo regional (Bolsa Família + Nordeste) nos outros notebooks.

    Returns:
        dict: Dicionário com todas as métricas do modelo regional
    """
    metrics_file = os.path.join("..", "data", "metrics", "regional_model_metrics.pkl")

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {metrics_file}")

    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)

    return metrics
