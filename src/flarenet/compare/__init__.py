"""Compare forecasting models on identical windows: CNN-BiLSTM (FlareNet), LSTM, Random Forest, XGBoost, gradient boosting."""

from flarenet.compare.metrics import full_metrics

__all__ = ["full_metrics"]
