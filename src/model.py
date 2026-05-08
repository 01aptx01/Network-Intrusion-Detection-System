import logging
from typing import Optional

import numpy as np


class BinaryLogisticRegression:
    """Logistic regression สองคลาสด้วย NumPy (mini-batch) + ถ่วง gradient ฝั่ง attack"""

    def __init__(self, learning_rate: float, epochs: int, batch_size: int):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        attack_loss_weight: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        num_samples, num_features = features.shape
        self.weights = np.zeros((num_features, 1))
        self.bias = 0.0
        labels = labels.reshape(-1, 1)

        for epoch in range(self.epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            features_shuffled = features[indices]
            labels_shuffled = labels[indices]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, num_samples, self.batch_size):
                batch_x = features_shuffled[start : start + self.batch_size]
                batch_y = labels_shuffled[start : start + self.batch_size]
                batch_size_actual = batch_x.shape[0]

                logits = np.dot(batch_x, self.weights) + self.bias
                predictions = self._sigmoid(logits)

                epsilon = 1e-15
                batch_loss = -np.mean(
                    batch_y * np.log(predictions + epsilon)
                    + (1 - batch_y) * np.log(1 - predictions + epsilon)
                )
                epoch_loss += batch_loss
                num_batches += 1

                weight_multiplier = 1.0 + batch_y * (attack_loss_weight - 1.0)
                delta = (predictions - batch_y) * weight_multiplier

                grad_weights = (1 / batch_size_actual) * np.dot(batch_x.T, delta)
                grad_bias = (1 / batch_size_actual) * np.sum(delta)

                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias

            average_loss = epoch_loss / num_batches
            self.loss_history.append(average_loss)

            if (epoch + 1) % 10 == 0 and logger is not None:
                logger.info(f"Epoch {epoch + 1:03d}/{self.epochs} | Loss: {average_loss:.4f}")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = np.dot(features, self.weights) + self.bias
        return self._sigmoid(logits)

    def predict(self, features: np.ndarray, decision_threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= decision_threshold).astype(int).flatten()
