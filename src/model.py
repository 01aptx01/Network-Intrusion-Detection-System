import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate: float, epochs: int, batch_size: int):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None
        self.b = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, weight_attack: float = 1.0, logger=None):
        N, M = X.shape
        self.W = np.zeros((M, 1))
        self.b = 0.0
        y = y.reshape(-1, 1)

        for epoch in range(self.epochs):
            # 1. Random Shuffle (กันโมเดลจำแพทเทิร์นข้อมูล)
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            num_batches = 0

            # 2. Mini-Batch Gradient Descent
            for i in range(0, N, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]
                batch_N = X_batch.shape[0]

                # Forward Pass
                Z = np.dot(X_batch, self.W) + self.b
                y_hat = self._sigmoid(Z)

                # Cost
                epsilon = 1e-15
                batch_loss = -np.mean(y_batch * np.log(y_hat + epsilon) + (1 - y_batch) * np.log(1 - y_hat + epsilon))
                epoch_loss += batch_loss
                num_batches += 1

                # Backward Pass พร้อม Class Weights สำหรับ Imbalanced Data
                C = 1.0 + y_batch * (weight_attack - 1.0)
                dZ = (y_hat - y_batch) * C

                # Gradients
                dW = (1 / batch_N) * np.dot(X_batch.T, dZ)
                db = (1 / batch_N) * np.sum(dZ)

                # Update
                self.W -= self.lr * dW
                self.b -= self.lr * db

            avg_loss = epoch_loss / num_batches
            self.loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0 and logger:
                logger.info(f"Epoch {epoch+1:03d}/{self.epochs} | Loss: {avg_loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = np.dot(X, self.W) + self.b
        return self._sigmoid(Z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int).flatten()