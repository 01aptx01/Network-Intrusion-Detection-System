import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        ฟังก์ชัน Sigmoid พร้อมระบบป้องกัน Numerical Instability
        Time Complexity: O(N)
        Space Complexity: O(N)
        """
        # ป้องกัน Overflow ใน e^(-z) หาก z มีค่าน้อยหรือมากเกินไป
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        กระบวนการ Training Model ด้วย Gradient Descent (Vectorized)
        
        N = จำนวนตัวอย่าง (Samples)
        M = จำนวนฟีเจอร์ (Features)
        """
        N, M = X.shape
        
        # 1. กำหนดค่าเริ่มต้น (Initialize Parameters)
        # W รูปร่าง (M, 1) และ b เป็น Scalar
        self.W = np.zeros((M, 1))
        self.b = 0.0
        
        # จัดรูปร่าง y ให้เป็น Column Vector (N, 1) เพื่อให้ Matrix Broadcasting ทำงานได้ถูกต้อง
        y = y.reshape(-1, 1)

        for epoch in range(self.epochs):
            # 2. Forward Pass: Z = XW + b
            # X(N, M) dot W(M, 1) -> Z(N, 1)
            Z = np.dot(X, self.W) + self.b
            y_hat = self._sigmoid(Z)

            # 3. Cost Calculation (Binary Cross-Entropy Loss)
            # เพิ่ม epsilon เล็กน้อยเพื่อป้องกัน Math Domain Error จาก log(0)
            epsilon = 1e-15
            cost = -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
            self.loss_history.append(cost)

            # 4. Backward Pass (คำนวณ Gradients)
            # Error Vector: dZ รูปร่าง (N, 1)
            dZ = y_hat - y
            
            # dW = (1/N) * X^T dot dZ
            # X.T(M, N) dot dZ(N, 1) -> dW(M, 1)
            dW = (1 / N) * np.dot(X.T, dZ)
            
            # db = ค่าเฉลี่ยของ Error ทั้งหมด (Scalar)
            db = (1 / N) * np.sum(dZ)

            # 5. Weight Update (ปรับปรุงน้ำหนัก)
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """คืนค่าความน่าจะเป็น (Probability) รูปร่าง (N, 1)"""
        Z = np.dot(X, self.W) + self.b
        return self._sigmoid(Z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """คืนค่า Class 0 หรือ 1 ตาม Threshold"""
        return (self.predict_proba(X) >= threshold).astype(int)