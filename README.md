# 🛡️ Anomaly-Based Network Intrusion Detection System (From Scratch)

## 📌 Project Overview
โปรเจกต์นี้มีวัตถุประสงค์เพื่อสร้างระบบตรวจจับการบุกรุกในเครือข่าย (NIDS) เพื่อจำแนกประเภท Traffic ระหว่างพฤติกรรมปกติ (Normal) และการโจมตี (Attack) โดยพัฒนาระบบคณิตศาสตร์และ Machine Learning Algorithm ขึ้นมาเองจากศูนย์ (From Scratch) ด้วย Object-Oriented Programming (OOP) 

เป้าหมายสูงสุดคือการสร้างสมการคำนวณผ่าน Matrix Operations เพื่อความรวดเร็ว และปรับแต่ง Loss Function เพื่อลดค่า **False Negative Rate** ให้เหลือน้อยที่สุดสำหรับชุดข้อมูลที่มีความไม่สมดุลสูง (Highly Imbalanced Data)

---

## 📊 Dataset
ใช้ชุดข้อมูล **NSL-KDD** ซึ่งเป็น Benchmark ที่ได้รับการยอมรับในงานวิจัยด้าน Cybersecurity
* **Features:** 41 features (Basic, Content, and Traffic features)
* **Target:** Binary Classification (`0` = Normal, `1` = Attack)

---

## 🛠 Tech Stack & Strict Constraints
โปรเจกต์นี้อยู่ภายใต้ข้อจำกัดทางวิศวกรรมขั้นสูงสุด:
* **Language:** Python 3.x
* **Permitted Libraries:** `NumPy` (สำหรับการประมวลผล Matrix/Linear Algebra), `Pandas` (สำหรับการโหลดข้อมูลเบื้องต้นเท่านั้น)
* **🚫 Forbidden Libraries:** ห้ามใช้ `Scikit-learn`, `XGBoost`, `TensorFlow`, `PyTorch` โดยเด็ดขาด
* **Core Algorithm:** Custom Logistic Regression (NumPy Implementation) พร้อมระบบจัดการ Class Weights ใน Loss Function

---

## 🏗 Project Structure

```
├── config.py                 # เก็บ Hyperparameters (LR, Epochs) และ File Paths
├── data/
│   ├── raw/                  # เก็บข้อมูลดิบ NSL-KDD (Read-only)
│   └── processed/            # เก็บข้อมูลที่ผ่าน Z-score Standardization แล้ว
├── notebooks/                # สำหรับ EDA พื้นฐาน
├── saved_models/             # เก็บไฟล์ .npy ของ Weight Matrix และ Bias
├── src/
│   ├── preprocessing.py      # Data Loading, Z-score, Class Imbalance Calculation
│   ├── model.py              # Logistic Regression class (NumPy only, Gradient Descent)
│   ├── evaluator.py          # Custom Confusion Matrix, Precision, Recall, F1-Score
│   └── utils.py              # ระบบ Logging 
├── main.py                   # Orchestrator สั่งรัน Pipeline ทั้งหมด
├── requirements.txt          # รายการ Dependencies (numpy, pandas)
├── .gitignore 
└── README.md                 # เอกสารอธิบายโปรเจกต์
```

## 📝 Commit Message Guide

ใช้หลักการ **Conventional Commits** เพื่อความเป็นระเบียบและเป็นมาตรฐานสากล

**Format:** `type(scope): subject`

**หัวข้อ (Type) ที่ควรใช้:**

-   **feat:** เพิ่มฟีเจอร์ใหม่ (New Feature)
    
-   **fix:** แก้บั๊ก (Bug Fix)
    
-   **docs:** แก้ไขเอกสาร เช่น README (Documentation)
    
-   **style:** จัด format โค้ด, เติม semicolon (ไม่กระทบ logic)
    
-   **refactor:** รื้อโค้ด เขียนใหม่ให้ดีขึ้น แต่ผลลัพธ์เหมือนเดิม
    
-   **chore:** งานจุกจิก เช่น อัปเดต version, แก้ .gitignore