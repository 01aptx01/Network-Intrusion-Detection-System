# Network Intrusion Detection System - Architecture Guide

เอกสารฉบับนี้อธิบายภาพรวมของ System Architecture และอธิบาย Workflow ของ Pipeline ในโปรเจกต์ Network Intrusion Detection System (NIDS) อย่างละเอียดตั้งแต่ต้นจนจบ

## 1. System Overview (ภาพรวมของระบบ)

โปรเจกต์นี้ถูกออกแบบโครงสร้างตามมาตรฐาน **Cookiecutter Data Science** โดยมีเป้าหมายหลักในการนำข้อมูล Network Traffic (NSL-KDD dataset) มาผ่านกระบวนการ Data Preprocessing (เช่น การทำ Standardize ข้อมูลที่เป็นตัวเลข และทำ One-Hot Encoding สำหรับข้อมูลประเภท Categorical) จากนั้นนำไปเทรนโมเดลเพื่อทำ Binary Classification (แยกระหว่าง ทราฟฟิกปกติ กับ การโจมตี) โดยใช้ไลบรารี Scikit-Learn

## 2. Directory Structure (โครงสร้างไฟล์ใน `src/`)

Source code หลักของระบบจะอยู่ในโฟลเดอร์ `src/` โดยแบ่งตามหน้าที่การทำงาน (Logical Function) ดังนี้:

*   **`src/config.py`**: ไฟล์ Configuration หลัก ใช้สำหรับเก็บ File Paths (เช่น path ของ dataset และ path สำหรับเซฟโมเดล) รวมถึง Hyperparameters ต่างๆ (เช่น Learning Rate, Decision Threshold และ Model Configurations อย่างเช่น ค่า `C` สำหรับ Logistic Regression หรือ `n_estimators` สำหรับ Random Forest)
*   **`src/data/preprocessing.py`**: จัดการเรื่องการโหลด Data และตั้งค่า `scikit-learn` `ColumnTransformer` (ซึ่งจะ apply `StandardScaler` กับ Numerical Features และ `OneHotEncoder` กับ Categorical Features) นอกจากนี้ยังมีฟังก์ชัน Fallback สำหรับสร้าง Synthetic Data (ข้อมูลจำลอง) เผื่อในกรณีที่ไม่พบ NSL-KDD dataset
*   **`src/features/build_features.py`**: มี Helper Functions สำหรับเชื่อมโยงตรรกะการทำ Preprocessing เข้ากับ Training Pipeline
*   **`src/models/`**: ประกอบด้วยลอจิกสำหรับการกำหนด (Define), เทรน (Train), เซฟ (Save) และประเมินผล (Evaluate) โมเดล
    *   **`model.py`**: กำหนดคลาส `ModelFactory` เพื่อสร้าง Scikit-Learn model ที่ถูกต้องตามคอนฟิกใน `config.py`
    *   **`train_model.py`**: Training Script หลักที่ร้อยเรียงทุกอย่างเข้าด้วยกัน (ตั้งแต่ Data Loading -> Preprocessing -> Training -> Evaluation)
    *   **`predict_model.py`**: ฟังก์ชันสำหรับการทำ Prediction โดยใช้ Model Pipeline ที่เทรนและเซฟไว้แล้ว
    *   **`metrics.py`**: Custom Classes สำหรับคำนวณและบันทึก Classification Metrics (เช่น Precision, Recall, F1-score, Confusion Matrix)
    *   **`artifacts.py`**: Helper Functions สำหรับการ Save/Load โมเดลที่เทรนแล้ว (ในรูปแบบไฟล์ `.joblib`) และตั้งค่าระบบ Logging
*   **`src/visualization/visualize.py`**: ลอจิกสำหรับสร้าง Plots ต่างๆ เช่น Feature Importance Chart จากโมเดลที่เทรนเสร็จแล้ว

---

## 3. End-to-End Pipeline Workflow (อธิบาย Pipeline ตั้งแต่ต้นจนจบ)

เมื่อคุณรันคำสั่ง `python main.py` ระบบจะทำงานตามลำดับขั้นตอนดังต่อไปนี้อย่างละเอียด:

### Step 1: Entry Point & Initialization (จุดเริ่มต้นและการตั้งค่า)
*   Script จะเริ่มทำงานที่ไฟล์ **`main.py`** โดยทำการเรียกฟังก์ชัน `train_pipeline()` จากโมดูล `src.models.train_model`
*   `train_pipeline` จะทำการ Initialize ตัว Logger เพื่อเก็บ Logs ของระบบลงในโฟลเดอร์ `logs/`
*   จากนั้นระบบจะอ่านค่า Settings และ Hyperparameters ทั้งหมดจาก **`src/config.py`** เพื่อเตรียมพร้อมสำหรับขั้นตอนต่อไป

### Step 2: Data Loading (การโหลดข้อมูล)
*   ระบบจะไปที่ฟังก์ชันใน **`src.data.preprocessing`** เพื่อพยายามโหลดไฟล์ Dataset จริง (`KDDTrain+.txt` และ `KDDTest+.txt`) จาก path ที่กำหนดไว้
*   *Fallback Mechanism:* หากระบบหาไฟล์ข้อมูลไม่พบ ระบบจะทำการ Generate **Synthetic Dataset** (ข้อมูลจำลอง) ขึ้นมาแทนโดยอัตโนมัติ เพื่อให้ Pipeline สามารถรันต่อไปได้โดยไม่ Crash ระหว่างการทดสอบระบบ (Testing/Debugging)

### Step 3: Data Preprocessing Setup (การเตรียมการแปลงข้อมูล)
*   สร้าง **`ColumnTransformer`** (จาก `scikit-learn`) เพื่อเตรียมทำ Feature Engineering
    *   **Numerical Data**: จะถูกส่งไปทำ **`StandardScaler`** เพื่อปรับค่าข้อมูลตัวเลขให้อยู่ในสเกลเดียวกัน (Mean=0, Variance=1)
    *   **Categorical Data**: จะถูกส่งไปทำ **`OneHotEncoder`** เพื่อแปลงค่าที่เป็นหมวดหมู่ให้กลายเป็น Binary Vectors

### Step 4: Pipeline Construction (การประกอบร่าง Pipeline)
*   ระบบจะทำการ Instantiate ตัวโมเดล Machine Learning โดยใช้ **`ModelFactory`** (เช่น เลือกว่าจะใช้ Logistic Regression หรือ Random Forest ตามที่คอนฟิกไว้)
*   จากนั้น ระบบจะนำ **Preprocessor** (จาก Step 3) และ **Model** มารวมเข้าด้วยกันเป็น Object เดียว เรียกว่า **`scikit-learn Pipeline`** 
*   *ข้อดีของการทำแบบนี้คือ ข้อมูลใหม่ๆ ที่เข้ามาจะถูกบังคับให้ผ่าน Preprocessor ก่อนเข้าไปที่ Model เสมอ ป้องกันปัญหา Data Leakage และทำให้โค้ดเป็นระเบียบ*

### Step 5: Training (การเทรนโมเดล)
*   ระบบจะเรียกคำสั่ง **`pipeline.fit(features_train, labels_train)`**
*   ในขั้นตอนนี้ ข้อมูล Training Data จะไหลผ่าน Preprocessor ก่อน (โดนสเกลและแปลงค่า) จากนั้นข้อมูลที่ถูกแปลงแล้วจะถูกส่งเข้าไปเทรนใน Classifier Model จนเสร็จสมบูรณ์

### Step 6: Artifact Saving (การบันทึกโมเดลเก็บไว้)
*   เมื่อเทรนเสร็จ ฟังก์ชันใน **`src.models.artifacts`** จะทำการ Serialize (แปลงเป็นไบต์) ตัว Pipeline แบบสมบูรณ์ (ซึ่งรวมทั้ง Preprocessor ที่ฟิตแล้ว และ Model ที่เทรนแล้ว)
*   เซฟลงเป็นไฟล์นามสกุล `.joblib` เก็บไว้ที่ `models/nids_pipeline.joblib` สำหรับนำไปใช้งานต่อ (Inference) ในอนาคตโดยไม่ต้องเทรนใหม่

### Step 7: Evaluation (การประเมินประสิทธิภาพ)
*   ระบบนำ Test Set มาทำ Model Inference ผ่านคำสั่ง `pipeline.predict()`
*   นำค่า Predictions ที่ได้ ไปเทียบกับ True Labels ด้วยฟังก์ชันใน **`src.models.metrics`**
*   ระบบจะสร้างและบันทึก Metrics ต่างๆ เช่น True Positives, False Positives, Precision, Recall, F1-score และสร้าง Confusion Matrix ออกมาให้เห็นประสิทธิภาพการทำงานของโมเดล

### Step 8: Visualization (การสร้างกราฟแสดงผล)
*   ในขั้นตอนสุดท้าย ลอจิกใน **`src.visualization.visualize`** จะดึงค่า Feature Importance ออกมาจากโมเดล (ถ้าโมเดลนั้นซัพพอร์ต เช่น Random Forest)
*   นำค่าเหล่านี้มาพล็อตเป็นกราฟแท่ง (Bar Chart) แล้วบันทึกไฟล์รูปภาพ (Output) ออกไปเก็บไว้ที่โฟลเดอร์ `docs/images/`

ด้วยสถาปัตยกรรมแบบ Pipeline นี้ ทำให้มั่นใจได้ว่า Data Transformation Steps ที่ถูก Apply ในช่วง Training จะถูก Apply แบบเป๊ะๆ ในช่วง Inference หรือตอนนำไป Deploy ใช้งานจริง (Production) ช่วยป้องกัน Bugs ที่เกิดจากการจัดการ Data ที่ไม่สม่ำเสมอได้อย่างมีประสิทธิภาพ
