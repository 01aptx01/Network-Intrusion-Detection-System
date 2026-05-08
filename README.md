# Network Intrusion Detection System (NIDS)

โปรเจกต์นี้เป็น **ระบบจำแนกทราฟฟิกเครือข่ายแบบจุดเดียว (binary)** ว่าเป็นพฤติกรรมปกติ (`normal`) หรือการโจมตี (`attack`) โดยใช้ชุดข้อมูล **NSL-KDD** เป็นฐาน และใช้ **Logistic Regression ที่เขียนเองด้วย NumPy** (mini-batch gradient descent, class weight ใน loss) โดยไม่พึ่ง Scikit-learn / PyTorch / TensorFlow

สิ่งที่รันได้จริงใน repo นี้:

1. โหลด train/test จากไฟล์ดิบ NSL-KDD (`.txt` คอมมาแยกฟิลด์)
2. แปลง categorical เป็น one-hot (`pandas.get_dummies`) แล้ว Z-score จากสถิติชุด train
3. คำนวณ class weights จากความไม่สมดุลของคลาส แล้วฝึกโมเดล
4. บันทึกน้ำหนัก (`saved_models/nids_weights.npz`) และสถานะ preprocessor (`.pkl`)
5. ประเมินด้วย confusion components + precision / recall / F1 และรายงาน feature importance จากค่า weight

---

## ภาพประกอบ (ภาพรวม)

| Pipeline (main.py) | ตัวอย่างผล evaluation (จำนวนตัวอย่างต่อช่อง) |
| :---: | :---: |
| ![Pipeline overview](docs/images/pipeline_overview.png) | ![Evaluation counts](docs/images/evaluation_counts.png) |

แท่งสีแดง (**FN** — missed attacks) ช่วยให้เห็นข้อจำกัดของ threshold / โมเดลเชิงเส้นในข้อมูลจริงได้ทันที

---

## ตัวเลขอ้างอิงจากการรันจริง (เครื่องพัฒนา)

รันครั้งล่าสุดที่ใช้ยืนยัน README: `python main.py` บนชุด **KDDTrain+.txt** / **KDDTest+.txt** ใน `data/raw/` พร้อมค่าใน `config.py` ณ ขณะนั้น (`LEARNING_RATE=0.05`, `EPOCHS=50`, `BATCH_SIZE=128`, `DECISION_THRESHOLD=0.4`)

| Metric | Value |
|--------|------:|
| TP (caught attacks) | 8088 |
| TN (normal) | 8943 |
| FP (false alarms) | 768 |
| FN (missed attacks) | 4745 |
| Precision | 0.9133 |
| Recall | 0.6303 |
| F1-score | 0.7458 |

ค่าจะเปลี่ยนตาม threshold, epoch, learning rate และการสุ่มลำดับ mini-batch

---

## Tech stack และข้อจำกัด

| รายการ | รายละเอียด |
|--------|------------|
| ภาษา | Python 3.x |
| ไลบรารีหลัก | `numpy`, `pandas` (pandas ใช้หนักที่ขั้นโหลด/ one-hot) |
| ห้ามใช้ (ตามสเปกโปรเจกต์) | `scikit-learn`, `xgboost`, `tensorflow`, `pytorch` |
| โมเดล | `BinaryLogisticRegression` ใน `src/model.py` |

สคริปต์ `scripts/generate_readme_images.py` ใช้ **matplotlib** เฉพาะตอนสร้างภาพใน `docs/images/` ไม่ได้เป็นส่วนของ pipeline ฝึกโมเดล

---

## ความต้องการของระบบและการติดตั้ง

```text
pip install -r requirements.txt
```

ไฟล์ดิบ NSL-KDD วางที่ `data/raw/KDDTrain+.txt` และ `data/raw/KDDTest+.txt` (ปรับ path ได้ที่ `config.py`)

รัน pipeline ทั้งก้อน:

```text
python main.py
```

- Log ไฟล์อยู่ในโฟลเดอร์ `logs/`
- น้ำหนักและ preprocessor อยู่ใน `saved_models/`

สร้างภาพ README ใหม่ (ต้องติดตั้ง matplotlib แยก):

```text
pip install matplotlib
python scripts/generate_readme_images.py
```

บน Windows ถ้าเทอร์มินัลยังแสดง emoji ไม่ครบ ให้ตั้ง `PYTHONUTF8=1` หรือใช้เทอร์มินัลที่รองรับ UTF-8; โค้ดจะพยายาม `reconfigure` stdout/stderr เป็น UTF-8 เมื่อเริ่ม `main.py`

---

## โครงสร้างโปรเจกต์ (ปัจจุบัน)

```text
├── config.py              # path ข้อมูล, hyperparameters, threshold
├── main.py                # orchestrator: โหลด → scale → train → save → evaluate
├── requirements.txt       # numpy, pandas
├── scripts/
│   ├── generate_readme_images.py
│   └── build_all_in_one_notebook.py
├── notebooks/
│   └── NIDS_All_In_One.ipynb   # สร้าง/ซิงก์จาก src ด้วย scripts/build_all_in_one_notebook.py
├── docs/
│   └── images/            # ภาพประกอบ README
├── data/
│   └── raw/               # NSL-KDD (.txt); มีสำเนาใต้ data/raw/nsl-kdd/ ด้วย
├── src/
│   ├── preprocessing.py   # IntrusionDatasetPreprocessor
│   ├── model.py           # BinaryLogisticRegression (NumPy)
│   ├── metrics.py         # BinaryClassifierMetrics
│   └── artifacts.py       # ArtifactStore — log, save/load weights & preprocessor
├── logs/                  # log รันล่าสุด (สร้างเมื่อรัน)
└── saved_models/          # .npz weights + preprocessor .pkl (สร้างเมื่อรัน)
```

---

## Commit messages

ใช้แนว **Conventional Commits**: `type(scope): subject` เช่น `feat(model): ...`, `docs(readme): ...`, `fix(metrics): ...`

---

## แหล่งข้อมูล

- NSL-KDD เป็น benchmark ที่ใช้กันแพร่หลายในงานวิจัยด้าน intrusion detection; รายละเอียดฟิลด์และความหมายของค่า label ดูจากเอกสารชุดข้อมูลต้นทาง
