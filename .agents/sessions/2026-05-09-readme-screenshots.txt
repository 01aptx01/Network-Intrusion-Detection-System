Checkpoint 2026-05-09
- อัปเดต README ให้ตรงกับโครงสร้าง repo และ pipeline จริง
- เพิ่ม docs/images/evaluation_counts.png, pipeline_overview.png (สคริปต์ scripts/generate_readme_images.py)
- config.py: path ข้อมูลเป็น .txt ใต้ data/raw/
- เพิ่ม requirements.txt; .gitignore: logs, saved_models, __pycache__
- รัน main.py ยืนยันตัวเลข evaluation ในตาราง README (TP/TN/FP/FN, P/R/F1)
