# Robot Arm Education Training Kit - Transfer Learning

## Deskripsi Proyek

Proyek ini merupakan implementasi metode Transfer Learning untuk melakukan klasifikasi warna objek pada Robot Arm Education Training Kit.

Model Computer Vision digunakan untuk mengenali tiga kelas warna:

- Blue
- Green
- Red

Hasil klasifikasi dapat digunakan sebagai dasar sistem sorting pada robot arm.

## Tujuan

1. Mengumpulkan dataset gambar berdasarkan warna.
2. Melakukan preprocessing dataset.
3. Membangun model klasifikasi menggunakan Transfer Learning.
4. Menggunakan MobileNetV2 sebagai pretrained model.
5. Mengevaluasi performa model menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.
6. Menyiapkan model untuk integrasi dengan sistem robot arm.

## Metode

Metode yang digunakan adalah Transfer Learning dengan arsitektur MobileNetV2 yang telah pretrained menggunakan ImageNet.

Alur sistem:

Camera → Image Processing → MobileNetV2 → Color Classification → Robot Arm Sorting

## Dataset

Dataset terdiri dari tiga kelas:

| Class | Training | Validation | Total |
|---|---:|---:|---:|
| Blue | 155 | 39 | 194 |
| Green | 149 | 38 | 187 |
| Red | 148 | 38 | 186 |
| **Total** | **452** | **115** | **567** |

Dataset digunakan sebagai dataset awal untuk eksperimen klasifikasi warna.

## Training Configuration

- Image size: 224 × 224
- Batch size: 32
- Epoch: 15
- Model: MobileNetV2
- Pretrained weights: ImageNet
- Optimizer: Adam
- Number of classes: 3

## Hasil

Model menghasilkan validation accuracy sebesar:

**85.22%**

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Blue | 76.60% | 92.31% | 83.72% |
| Green | 84.38% | 71.05% | 77.14% |
| Red | 97.22% | 92.11% | 94.59% |

### Confusion Matrix

| Actual / Predicted | Blue | Green | Red |
|---|---:|---:|---:|
| Blue | 36 | 2 | 1 |
| Green | 11 | 27 | 0 |
| Red | 0 | 3 | 35 |

Kesalahan klasifikasi terbesar terjadi pada objek Green yang diprediksi sebagai Blue sebanyak 11 gambar.

## Struktur Project

```text
robot-arm-transfer-learning/
│
├── dataset/
│   ├── train/
│   │   ├── blue/
│   │   ├── green/
│   │   └── red/
│   │
│   └── validation/
│       ├── blue/
│       ├── green/
│       └── red/
│
├── model/
│   └── robot_arm_mobilenetv2.keras
│
├── results/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── Train.py
├── Evaluate.py
├── predict.py
├── robot_arm_control.ino
├── requirements.txt
├── .gitignore
└── README.md