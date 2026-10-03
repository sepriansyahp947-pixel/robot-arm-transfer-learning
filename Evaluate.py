import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ==============================
# KONFIGURASI
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

VAL_DIR = "dataset/validation"
MODEL_PATH = "model/robot_arm_mobilenetv2.keras"
RESULT_DIR = "results"

CLASS_NAMES = ["blue", "green", "red"]

os.makedirs(RESULT_DIR, exist_ok=True)


# ==============================
# LOAD DATA VALIDATION
# ==============================

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    shuffle=False
)

print("\nClass names:", CLASS_NAMES)
print("Jumlah data validation:", sum(1 for _ in val_ds))


# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel berhasil dimuat:")
print(MODEL_PATH)


# ==============================
# PREDIKSI
# ==============================

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==============================
# CONFUSION MATRIX
# ==============================

cm = tf.math.confusion_matrix(
    y_true,
    y_pred,
    num_classes=len(CLASS_NAMES)
).numpy()

print("\nConfusion Matrix:")
print(cm)


# ==============================
# GAMBAR CONFUSION MATRIX
# ==============================

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest")

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.close()


# ==============================
# CLASSIFICATION REPORT
# ==============================

report_lines = []

report_lines.append("CLASSIFICATION REPORT")
report_lines.append("=" * 50)
report_lines.append("")

total_correct = np.trace(cm)
total_samples = np.sum(cm)

accuracy = total_correct / total_samples

report_lines.append(
    f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)"
)

report_lines.append("")

for i, class_name in enumerate(CLASS_NAMES):

    TP = cm[i, i]
    FP = np.sum(cm[:, i]) - TP
    FN = np.sum(cm[i, :]) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    report_lines.append(f"Class: {class_name}")
    report_lines.append(f"Precision: {precision:.4f}")
    report_lines.append(f"Recall:    {recall:.4f}")
    report_lines.append(f"F1-Score:  {f1:.4f}")
    report_lines.append("")


# ==============================
# SIMPAN REPORT
# ==============================

report_path = os.path.join(
    RESULT_DIR,
    "classification_report.txt"
)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))


# ==============================
# OUTPUT
# ==============================

print("\n" + "=" * 50)
print("EVALUASI SELESAI!")
print("=" * 50)

print("\nOverall Accuracy:")
print(f"{accuracy*100:.2f}%")

print("\nFile hasil:")
print("results/confusion_matrix.png")
print("results/classification_report.txt")