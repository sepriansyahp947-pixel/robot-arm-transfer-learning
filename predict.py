import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# ==============================
# KONFIGURASI
# ==============================

MODEL_PATH = "model/robot_arm_mobilenetv2.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "blue",
    "green",
    "red"
]


# ==============================
# CEK ARGUMENT
# ==============================

if len(sys.argv) < 2:
    print("Cara penggunaan:")
    print("python predict.py nama_gambar.jpg")
    sys.exit()


IMAGE_PATH = sys.argv[1]


# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat.")


# ==============================
# LOAD IMAGE
# ==============================

image = load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

image_array = img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ==============================
# PREDICTION
# ==============================

prediction = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(prediction[0])

predicted_class = CLASS_NAMES[predicted_index]

confidence = prediction[0][predicted_index] * 100


# ==============================
# OUTPUT
# ==============================

print("\n==============================")
print("ROBOT ARM COLOR CLASSIFICATION")
print("==============================")

print(f"Predicted Class : {predicted_class}")
print(f"Confidence      : {confidence:.2f}%")

print("\nProbability:")

for class_name, probability in zip(
    CLASS_NAMES,
    prediction[0]
):
    print(
        f"{class_name:>6}: "
        f"{probability * 100:.2f}%"
    )