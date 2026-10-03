import os
import tensorflow as tf
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

MODEL_DIR = "model"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\n================================")
print("CLASS YANG TERDETEKSI:")
print(class_names)
print("================================\n")


# ==========================================
# 3. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])


# ==========================================
# 4. LOAD MOBILENETV2
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze MobileNetV2
base_model.trainable = False


# ==========================================
# 5. BUILD MODEL
# ==========================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# ==========================================
# 6. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 7. MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# 8. TRAINING
# ==========================================

print("\n================================")
print("MULAI TRAINING...")
print("================================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# ==========================================
# 9. SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "robot_arm_mobilenetv2.keras"
)

model.save(model_path)

print("\n================================")
print("MODEL BERHASIL DISIMPAN")
print(model_path)
print("================================")


# ==========================================
# 10. ACCURACY GRAPH
# ==========================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "accuracy.png"
    )
)

plt.show()


# ==========================================
# 11. LOSS GRAPH
# ==========================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "loss.png"
    )
)

plt.show()


# ==========================================
# 12. FINISH
# ==========================================

print("\n================================")
print("TRAINING SELESAI!")
print("================================")