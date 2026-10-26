"""Train MobileNetV2 to classify soil image folders.

Expected data/soil_images/<class name>/<image files> folders.
Run: python src/train_soil_cnn.py
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "soil_images"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 24
SEED = 42
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images():
    class_names = sorted(folder.name for folder in IMAGE_DIR.iterdir() if folder.is_dir())
    if len(class_names) < 2:
        raise FileNotFoundError(
            f"Expected class folders inside {IMAGE_DIR}. See README.md for the seven source classes."
        )
    paths, labels = [], []
    seen_hashes = set()
    cache_root = ROOT / "data" / "processed_soil_images"
    for label, class_name in enumerate(class_names):
        folder = IMAGE_DIR / class_name
        for path in folder.rglob("*"):
            if path.suffix.lower() not in SUPPORTED:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    print(f"Skipping exact duplicate image: {path}")
                    continue
                seen_hashes.add(digest)
                # TensorFlow's decoder does not read every source format
                # (for example, WebP), so make a standard RGB JPEG cache.
                cached_path = cache_root / class_name / f"{digest[:16]}.jpg"
                cached_path.parent.mkdir(parents=True, exist_ok=True)
                if not cached_path.exists():
                    with Image.open(path) as image:
                        image.convert("RGB").save(cached_path, format="JPEG", quality=95)
                paths.append(str(cached_path))
                labels.append(label)
            except (UnidentifiedImageError, OSError):
                print(f"Skipping unreadable image: {path}")
    if not paths:
        raise ValueError("No readable images found.")
    return np.asarray(paths), np.asarray(labels), class_names


def main():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    paths, labels, class_names = collect_images()
    print("Images per class:")
    for label, name in enumerate(class_names):
        print(f"  {name}: {(labels == label).sum()}")
    train_paths, remaining_paths, train_y, remaining_y = train_test_split(
        paths, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    val_paths, test_paths, val_y, test_y = train_test_split(
        remaining_paths, remaining_y, test_size=0.5, random_state=SEED,
        stratify=remaining_y
    )

    def decode(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    def make_dataset(image_paths, image_labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
        if training:
            ds = ds.shuffle(len(image_paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_paths, train_y, training=True)
    val_ds = make_dataset(val_paths, val_y)
    test_ds = make_dataset(test_paths, test_y)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.08),
    ], name="training_augmentation")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="soil_image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(ROOT / "models" / "soil_cnn.keras"),
                                           monitor="val_loss", save_best_only=True),
    ]
    weights = compute_class_weight(class_weight="balanced",
                                   classes=np.unique(train_y), y=train_y)
    class_weights = {int(label): float(weight)
                     for label, weight in zip(np.unique(train_y), weights)}
    print("Training MobileNetV2 classifier (base is frozen)...")
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks,
              class_weight=class_weights, shuffle=False)
    model = tf.keras.models.load_model(ROOT / "models" / "soil_cnn.keras")
    probabilities = model.predict(test_ds, verbose=0)
    prediction = probabilities.argmax(axis=1)
    accuracy = accuracy_score(test_y, prediction)
    macro_f1 = f1_score(test_y, prediction, average="macro", zero_division=0)
    report = classification_report(test_y, prediction, labels=np.arange(len(class_names)),
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    cm = confusion_matrix(test_y, prediction, labels=np.arange(len(class_names)))
    evaluation_dir = ROOT / "artifacts" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "soil_class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    (evaluation_dir / "soil_cnn_metrics.json").write_text(json.dumps({
        "model": "MobileNetV2 transfer learning",
        "image_size": list(IMAGE_SIZE),
        "split_sizes": {"train": len(train_paths), "validation": len(val_paths),
                        "test": len(test_paths)},
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_classification_report": report,
        "class_names": class_names,
        "augmentation": "Random flip, rotation, zoom applied in the model during training only",
    }, indent=2), encoding="utf-8")
    np.savetxt(evaluation_dir / "soil_cnn_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Soil CNN held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(evaluation_dir / "soil_cnn_confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"Held-out test accuracy: {accuracy:.4f}; Macro F1: {macro_f1:.4f}")
    print("Saved models/soil_cnn.keras, artifacts/soil_class_names.json, and evaluation files.")


if __name__ == "__main__":
    main()
