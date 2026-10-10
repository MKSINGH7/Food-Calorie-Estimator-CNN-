"""Train the food classifier.

    python -m src.train                       # MobileNetV2 transfer learning (recommended)
    python -m src.train --arch cnn            # small from-scratch CNN
    python -m src.train --data-dir dataset_dummy --no-pretrained --epochs 2 --finetune-epochs 1

Outputs (in models/ and outputs/):
    models/food_classifier.keras   trained model
    models/model_meta.json         class names, image size, architecture
    outputs/accuracy.png, loss.png, confusion_matrix.png
    outputs/classification_report.txt, metrics.json
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from . import config
from .data import load_datasets
from .model import build_model, compile_model, unfreeze_top


def _fit_phase(model, train, val, epochs, ckpt_path, patience):
    """Run one training phase and load the best (lowest val_loss) weights afterwards."""
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]
    hist = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)
    if ckpt_path.exists():
        model.load_weights(str(ckpt_path))
    return hist.history


def _merge(*histories):
    out = {}
    for h in histories:
        for k, v in h.items():
            out.setdefault(k, []).extend(v)
    return out


def _plot_history(hist, out_dir):
    for key, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        plt.figure()
        plt.plot(hist[key], label="Train")
        plt.plot(hist["val_" + key], label="Validation")
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(title)
        plt.legend()
        plt.savefig(out_dir / f"{key}.png", dpi=120, bbox_inches="tight")
        plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--arch", default="mobilenetv2", choices=["mobilenetv2", "cnn"])
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=8, help="phase 1: train head (or whole CNN)")
    p.add_argument("--finetune-epochs", type=int, default=8, help="phase 2: fine-tune backbone")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--no-pretrained", action="store_true", help="do not download ImageNet weights")
    p.add_argument("--patience", type=int, default=4)
    args = p.parse_args()

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)

    train, val, class_names = load_datasets(args.data_dir, args.img_size, args.batch_size)
    print(f"Classes ({len(class_names)}): {class_names}")

    model, backbone = build_model(
        len(class_names), args.arch, args.img_size, pretrained=not args.no_pretrained
    )
    compile_model(model, args.lr)
    model.summary()

    ckpt = config.MODELS_DIR / "_best.weights.h5"
    if ckpt.exists():
        ckpt.unlink()

    print("\n=== Phase 1: train classifier head ===")
    h1 = _fit_phase(model, train, val, args.epochs, ckpt, args.patience)

    h2 = {}
    if backbone is not None and args.finetune_epochs > 0:
        print("\n=== Phase 2: fine-tune top of the backbone ===")
        unfreeze_top(backbone)
        compile_model(model, args.finetune_lr)  # must re-compile after changing trainable
        if ckpt.exists():
            ckpt.unlink()
        h2 = _fit_phase(model, train, val, args.finetune_epochs, ckpt, args.patience)
    if ckpt.exists():
        ckpt.unlink()

    # ---- save model + metadata ----------------------------------------------
    model.save(config.MODEL_PATH)
    meta = {"class_names": class_names, "img_size": args.img_size, "arch": args.arch}
    config.META_PATH.write_text(json.dumps(meta, indent=2))

    # ---- evaluation ----------------------------------------------------------
    hist = _merge(h1, h2)
    _plot_history(hist, config.OUTPUTS_DIR)

    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in val])
    y_pred = np.argmax(model.predict(val, verbose=0), axis=1)
    report = classification_report(
        y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0
    )
    (config.OUTPUTS_DIR / "classification_report.txt").write_text(report)
    print("\n" + report)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 9))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    plt.yticks(range(len(class_names)), class_names, fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (validation)")
    plt.colorbar()
    plt.savefig(config.OUTPUTS_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    metrics = {
        "train_accuracy": float(hist["accuracy"][-1]),
        "val_accuracy": float(hist["val_accuracy"][-1]),
        "best_val_accuracy": float(max(hist["val_accuracy"])),
        "epochs_run": len(hist["accuracy"]),
    }
    (config.OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Metrics:", metrics)
    print(f"Saved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
