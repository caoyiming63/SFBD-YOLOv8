from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent

    model = YOLO(str(ROOT / "SFBD-YOLOv8.yaml"))

    model.train(
        data=str(ROOT / "Laboro_Tomato.yaml"),
        workers=0,
        epochs=300,
        batch=8,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        seed=0,
        deterministic=True,
        amp=True,
        project=str(ROOT / "runs" / "Laboro"),
        name="SFBD-YOLOv8",
    )