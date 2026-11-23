"""Inferencia de PneumoVision: no entrena modelos ni guarda radiografías."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

MODEL_ID = "densenet121-res224-all"
ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "head_weights.json"

def validate_head_file(path=WEIGHTS_PATH):
    with Path(path).open("r", encoding="utf-8") as f:
        head = json.load(f)
    if head.get("base_model") != MODEL_ID:
        raise ValueError("El modelo base no coincide con el entrenamiento")
    if head.get("num_classes") != 2 or head.get("labels") != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Clases incompatibles")
    if head.get("source_selection") != "cabeza_nueva":
        raise ValueError("Se requiere el checkpoint ganador de cabeza nueva")
    w = np.asarray(head["weight"], dtype=np.float32)
    b = np.asarray(head["bias"], dtype=np.float32)
    if w.shape != (2, 1024) or b.shape != (2,) or not (np.isfinite(w).all() and np.isfinite(b).all()):
        raise ValueError("Los pesos o sesgos tienen dimensiones/valores inválidos")
    return head

def load_classifier(path=WEIGHTS_PATH):
    import torch
    from torch import nn
    import torchxrayvision as xrv
    h = validate_head_file(path)
    net = xrv.models.DenseNet(weights=h["base_model"])
    assert net.classifier.in_features == 1024, "Extractor inesperado"
    layer = nn.Linear(1024, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(h["weight"], dtype=torch.float32))
        layer.bias.copy_(torch.as_tensor(h["bias"], dtype=torch.float32))
    net.classifier = layer
    net.op_threshs = None
    net.apply_sigmoid = False
    net.eval()
    return net.to("cpu")

def preprocess_image(image: Image.Image):
    import torch
    import torchvision.transforms as T
    import torchxrayvision as xrv
    pixels = np.asarray(image.convert("L"), dtype=np.float32)
    pixels = xrv.datasets.normalize(pixels, 255)
    pixels = pixels[None, :, :]
    pipeline = T.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    pixels = pipeline(pixels)
    return torch.from_numpy(np.asarray(pixels, dtype=np.float32)).unsqueeze(0)

def predict(model, image: Image.Image):
    import torch
    with torch.inference_mode():
        out = model(preprocess_image(image))
        if tuple(out.shape) != (1, 2):
            raise RuntimeError(f"Salida inesperada: {tuple(out.shape)}")
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return float(probs[0]), float(probs[1])
