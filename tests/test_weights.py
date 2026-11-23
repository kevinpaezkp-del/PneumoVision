"""Pruebas de integridad del clasificador que se publica."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_utils import MODEL_ID, WEIGHTS_PATH, validate_head_file

def test_modelo_y_etiquetas():
    data=validate_head_file()
    assert data["base_model"]==MODEL_ID
    assert data["labels"]==["NORMAL","PNEUMONIA"]
    assert data["source_selection"]=="cabeza_nueva"

def test_numero_de_parametros():
    with WEIGHTS_PATH.open(encoding="utf-8") as f:
        data=json.load(f)
    assert len(data["weight"])==2
    assert all(len(w)==1024 for w in data["weight"])
    assert len(data["bias"])==2
    assert sum(map(len,data["weight"]))+len(data["bias"])==2050
