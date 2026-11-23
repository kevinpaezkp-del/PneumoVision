# PneumoVision — Proyecto Final · Visión por Computador

Demo académica en **Streamlit + PyTorch + TorchXRayVision** para clasificar radiografías de tórax pediátricas entre `NORMAL` y `PNEUMONIA`.

> **Advertencia:** no es una herramienta de diagnóstico. Los porcentajes son puntuaciones del modelo (softmax), **no probabilidades clínicas calibradas**. El modelo registró una sensibilidad baja sobre un conjunto de validación externa y no debe utilizarse para tomar decisiones clínicas.

## Archivos

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación visual Streamlit; subida de radiografía y resultados. |
| `model_utils.py` | Misma normalización, recorte y transformación del cuaderno original. |
| `head_weights.json` | **Tus pesos entrenados** (2 × 1024 y sesgos); no se reentrena. |
| `requirements.txt` | Dependencias para Python 3.12 en Streamlit Community Cloud (PyTorch solo CPU). |
| `.streamlit/config.toml` | Colores y límites del servidor. |
| `tests/test_weights.py` | Comprobación básica del archivo de pesos. |

## Publicación gratuita en Streamlit Community Cloud

1. Crear un **repositorio público** en [github.com/new](https://github.com/new), por ejemplo `PneumoVision`.
2. Subir **todos** los archivos y conservar la carpeta `.streamlit`.
3. Entrar en [share.streamlit.io](https://share.streamlit.io/) e iniciar sesión con GitHub.
4. Pulsar **Create app**, seleccionar el repositorio, la rama `main` y `app.py` como archivo principal.
5. Elegir **Python 3.12** en los ajustes de despliegue (o 3.11 si la plataforma ofrece incompatibilidades).
6. Publicar. La primera vez la aplicación **descargará automáticamente** el DenseNet121 público. Deberá tener acceso de salida a GitHub; los 2.050 pesos particulares ya están incluidos en `head_weights.json`.
7. Probar con una radiografía de ejemplo **sin datos de paciente**.

El repositorio de GitHub **almacena** el código y los pesos de la cabeza; no ejecuta el modelo. Streamlit Community Cloud ejecuta Python de forma gratuita con límites de memoria y tiempo de inactividad.

## ¿Por qué funciona con un JSON tan pequeño?

En el cuaderno se reutilizó el extractor DenseNet121 médico `densenet121-res224-all`, se congelaron sus parámetros y solo se ajustó una capa `Linear(1024, 2)` (2.050 parámetros). El JSON **no es el modelo completo**: al iniciar, la aplicación descarga el extractor original y monta la cabeza entrenada.

La normalización es `xrv.datasets.normalize(img, 255)` seguida de `XRayCenterCrop()` y `XRayResizer(224)`, igual que `preprocess_pil()` del cuaderno original. Después de inferencia se aplica `softmax` y el umbral de clasificación 0,5.

## Resultados observados en el cuaderno

| Conjunto | Accuracy | Observación |
|---|---:|---|
| Prueba reservada Guangzhou (n=120) | 85 % | Muestra equilibrada de la misma fuente |
| Prueba externa BDCXR-3257 Bangladés (n=100) | 56 % | Sensibilidad del 16 %; generalización deficiente |

**No** se hizo calibración, no se comprobó aplicabilidad clínica, no se debe presentar el porcentaje como probabilidad de enfermedad, y no se deben subir datos clínicos reales identificables.

## Ejecución local (opcional)

```bash
python -m venv .venv
# Activar .venv con el método correspondiente al sistema operativo
pip install -r requirements.txt
streamlit run app.py
```

## Verificación sin inferencia

```bash
python -m pytest -q
```

La validación del archivo y la sintaxis puede realizarse sin descargar el modelo completo. La primera inferencia real requiere instalar dependencias y descargar el DenseNet121 desde TorchXRayVision.

## Referencias

- TorchXRayVision: https://github.com/mlmed/torchxrayvision
- Dataset: https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia
- Publicación: https://docs.streamlit.io/deploy/streamlit-community-cloud