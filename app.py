"""PneumoVision — Proyecto final de visión por computador. Uso académico."""
from __future__ import annotations

from io import BytesIO
import logging

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from model_utils import load_classifier, predict

st.set_page_config(
    page_title="PneumoVision | Proyecto final",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource(show_spinner=False)
def get_model():
    import torch
    torch.set_num_threads(min(2, torch.get_num_threads()))
    return load_classifier()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--navy:#10263b;--teal:#1389a0;--light:#edf6f8;--muted:#667a8c}
.stApp{background:linear-gradient(160deg,#eef6f9,#f9fbfd 56%,#eff5f9);color:var(--navy)}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:2.5rem}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.brand{font:700 1.33rem 'Space Grotesk',sans-serif;color:#102b40;letter-spacing:-.05em}
.brand span{color:#13a3b7}
.topright{text-align:right;font-size:.76rem;letter-spacing:.08em;font-weight:800;color:#6a8294;padding-top:.4rem}
.hero{margin:1.55rem 0 2rem;padding:2.6rem 3rem;border-radius:24px;color:white;background:linear-gradient(115deg,#102b45 2%,#16445c 58%,#0d7b88 120%);box-shadow:0 22px 55px #143f5a19}
.hero .tag{display:inline-block;border:1px solid #89dbe56b;background:#ffffff14;border-radius:30px;padding:.35rem .85rem;font-size:.69rem;letter-spacing:.1em;font-weight:800;color:#c7f4f7}
.hero h1{color:#fff;font:700 clamp(2.1rem,4vw,3.6rem)/1.08 'Space Grotesk',sans-serif;letter-spacing:-.055em;margin:1.3rem 0 .9rem}
.hero p{color:#d7eaf0;font-size:1.02rem;max-width:670px;line-height:1.75;margin:0}
.chips{display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1.5rem}
.chips span{background:#ffffff15;border:1px solid #ffffff22;border-radius:10px;padding:.45rem .75rem;font-size:.76rem;color:#e7f6fa;font-weight:700}
.kicker{font-size:.74rem;letter-spacing:.1em;color:#12849a;font-weight:800;text-transform:uppercase;margin-bottom:.2rem}
.sectiontitle{font:700 1.5rem 'Space Grotesk',sans-serif;color:#173449;letter-spacing:-.04em}
.muted{font-size:.95rem;color:#6a7f91;line-height:1.7;margin:.4rem 0 1.25rem}
.surface{background:white;border:1px solid #dfeaf0;border-radius:17px;padding:1.3rem;box-shadow:0 7px 26px #1c45650a}
.surface h3{font:700 1.08rem 'Space Grotesk',sans-serif;color:#173449;margin:.1rem 0 .5rem}
.surface p{font-size:.9rem;color:#718597;line-height:1.65}
.result{background:white;border:1px solid #d6e5eb;border-radius:18px;padding:1.6rem;box-shadow:0 8px 25px #173b5810;margin-bottom:1rem}
.result .k{color:#678092;font-size:.73rem;letter-spacing:.08em;text-transform:uppercase;font-weight:800}
.result .v{font:700 clamp(1.5rem,3vw,2.2rem) 'Space Grotesk',sans-serif;letter-spacing:-.05em;margin:.8rem 0;color:#1b3e54}
.result .pct{font:700 2.45rem 'Space Grotesk',sans-serif;color:#117a90;letter-spacing:-.05em}
.result small{color:#788b9a;font-size:.81rem;line-height:1.65}
.pair{display:flex;justify-content:space-between;font-weight:800;font-size:.87rem;color:#425f72;margin-top:.5rem}
.track{height:12px;border-radius:20px;background:#e9f1f5;margin:.35rem 0 .9rem;overflow:hidden}
.bar{height:100%;background:linear-gradient(90deg,#147b91,#21c6d5);border-radius:20px}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:1rem 0 1.5rem}
.stats div{background:white;border:1px solid #dfe9f0;padding:1.2rem;border-radius:15px}
.stats strong{display:block;font:700 1.48rem 'Space Grotesk',sans-serif;color:#18435a}
.stats span{font-size:.79rem;color:#698193}
[data-testid="stFileUploader"]{border:1px solid #dce9f0;border-radius:16px;padding:.6rem;background:#fff}
[data-testid="stFileUploaderDropzone"]{border:1px dashed #82c8d2;border-radius:10px;background:#f7fcfc}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#0d6b85,#139eaf)!important;border:0!important;border-radius:12px!important;font-weight:800!important;min-height:3rem}
.stButton>button[kind="primary"]:hover{background:#0d6b85!important}
.end{margin-top:2rem;padding:1.4rem 0;border-top:1px solid #dce8ee;color:#7c90a0;font-size:.79rem;text-align:center}
@media(max-width:700px){.hero{padding:1.8rem 1.5rem}.stats{grid-template-columns:1fr}.topright{font-size:.58rem}}
</style>
""", unsafe_allow_html=True)

a, b = st.columns([1, 1])
with a:
    st.markdown('<div class="brand"><span>✚</span> PneumoVision</div>', unsafe_allow_html=True)
with b:
    st.markdown('<div class="topright">PROYECTO FINAL · VISIÓN POR COMPUTADOR · 2026</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <span class="tag">INTELIGENCIA ARTIFICIAL · IMÁGENES MÉDICAS</span>
  <h1>Explorando la neumonía<br>con aprendizaje profundo.</h1>
  <p>Proyecto académico de clasificación de radiografías pediátricas.
  Reutilizamos una red DenseNet121 preentrenada y adaptamos su cabeza
  mediante transferencia de aprendizaje.</p>
  <div class="chips"><span>DenseNet121</span><span>PyTorch</span>
  <span>Transfer learning</span><span>Radiografías pediátricas</span></div>
</div>
""", unsafe_allow_html=True)

st.warning(
    "DEMOSTRACIÓN ACADÉMICA: este sistema NO diagnostica ni descarta neumonía. "
    "Sus resultados no son clínicamente fiables para pacientes individuales. "
    "No subas radiografías con datos personales."
)

st.markdown('<div class="kicker">01 / Análisis interactivo</div><div class="sectiontitle">Carga una radiografía de tórax</div>', unsafe_allow_html=True)
st.markdown('<p class="muted">Prueba el modelo entrenado: selecciona una imagen JPG o PNG y consulta la clase asignada y su puntuación. Utiliza radiografías públicas de ejemplo sin información identificable.</p>', unsafe_allow_html=True)

left, right = st.columns([1, 1.1], gap="large")
image = None
with left:
    uploaded = st.file_uploader(
        "Seleccionar imagen", type=["jpg", "jpeg", "png"],
        help="Radiografía frontal de tórax, preferiblemente pediátrica.",
    )
    if uploaded is not None:
        if uploaded.size > 12 * 1024 * 1024:
            st.error("El archivo supera los 12 MB permitidos.")
            st.stop()
        try:
            raw = Image.open(BytesIO(uploaded.getvalue()))
            raw.load()
            if min(raw.size) < 64 or max(raw.size) > 10000:
                st.error("Resolución no compatible. Usa imágenes entre 64 y 10.000 píxeles por lado.")
                st.stop()
            image = ImageOps.exif_transpose(raw).convert("RGB")
            st.image(image, caption="Radiografía cargada", use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("No fue posible abrir la imagen. Usa JPG o PNG válido.")
            st.stop()
    analyze = st.button("Analizar radiografía", type="primary", use_container_width=True, disabled=image is None)
    st.markdown('<div class="surface"><h3>¿Cómo funciona?</h3><p>La radiografía se convierte a escala de grises, se normaliza, se recorta y se ajusta a 224 × 224 píxeles. Una red convolucional extrae sus características y la cabeza entrenada produce dos puntuaciones.</p></div>', unsafe_allow_html=True)

with right:
    if not analyze:
        st.markdown('<div class="result"><div class="k">Resultado del algoritmo</div><div class="v">Tu análisis aparecerá aquí</div><small>Sube una radiografía y pulsa “Analizar radiografía”.</small></div>', unsafe_allow_html=True)
    else:
        try:
            with st.spinner("Cargando el modelo y analizando la radiografía..."):
                model = get_model()
                normal, pneumonia = predict(model, image)
        except Exception:
            logging.exception("Fallo al cargar o ejecutar PneumoVision")
            st.error("No fue posible analizar la imagen. Consulta los registros de la aplicación.")
            st.stop()

        selected = "PNEUMONIA" if pneumonia >= 0.5 else "NORMAL"
        score = pneumonia if selected == "PNEUMONIA" else normal
        label = "Patrón clasificado como NEUMONÍA" if selected == "PNEUMONIA" else "Radiografía clasificada como NORMAL"
        st.markdown(f"""
        <div class="result">
            <div class="k">Resultado de clasificación</div>
            <div class="v">{label}</div>
            <div class="pct">{score:.1%}</div>
            <small>Puntuación softmax de la clase elegida. No representa una probabilidad médica calibrada.</small>
        </div>""", unsafe_allow_html=True)
        for label_name, value in [("NEUMONÍA", pneumonia), ("NORMAL", normal)]:
            st.markdown(f'<div class="pair"><span>{label_name}</span><span>{value:.1%}</span></div><div class="track"><div class="bar" style="width:{value*100:.2f}%"></div></div>', unsafe_allow_html=True)
        st.caption("Umbral de demostración: 50 %. El modelo puede equivocarse.")
        st.info("La clasificación no reemplaza la valoración médica ni la lectura radiológica.")

st.write("")
st.markdown('<div class="kicker">02 / Sobre el proyecto</div><div class="sectiontitle">Transferencia de aprendizaje</div>', unsafe_allow_html=True)
st.markdown('<p class="muted">Partimos de un modelo preentrenado con grandes colecciones de radiografías. Congelamos el extractor de características y entrenamos una nueva capa binaria con imágenes pediátricas etiquetadas.</p>', unsafe_allow_html=True)

st.markdown("""
<div class="stats">
    <div><strong>DenseNet121</strong><span>Arquitectura médica preentrenada</span></div>
    <div><strong>2.050</strong><span>Parámetros de la cabeza ajustados</span></div>
    <div><strong>2 clases</strong><span>NORMAL y PNEUMONIA</span></div>
</div>""", unsafe_allow_html=True)

with st.expander("Metodología y validación"):
    st.markdown("""
    - **Extractor:** TorchXRayVision, pesos densenet121-res224-all.
    - **Adaptación:** reemplazo de la última capa por Linear(1024, 2).
    - **Entrenamiento:** 300 imágenes pediátricas; solo la cabeza nueva.
    - **Prueba de la misma fuente:** 85 % de accuracy en 120 imágenes.
    - **Prueba externa:** 56 % de accuracy en 100 imágenes de Bangladés, con sensibilidad del 16 %.

    La caída en validación externa evidencia una capacidad de generalización limitada. **Este modelo es exclusivamente didáctico y no está validado para uso clínico.**
    """)

with st.expander("Fuentes y tecnologías"):
    st.markdown("""
    - [TorchXRayVision — DenseNet121](https://github.com/mlmed/torchxrayvision)
    - [Chest X-Ray Pneumonia — conjunto de entrenamiento](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)
    - [BDCX​R-3257 — validación externa](https://figshare.com/articles/dataset/_b_BDCXR-3257_b_A_Comprehensive_Chest_Radiography_Dataset_with_Multi-Etiological_Pneumonia_Labels_from_a_Clinical_Setting/32086362)
    - Tecnologías: Python, PyTorch, NumPy y Streamlit.
    """)

st.markdown('<div class="end">PneumoVision · Proyecto Final · Maestría en Ciencia de Datos · 2026<br>Exclusivamente para fines educativos y de investigación</div>', unsafe_allow_html=True)
