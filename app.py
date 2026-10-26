"""Smart Agriculture Assistant: a simple Streamlit app for three separate models."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.soil_advice import get_soil_improvement_advice

ROOT = Path(__file__).resolve().parent
MODEL_FOLDER = ROOT / "models"

st.set_page_config(page_title="Fieldwise | Smart Agriculture", page_icon="🌿", layout="wide")

# A dark forest theme gives the app a calm, readable agricultural look.
st.markdown("""
<style>
:root { --night: #0d1711; --forest: #14231a; --panel: #192a20; --leaf: #86c982; --mint: #c5e3a8; --text: #edf3ed; --muted: #b1c2b4; --line: #304739; }
html, body, [data-testid="stAppViewContainer"] { background: var(--night); color: var(--text); }
[data-testid="stHeader"] { background: rgba(13,23,17,.94); }
.main .block-container { max-width: 1180px; padding-top: 1.8rem; padding-bottom: 3rem; }
.hero { background: linear-gradient(125deg, #11251a 0%, #19432d 58%, #24563a 100%); border: 1px solid #315741; box-shadow: 0 14px 38px rgba(0,0,0,.22); color: var(--text); padding: 1.9rem 2rem; border-radius: 22px; margin: .2rem 0 1.3rem; }
.hero h1 { color: #f3f8f2; margin: 0; font-size: 2.35rem; letter-spacing: -.035em; }
.hero p { color: #d1e3d3; margin: .45rem 0 0; font-size: 1rem; }
.eyebrow { color: var(--mint); font-size: .76rem; font-weight: 750; letter-spacing: .15em; text-transform: uppercase; margin-bottom: .25rem; }
div[data-testid="stTabs"] button { color: var(--muted); font-weight: 650; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--leaf); }
.stButton button, [data-testid="stFormSubmitButton"] button { background: var(--leaf); color: #102017; border: 1px solid var(--leaf); border-radius: 10px; font-weight: 700; padding: .58rem 1.1rem; transition: .15s ease; }
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover { background: #a1d995; color: #102017; border-color: #a1d995; }
div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); padding: .9rem 1rem; border-radius: 14px; }
div[data-testid="stMetricLabel"] { color: var(--muted); }
div[data-testid="stMetricValue"] { color: var(--text); }
div[data-testid="stAlert"] { border-radius: 12px; }
.section-note { color: var(--muted); font-size: .95rem; }
.footer-note { background: #17291e; border: 1px solid #2c4935; border-left: 4px solid var(--leaf); padding: 1rem 1.2rem; border-radius: 14px; color: #dce9dd; }
[data-testid="stFileUploader"] section { background: var(--panel); border: 1px dashed #49674f; border-radius: 14px; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: var(--muted); }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea { background: var(--panel); color: var(--text); }
[data-baseweb="select"] > div { background: var(--panel); }
hr { border-color: var(--line); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">SMART AGRICULTURE ASSISTANT</div>
  <h1>Fieldwise</h1>
  <p>Practical, data-supported tools for understanding soil and planning crops.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Educational decision support only. Predictions do not replace '
    'laboratory soil testing or advice from an agricultural specialist.</div>',
    unsafe_allow_html=True,
)
st.write("")


def load_model_bundle(filename):
    """Load one saved tabular model. Return None when it has not been trained yet."""
    path = MODEL_FOLDER / filename
    if not path.exists():
        return None
    return joblib.load(path)


def show_missing_model(name, command):
    st.info(f"The {name} model is not available yet. Run `{command}` from the project folder.")


def make_input_row(values, columns):
    """Put the entered values into the exact order used when the model was trained."""
    row = {column: values.get(column.lower()) for column in columns}
    return pd.DataFrame([row], columns=columns)


def show_prediction_table(classes, scores):
    results = pd.DataFrame({"Class": classes, "Model score": scores})
    results = results.sort_values("Model score", ascending=False)
    st.dataframe(results, hide_index=True, use_container_width=True)


soil_tab, crop_tab, fertility_tab = st.tabs([
    "📷  Soil image", "🌱  Crop suggestion", "🧪  Soil fertility"
])

with soil_tab:
    st.subheader("Identify the visual soil class")
    st.markdown('<p class="section-note">Upload a clear soil photo. The model recognizes image categories only; it does not measure nutrients or fertility.</p>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload a JPG, PNG, or WEBP photo", type=["jpg", "jpeg", "png", "webp"])
    if upload:
        image = Image.open(upload).convert("RGB")
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            st.image(image, caption="Uploaded soil photo", use_container_width=True)
        model_path = MODEL_FOLDER / "soil_cnn.keras"
        names_path = ROOT / "artifacts" / "soil_class_names.json"
        if not model_path.exists() or not names_path.exists():
            with right:
                show_missing_model("soil image", "python src/train_soil_cnn.py")
        else:
            with st.spinner("Checking the soil image..."):
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path)
                    classes = json.loads(names_path.read_text(encoding="utf-8"))
                    pixels = np.asarray(image.resize((224, 224)), dtype=np.float32)[None, ...]
                    scores = model.predict(pixels, verbose=0)[0]
                    best = int(np.argmax(scores))
                    with right:
                        st.success(f"Predicted visual class: **{classes[best].replace('_', ' ')}**")
                        st.metric("Top model score", f"{scores[best]:.1%}")
                        st.caption("This is a model score, not a calibrated probability or a soil-fertility measurement.")
                        show_prediction_table([name.replace("_", " ") for name in classes], scores)
                except Exception as error:
                    with right:
                        st.error(f"The image model could not run: {error}")

with crop_tab:
    st.subheader("Find crops that match the entered conditions")
    st.markdown('<p class="section-note">This model uses the crop dataset’s nutrient and weather inputs. It is separate from the soil-image and fertility models.</p>', unsafe_allow_html=True)
    crop_model = load_model_bundle("crop_model.joblib")
    if crop_model is None:
        show_missing_model("crop recommendation", "python src/train_tabular.py --task crop")
    else:
        with st.form("crop_inputs"):
            first, second, third = st.columns(3)
            with first:
                nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=50.0)
                phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0)
                potassium = st.number_input("Potassium (K)", min_value=0.0, value=50.0)
            with second:
                temperature = st.number_input("Temperature (°C)", value=25.0)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
            with third:
                ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
            crop_submitted = st.form_submit_button("Suggest a crop")
        if crop_submitted:
            values = {"n": nitrogen, "p": phosphorus, "k": potassium,
                      "temperature": temperature, "humidity": humidity,
                      "ph": ph, "rainfall": rainfall}
            try:
                row = make_input_row(values, crop_model["feature_columns"])
                prediction = int(crop_model["pipeline"].predict(row)[0])
                crop_name = crop_model["classes"][prediction]
                st.success(f"Dataset-predicted crop: **{crop_name.title()}**")
                st.caption(f"Selected model: {crop_model['selected_model']}. The source data is India-focused and may not match local farm conditions.")
            except Exception as error:
                st.error(f"Could not make a crop prediction: {error}")

with fertility_tab:
    st.subheader("Check the dataset-based fertility level")
    st.markdown('<p class="section-note">This model uses Soil Sight remote-sensing and location features. It does not use the crop model’s NPK and weather inputs.</p>', unsafe_allow_html=True)
    fertility_model = load_model_bundle("fertility_model.joblib")
    if fertility_model is None:
        show_missing_model("soil fertility", "python src/train_tabular.py --task fertility")
    else:
        feature_columns = fertility_model["feature_columns"]
        feature_info = fertility_model.get("feature_metadata", {})
        with st.form("fertility_inputs"):
            st.caption("Enter values in the same units and ranges as the source dataset.")
            entered_values = {}
            left, right = st.columns(2)
            for index, column in enumerate(feature_columns):
                info = feature_info.get(column, {})
                with (left if index % 2 == 0 else right):
                    if info.get("kind") == "categorical":
                        choices = info.get("categories", [])
                        if choices:
                            entered_values[column] = st.selectbox(column, choices)
                        else:
                            entered_values[column] = st.text_input(column)
                    else:
                        entered_values[column] = st.number_input(
                            column, value=float(info.get("median", 0.0)),
                            min_value=info.get("min"), max_value=info.get("max"),
                            format="%.6f",
                        )
            fertility_submitted = st.form_submit_button("Check fertility level")

        if fertility_submitted:
            try:
                row = pd.DataFrame([entered_values], columns=feature_columns)
                class_number = int(fertility_model["pipeline"].predict(row)[0])
                fertility_label = str(fertility_model["classes"][class_number])
                st.session_state["fertility_prediction"] = fertility_label
                st.session_state.pop("soil_advice", None)
            except Exception as error:
                st.session_state.pop("fertility_prediction", None)
                st.error(f"Could not make a fertility prediction: {error}")

        fertility_label = st.session_state.get("fertility_prediction")
        if fertility_label:
            st.metric("Predicted fertility level", fertility_label)
            st.caption(f"Selected model: {fertility_model['selected_model']}. This is a remote-sensing-based dataset prediction, not a laboratory soil test.")

            if fertility_label.lower() == "low":
                st.warning("The model predicted Low fertility. Confirm the result with a soil laboratory before changing how you manage the soil.")
                st.markdown("#### Get general improvement guidance")
                guide_left, guide_right = st.columns(2)
                with guide_left:
                    crop_context = st.text_input("Current or planned crop (optional)", key="advice_crop")
                with guide_right:
                    region_context = st.text_input("Country or region (optional)", key="advice_region")
                st.caption("AI-generated guidance uses the OpenAI API when you request it. API usage charges may apply. No advice is generated automatically.")
                if st.button("Generate safe next-step guidance", type="primary"):
                    with st.spinner("Preparing general soil-improvement guidance..."):
                        try:
                            st.session_state["soil_advice"] = get_soil_improvement_advice(
                                crop=crop_context, region=region_context
                            )
                        except Exception as error:
                            st.session_state["soil_advice"] = None
                            st.error(str(error))
                if st.session_state.get("soil_advice"):
                    st.markdown(st.session_state["soil_advice"])
                st.info("The AI guide cannot diagnose nutrient shortages from this remote-sensing result. It will not prescribe fertilizer types, quantities, or application rates.")
            else:
                st.success("The low-fertility advice feature is available only when the model predicts Low.")

with st.expander("About the models"):
    st.markdown(
        "- **Soil image:** predicts a visual soil category only.\n"
        "- **Crop suggestion:** uses N, P, K, temperature, humidity, pH, and rainfall.\n"
        "- **Soil fertility:** predicts the dataset’s four labels from separate remote-sensing and location features.\n"
        "- One model’s result is not used as another model’s input."
    )
