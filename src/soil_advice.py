"""Create cautious soil-improvement guidance after a Low fertility prediction."""
import os
import json

import streamlit as st
from openai import OpenAI


def read_setting(name, default=""):
    """Read a value from Streamlit Secrets first, then from environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        # Local runs may not have a Streamlit secrets file.
        pass
    return os.getenv(name, default)


def get_soil_improvement_advice(crop="", region="", fertility_inputs=None, predicted_label="Low", selected_model=""):
    """Use the predicted label and entered features to tailor safe, general next steps."""
    api_key = read_setting("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Add your OpenAI key as OPENAI_API_KEY in the hosting platform's private Secrets settings. "
            "Do not put it in this code or commit it to GitHub."
        )

    base_url = read_setting("OPENAI_API_BASE")
    model_name = read_setting("OPENAI_MODEL", "gpt-5-mini")
    client_options = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    client = OpenAI(**client_options)

    crop_text = crop.strip() or "not provided"
    region_text = region.strip() or "not provided"
    feature_text = json.dumps(fertility_inputs or {}, ensure_ascii=False, indent=2, default=str)
    model_text = selected_model or "not provided"
    response = client.chat.completions.create(
        model=model_name,
        # Reasoning models use part of the token budget before writing the answer.
        max_completion_tokens=1600,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful agricultural education assistant. The only soil result is a "
                    "remote-sensing/geospatial model prediction; it is not a laboratory test and "
                    "does not identify nutrient deficiencies. The numeric inputs may be vegetation "
                    "indices, moisture estimates, elevation, or location—not direct laboratory NPK "
                    "measurements. Use the supplied feature names and values only as context. Do not "
                    "infer a specific nutrient shortage, recommend a fertilizer product, give an exact "
                    "dose/application rate, or claim the values prove a cause. Give 3 to 5 short, "
                    "practical next steps: confirm with a lab test for pH, NPK, organic matter and "
                    "salinity, consult a local agronomist, then consider locally suitable actions such "
                    "as improving organic matter or cover crops. Explain how the measurements inform "
                    "general follow-up without overstating what they mean. Use plain English and bullets."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Fertility model prediction: {predicted_label}. Selected algorithm: {model_text}. "
                    f"Entered fertility-model feature values (names are the dataset's original fields):\n{feature_text}\n"
                    f"Current/planned crop: {crop_text}. Region: {region_text}. "
                    "Personalize safe next steps using these inputs where appropriate, clearly "
                    "distinguish recorded values from confirmed soil measurements, and avoid exact doses."
                ),
            },
        ],
    )
    advice = response.choices[0].message.content
    if not advice:
        raise RuntimeError("The AI service returned an empty answer. Please try again later.")
    return advice.strip()
