"""Create cautious soil-improvement guidance after a Low fertility prediction."""
import os

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


def get_soil_improvement_advice(crop="", region=""):
    """Ask the LLM for broad next steps; never treat the model result as a lab test."""
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
    response = client.chat.completions.create(
        model=model_name,
        # Reasoning models use part of the token budget before writing the answer.
        max_completion_tokens=1600,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful agricultural education assistant. The only soil result is a "
                    "remote-sensing model prediction of Low fertility; it is not a laboratory test "
                    "and does not identify a nutrient deficiency. Give 3 to 5 short, practical next "
                    "steps. First recommend a laboratory soil test for pH, NPK, organic matter and "
                    "salinity, followed by local agronomist advice. Offer general practices only, "
                    "such as improving organic matter or considering suitable cover crops after local "
                    "review. Do not prescribe a fertilizer product, exact dose, application rate, "
                    "or claim a specific nutrient is deficient. State that suitability depends on "
                    "the crop and local conditions. Use plain English and a short bullet list."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The app predicted Low fertility from remote-sensing/geospatial data. "
                    f"Current/planned crop: {crop_text}. Region: {region_text}. "
                    "Give safe next steps to investigate and improve the soil, without exact doses."
                ),
            },
        ],
    )
    advice = response.choices[0].message.content
    if not advice:
        raise RuntimeError("The AI service returned an empty answer. Please try again later.")
    return advice.strip()
