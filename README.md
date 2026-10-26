# Smart Agriculture Assistant

A beginner-friendly learning project with three separate workflows:

1. **Soil image classification:** MobileNetV2 transfer learning classifies seven visual soil categories. It does not measure soil fertility.
2. **Crop recommendation:** Logistic Regression, SVM, and XGBoost are compared on the crop dataset using N, P, K, temperature, humidity, pH, and rainfall.
3. **Soil fertility classification:** The same three tabular algorithms are compared on the Soil Sight dataset's documented four labels: Very High, High, Moderate, and Low. Its remote-sensing/geospatial inputs are independent of the crop model inputs.

The Streamlit app displays separate tabs. A prediction in one tab does not trigger another model. Predictions are educational, not agronomic advice or laboratory tests.

## 1. Set up Python

Use Python 3.10, 3.11, or 3.12. In a terminal, enter:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow is a large dependency. If your computer cannot install it, you can still train and run both tabular models by removing `tensorflow` from `requirements.txt` and skipping the CNN commands.

## 2. Download the datasets

From the project folder, use the downloader for both official tabular datasets:

```bash
python src/download_data.py
```

To also download the original soil images (about 238 MB), run:

```bash
python src/download_data.py --images
```

The files should be organized as follows:

Create the folders shown below. Never commit large datasets unless your instructor/team has approved it.

```text
data/
  crop/Crop_recommendation.csv
  fertility/soil_sight.csv
  soil_images/Alluvial Soil/*.jpg
  soil_images/Black Soil/*.jpg
  soil_images/Laterite Soil/*.jpg
  soil_images/Red Soil/*.jpg
  soil_images/Yellow Soil/*.jpg
  soil_images/Arid Soil/*.jpg
  soil_images/Mountain Soil/*.jpg
```

- **Crop data:** download `Crop_recommendation.csv` from [Kaggle: Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). It is based on India-focused climate, rainfall, and fertilizer data.
- **Fertility data:** the downloader selects the current Version 2 CSV from [Mendeley Data: Soil Sight](https://data.mendeley.com/datasets/h73wn2nw56/2). Its target is `Fertility_Level`; features include latitude, longitude, NDVI, EVI, SAVI, soil moisture, and elevation. The Version 2 dataset uses four target labels: Very High, High, Moderate, and Low. Its license is CC BY 4.0.
- **Soil images:** obtain the original (not CyAUG) image folders from [Phantom-fs/Soil-Classification-Dataset](https://github.com/Phantom-fs/Soil-Classification-Dataset). The repository lists 1,189 original images across the seven classes. The repository license is CC BY-NC 4.0; use for non-commercial educational/research work only and cite the authors as described in the source repository. Keep original images separate from the CyAUG images to reduce near-duplicate leakage.

If a source's current file layout differs, keep its real labels/features and update the path or documented target column. Do not fabricate training examples or fill missing target values with guesses.

## 3. Audit and train tabular models

```bash
python src/train_tabular.py --audit
python src/train_tabular.py --task all
```

Or train one task at a time:

```bash
python src/train_tabular.py --task crop
python src/train_tabular.py --task fertility
```

The script prints missing values, duplicate counts, feature types, and class balance. Crop data uses a stratified train/validation/test split. Fertility data uses coordinate-block grouping for its split when coordinates are present, to reduce nearby-location leakage. Imputation and scaling are fitted only inside each model pipeline. Logistic Regression and linear SVM use scaling; XGBoost uses imputation without scaling. All three algorithms are compared using validation Macro F1; the selected algorithm is refitted on train+validation and evaluated once on the held-out test set. Metrics and confusion matrices are saved under `artifacts/evaluation/`, and the selected complete preprocessing/model pipeline is saved under `models/`.

Baseline results from the current source files are in `artifacts/evaluation/`. On crop data, validation Macro F1 was 0.9636 for Logistic Regression, 0.9681 for linear SVM, and 0.9909 for XGBoost; the selected model's held-out test accuracy was 0.9909 and Macro F1 was 0.9908. On fertility data, validation Macro F1 was 0.9811 for Logistic Regression, 0.8943 for linear SVM, and 0.9911 for XGBoost; its coordinate-block test accuracy was 0.9977 and Macro F1 was 0.9955. The raw fertility v2 CSV also has 263 NDVI values and 6 EVI values outside the usual [-1, 1] index range. The code preserves source values and the audit warns about them; confirm their units or correct interpretation with the dataset authors before treating results as scientific evidence. These high scores and source anomalies mean the benchmark does not demonstrate generalization to other regions or independent farms.

The MobileNetV2 image model scored 0.7982 test accuracy and 0.7352 Macro F1 on its 114-image held-out split. Class-wise test F1 was weakest for Alluvial Soil (0.500) and Red Soil (0.556); those classes had only 5 and 11 test images respectively. Use the full class report and confusion matrix rather than relying on accuracy alone.

If XGBoost cannot be installed for your computer, install it separately with `pip install xgboost`; the script will still compare the other models and clearly report the skipped model. Do not treat a high score on these curated datasets as proof of performance on a real farm.

## 4. Train the image model

After placing original images in the seven class folders:

```bash
python src/train_soil_cnn.py
```

The script removes byte-for-byte duplicate files, converts verified images to a local high-quality RGB JPEG cache (so WebP and other source formats work with TensorFlow), then makes deterministic stratified train/validation/test splits. The original images remain unchanged. It uses MobileNetV2 with ImageNet weights, freezes the base model, applies augmentations only during training, balances loss across classes, and uses early stopping. It writes the trained model, class names, metrics, and a confusion matrix. The test split is not augmented. Exact hashes do not identify visually similar near-duplicates, so review the source images where possible.

## 5. Run the app

After training the relevant model(s):

```bash
streamlit run app.py
```

The app handles missing model files by showing training instructions. The crop and fertility tabs require the matching saved model. The fertility form uses the feature schema saved from the fertility training CSV, so it can collect the dataset's actual numeric and categorical columns instead of pretending those inputs are NPK/weather readings.

## 6. Low-fertility AI guidance

The app shows an AI guidance button only after the fertility model predicts **Low**. The user must press the button; advice is not generated automatically. When requested, the helper sends the predicted label, selected classifier, entered fertility feature names and values, and optional crop/region context to the configured language-model provider. These remote-sensing/geospatial features are not lab measurements; the assistant must not interpret them as confirmed nutrient deficiencies or prescribe fertilizer products, amounts, or application rates. It recommends confirming soil pH, NPK, organic matter, and salinity with a laboratory and consulting a local agricultural specialist.

The feature uses the OpenAI Python client with OpenRouter's OpenAI-compatible streaming chat-completions endpoint. Add these settings to the hosting platform's private **Secrets** box, replacing the placeholder with your own OpenRouter API key:

```toml
OPENROUTER_API_KEY = "your-openrouter-key"
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
```

The code defaults to this model and endpoint, but your key must be configured for the hosted app to make requests. The `:free` model designation does not guarantee uninterrupted availability; provider access, rate limits, and model availability can change. Do not put a real key in code or GitHub. For local testing, create `.streamlit/secrets.toml` using `.streamlit/secrets.toml.example` as a guide; `.streamlit/secrets.toml` is ignored by Git.

## 7. Publish the Streamlit app

The project is prepared for a Streamlit-compatible host such as Streamlit Community Cloud:

1. Put the project in a GitHub repository. Include the trained files in `models/` and `artifacts/soil_class_names.json`; keep the raw `data/` folder out of Git.
2. In the Streamlit hosting dashboard, create an app from that repository and select `app.py` as the entry point. Use Python 3.12 to match the TensorFlow dependency.
3. In the app's private **Secrets** settings, add the three OpenRouter values shown above and put your real key only in `OPENROUTER_API_KEY`. Do not add secrets to source code, GitHub, or screenshots.
4. After deployment, open the app's public URL. The three model workflows remain separate.

The Streamlit host needs access to the GitHub repository. This project folder alone is not yet a published website; deployment is complete only after the repository is connected and the host has built the app.

## Project structure

```text
app.py
src/
  train_tabular.py
  train_soil_cnn.py
  soil_advice.py        # optional, on-demand OpenAI guidance
models/                 # trained model files; include these for deployment
artifacts/              # class names and evaluation reports
 data/                  # downloaded data; not committed
```

## What the models can and cannot say

The CNN predicts the visual class named by its training images. The crop model maps its seven recorded conditions to crop labels in the supplied crop dataset. The fertility model predicts the Soil Sight `Fertility_Level` from that dataset's inputs. These datasets do not establish a validated mapping between a fertility class and a crop recommendation. Soil images do not reveal laboratory NPK/pH values. Use local agronomic guidance and laboratory testing for real decisions.

## Sources

- Sheth et al., [Soil Classification Dataset repository](https://github.com/Phantom-fs/Soil-Classification-Dataset), CC BY-NC 4.0; see repository citation and license.
- Ingle, [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
- Punjabi, Pantawane, Patil, and Kadam, [Soil Sight: Remote Sensing-Based Soil Fertility Dataset](https://doi.org/10.17632/h73wn2nw56.1), Mendeley Data, 2025, CC BY 4.0.
