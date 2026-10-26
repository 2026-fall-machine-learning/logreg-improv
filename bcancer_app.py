from fasthtml.common import *
import joblib
from sklearn.datasets import load_breast_cancer

MODEL_PATH = "bcancer_logreg.joblib"
model = joblib.load(MODEL_PATH)

data = load_breast_cancer()
FEATURE_NAMES = list(data.feature_names)
FEATURE_MEANS = data.data.mean(axis=0)
TARGET_NAMES = list(data.target_names)  # ['malignant', 'benign']

app, rt = fast_app()


def feature_field(name: str, default: float):
    # sanitize "mean radius" -> "mean_radius" for use as an input name/id
    field_id = name.replace(" ", "_")
    return Div(
        Label(name, fr=field_id),
        Input(
            type="number",
            step="any",
            id=field_id,
            name=field_id,
            value=f"{default:.4f}",
            required=True,
        ),
        cls="feature-field",
    )


@rt("/")
def get():
    fields = [feature_field(name, FEATURE_MEANS[i]) for i, name in enumerate(FEATURE_NAMES)]
    return Titled(
        "Breast Cancer Predictor",
        P("Enter the 30 predictors below (pre-filled with dataset averages), then hit Predict."),
        Form(
            Div(*fields, style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem;"),
            Button("Predict", type="submit"),
            hx_post="/predict",
            hx_target="#result",
            hx_swap="innerHTML",
        ),
        Div(id="result"),
    )


@rt("/predict")
async def predict(req):
    form = await req.form()

    # validate every field is present, non-blank, and numeric before predicting
    missing = []
    x = []
    for name in FEATURE_NAMES:
        raw = form.get(name.replace(" ", "_"), "").strip()
        if not raw:
            missing.append(name)
            continue
        try:
            x.append(float(raw))
        except ValueError:
            missing.append(name)

    if missing:
        return Div(
            H3("Missing or invalid values"),
            P("Please fill in a valid number for:"),
            Ul(*[Li(name) for name in missing]),
            id="result",
            style="color: crimson;",
        )

    pred = model.predict([x])[0]
    proba = model.predict_proba([x])[0]
    label = TARGET_NAMES[pred]
    confidence = proba[pred]

    return Div(
        H3(f"Prediction: {label}"),
        P(f"Confidence: {confidence:.1%}"),
        id="result",
    )


serve()