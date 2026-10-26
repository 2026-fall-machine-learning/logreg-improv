import joblib
import numpy as np
from fasthtml.common import *
from sklearn.datasets import load_breast_cancer

app, rt = fast_app()

model = joblib.load("bcancer_model.joblib")

# Pull feature names + a sane default (mean value) for each, just so the
# form isn't 30 empty boxes to fill in by hand.
data = load_breast_cancer()
FEATURE_NAMES = list(data.feature_names)
DEFAULTS = {name: float(val) for name, val in zip(FEATURE_NAMES, data.data.mean(axis=0))}


def field_name(feature: str) -> str:
    """Turn 'mean radius' into a valid form/field name 'mean_radius'."""
    return feature.replace(" ", "_")


def predictor_inputs():
    return [
        Div(
            Label(feature, fr=field_name(feature)),
            Input(
                type="number",
                step="any",
                name=field_name(feature),
                id=field_name(feature),
                value=str(DEFAULTS[feature]),
                required=True,
            ),
            cls="predictor-field",
        )
        for feature in FEATURE_NAMES
    ]


@rt("/")
def get():
    return Titled(
        "Breast Cancer Predictor",
        P("Enter the predictors below (pre-filled with dataset means), then hit Predict."),
        Form(
            *predictor_inputs(),
            Button("Predict", type="submit"),
            hx_post="/predict",
            hx_target="#result",
            hx_swap="innerHTML",
        ),
        Div(id="result"),
    )


@rt("/predict")
async def predict(request):
    form = await request.form()

    # Validate every predictor is present and numeric before we try to
    # predict on it - no silently defaulting missing values.
    values = {}
    invalid = []
    for feature in FEATURE_NAMES:
        raw = form.get(field_name(feature), "").strip()
        try:
            values[feature] = float(raw)
        except ValueError:
            invalid.append(feature)

    if invalid:
        return Div(
            P(Strong("Please provide a numeric value for all predictors.")),
            P("Missing or invalid: " + ", ".join(invalid)),
            cls="error",
        )

    x = [[values[feature] for feature in FEATURE_NAMES]]

    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    label = data.target_names[pred]

    return Div(
        P(Strong(f"Prediction: {label}")),
        P(f"P(malignant)={proba[0]:.3f}  P(benign)={proba[1]:.3f}"),
    )


serve()