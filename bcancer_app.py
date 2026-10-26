from fasthtml.common import *
from sklearn.datasets import load_breast_cancer
import joblib

MODEL_PATH = "bcancer_model.joblib"
model = joblib.load(MODEL_PATH)

data = load_breast_cancer()
FEATURE_NAMES = data.feature_names  # 30 features, in the order the model expects
FEATURE_MEANS = data.data.mean(axis=0)  # used as handy default values
TARGET_NAMES = data.target_names  # ["malignant", "benign"]

app, rt = fast_app()


def field_id(name):
    return name.replace(" ", "_")


def predictor_form():
    fields = [
        Div(
            Label(name.title(), fr=field_id(name)),
            Input(
                type="number",
                step="any",
                name=field_id(name),
                id=field_id(name),
                value=f"{mean:.4f}",
                required=True,
            ),
            cls="field",
        )
        for name, mean in zip(FEATURE_NAMES, FEATURE_MEANS)
    ]
    return Form(
        Div(*fields, style="display:grid; grid-template-columns: repeat(3, 1fr); gap:0.5rem;"),
        Button("Predict", type="submit", style="margin-top:1rem;"),
        id="predict-form",
        hx_post="/predict",
        hx_target="#result",
        hx_swap="innerHTML",
    )


@rt("/")
def get():
    return Titled(
        "Breast Cancer Predictor",
        P("Enter the measurements below (defaults are dataset averages) and press Predict."),
        predictor_form(),
        Div(id="result"),
    )


@rt("/predict")
async def post(request):
    form = await request.form()

    values = []
    missing = []
    invalid = []
    for name in FEATURE_NAMES:
        raw = form.get(field_id(name), "")
        raw = raw.strip() if isinstance(raw, str) else ""
        if not raw:
            missing.append(name.title())
            continue
        try:
            values.append(float(raw))
        except ValueError:
            invalid.append(name.title())

    if missing or invalid:
        messages = []
        if missing:
            messages.append(P(f"Missing values for: {', '.join(missing)}."))
        if invalid:
            messages.append(P(f"Invalid (non-numeric) values for: {', '.join(invalid)}."))
        return Div(*messages, style="color:red;")

    prediction = model.predict([values])[0]
    probability = model.predict_proba([values])[0]

    label = TARGET_NAMES[prediction]
    confidence = probability[prediction] * 100

    return Div(
        H3(f"Prediction: {label.title()}"),
        P(f"Confidence: {confidence:.1f}%"),
    )


serve()