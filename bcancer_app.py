import joblib
from fasthtml.common import *
from sklearn.datasets import load_breast_cancer

MODEL_PATH = "bcancer_model.joblib"

model = joblib.load(MODEL_PATH)
dataset = load_breast_cancer()
feature_names = dataset.feature_names
target_names = dataset.target_names

# Dataset means, used as starting values so the form is quick to try out.
DEFAULTS = [
    14.127, 19.29, 91.969, 654.889, 0.096, 0.104, 0.089, 0.049, 0.181, 0.063,
    0.405, 1.217, 2.866, 40.337, 0.007, 0.025, 0.032, 0.012, 0.021, 0.004,
    16.269, 25.677, 107.261, 880.583, 0.132, 0.254, 0.272, 0.115, 0.29, 0.084,
]

app, rt = fast_app()


def feature_input(name, default):
    field_id = name.replace(" ", "_")
    return Div(
        Label(name, fr=field_id),
        Input(
            type="number",
            step="any",
            name=field_id,
            id=field_id,
            value=str(default),
        ),
    )


@rt('/')
def get():
    return Titled(
        "Breast Cancer Predictor",
        Form(
            *[feature_input(n, d) for n, d in zip(feature_names, DEFAULTS)],
            Button("Predict", type="submit"),
            hx_post='/predict',
            hx_target='#result',
            hx_swap="innerHTML",
        ),
        Div(id='result'),
    )


@rt('/predict')
async def post(req):
    form = await req.form()
    try:
        values = [float(form[n.replace(" ", "_")]) for n in feature_names]
    except (KeyError, ValueError):
        return Div(P("Please fill in all fields with numeric values."), id='result')

    prediction = model.predict([values])[0]
    proba = model.predict_proba([values])[0]
    label = target_names[prediction]

    return Div(
        H3(f"Prediction: {label}"),
        P(f"Confidence: {proba[prediction]:.2%}"),
        id='result',
    )


serve()
