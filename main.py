from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI()

model = joblib.load('model (1).pkl')
scaler = joblib.load('scaler (1).pkl')


class PersonSchema(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


@app.post('/predict')
async def predict(person: PersonSchema):
    person_dict = person.dict()
    features = list(person_dict.values())
    scaled = scaler.transform([features])
    pred = model.predict_proba(scaled)[0]
    print(model.predict_proba([features]))
    if pred[1] >= 0.5:
        diagnosis = "болен"
    else:
        diagnosis = "не болен"

    return {"approved": diagnosis, "probability": round(pred[0], 2)}
