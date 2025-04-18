

# Define the /recommend endpoint
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Load necessary objects (scaler, model, label encoders)
model = joblib.load('course_recommender.pkl')  # Load your trained model
scaler = joblib.load('scaler.pkl')  # Load your StandardScaler
label_encoders_subject = joblib.load('label_encoders_subject.pkl')
label_encoders_level = joblib.load('label_encoders_level.pkl')

# Load your dataset (assuming you have a CSV or other source)
df = pd.read_csv("udemy_courses.csv")  # Replace with actual dataset loading

# Define the request and response models for FastAPI
class UserRequest(BaseModel):
    subject: str
    level: str
    price: float

class CourseRecommendation(BaseModel):
    courseTitle: str
    price: float
    numSubscribers: int

# FastAPI endpoint for recommending courses
@app.post("/recommend")
def recommend_courses(req: UserRequest):
    try:
        # Prepare input features
        level_encoded = label_encoders_level.transform([req.level])[0]

        default_is_paid = 1 if req.price > 0 else 0
        avg_subscribers = df["num_subscribers"].mean()
        avg_reviews = df["num_reviews"].mean()
        avg_lectures = df["num_lectures"].mean()
        avg_duration = df["content_duration"].mean()

        X = np.array([[default_is_paid, req.price, avg_subscribers,
                       avg_reviews, avg_lectures, level_encoded, avg_duration]])
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)
        recommended_subject = req.subject


        # 🧠 Filter top 5 courses for predicted subject
        top_courses = (
            df[df["subject"] == recommended_subject]
            .sort_values(by="num_subscribers", ascending=False)
            .head(6)
        )

        # 🧾 Build response
        recommendations = [
            {
                "courseTitle": row["course_title"],
                "price": row["price"],
                "numSubscribers": int(row["num_subscribers"]),
            }
            for _, row in top_courses.iterrows()
        ]

        return {"recommended_courses": recommendations}

    except Exception as e:
        print(f"❌ Error in recommendation process: {e}")
        return {"error": str(e)}