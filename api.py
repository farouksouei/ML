# api.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import uvicorn
from datetime import datetime
import pickle
import numpy as np
# Import the model class from inference.py
from inference import SkillExchangeModel

# Initialize the model
skill_model = SkillExchangeModel()

# Create FastAPI app
app = FastAPI(
    title="Skill Exchange Platform API",
    description="API for user skill clustering and recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define Pydantic models for request/response validation
class UserData(BaseModel):
    user_id: Optional[str] = Field(None, description="User identifier")
    joinedDate: Optional[str] = Field(None, description="Date when user joined (YYYY-MM-DD)")
    joinedCourses: Optional[str] = Field(None, description="Comma-separated list of courses")
    skills: Optional[str] = Field(None, description="Comma-separated list of user skills")
    desired_skills: Optional[str] = Field(None, description="Comma-separated list of skills user wants to learn")
    isVerified: Optional[Union[bool, str]] = Field(None, description="Whether user is verified")
    engagement_level: Optional[Union[str, int]] = Field(None,
                                                        description="User engagement level (Low/Medium/High or 0/1/2)")
    peer_user_id: Optional[str] = Field(None, description="Peer user identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "new_user_123",
                "joinedDate": "2023-10-15",
                "joinedCourses": "Python, Data Science, Machine Learning",
                "skills": "HTML, CSS, JavaScript, Python",
                "desired_skills": "Machine Learning, AI, Blockchain",
                "isVerified": True,
                "engagement_level": "Medium"
            }
        }


class SimilarUser(BaseModel):
    user_id: str
    match_score: float
    skills: List[str]


class PredictionResponse(BaseModel):
    cluster: int
    confidence: float
    cluster_description: str
    recommended_skills: Optional[List[str]] = None
    skill_scores: Optional[Dict[str, float]] = None
    similar_users: Optional[List[SimilarUser]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool


# Sample data based on the dataset provided
SAMPLE_USERS = [
    {
        "user_id": "1",
        "joinedDate": "2022-08-28",
        "joinedCourses": "Machine Learning, CSS, Excel, SQL, HTML",
        "skills": "HTML, SQL",
        "desired_skills": "CSS, Java, Machine Learning, Blockchain, Data Science",
        "isVerified": False,
        "engagement_level": "Medium",
        "peer_user_id": "3383"
    },
    {
        "user_id": "2",
        "joinedDate": "2023-12-04",
        "joinedCourses": "Data Science, Excel, Python, JavaScript",
        "skills": "HTML, CSS, JavaScript, Excel, SQL",
        "desired_skills": "JavaScript, Python, Java, Node.js, AI",
        "isVerified": True,
        "engagement_level": "Medium",
        "peer_user_id": "5095"
    },
    {
        "user_id": "25",
        "joinedDate": "2023-10-14",
        "joinedCourses": "JavaScript, CSS, Node.js, Machine Learning",
        "skills": "HTML, CSS, Excel",
        "desired_skills": "HTML, CSS, JavaScript, Python, Node.js, Machine Learning, AI, Data Science",
        "isVerified": False,
        "engagement_level": "High",
        "peer_user_id": "9829"
    }
]


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check if the API and model are working properly
    """
    return {
        "status": "ok" if skill_model.kmeans_model is not None else "model_error",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": skill_model.kmeans_model is not None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
        user_data: UserData = Body(
            ...,
            description="User data for prediction",
            examples={
                "new_user": {
                    "summary": "New user with web development background",
                    "description": "A user with frontend development skills looking to expand into modern frameworks",
                    "value": {
                        "joinedDate": "2023-11-30",
                        "joinedCourses": "JavaScript, CSS, HTML",
                        "skills": "HTML, CSS, JavaScript",
                        "desired_skills": "React, Node.js, MongoDB",
                        "isVerified": True,
                        "engagement_level": "Medium"
                    }
                },
                "data_scientist": {
                    "summary": "Data scientist user",
                    "description": "An experienced data professional looking to expand AI skills",
                    "value": {
                        "joinedDate": "2023-05-15",
                        "joinedCourses": "Python, Data Science, Machine Learning, AI",
                        "skills": "Python, SQL, Data Science, Statistics",
                        "desired_skills": "Deep Learning, NLP, Computer Vision",
                        "isVerified": True,
                        "engagement_level": "High"
                    }
                },
                "beginner": {
                    "summary": "Beginner with minimal skills",
                    "description": "A new user just starting their tech journey",
                    "value": {
                        "joinedDate": "2024-01-10",
                        "joinedCourses": "Excel",
                        "skills": "Excel",
                        "desired_skills": "HTML, CSS, JavaScript, Python",
                        "isVerified": False,
                        "engagement_level": "Low"
                    }
                },
                "sample_from_dataset": {
                    "summary": "Sample from actual dataset",
                    "description": "A real user from the training data",
                    "value": SAMPLE_USERS[0]
                }
            }
        )
):
    """
    Predict cluster and skill recommendations for a user
    """
    # Convert Pydantic model to dict
    user_dict = user_data.dict(exclude_none=True)

    if not user_dict:
        raise HTTPException(status_code=400, detail="No user data provided")

    # Make prediction
    try:
        result = skill_model.predict(user_dict)

        # If there was an error in prediction
        if "error" in result:
            return {
                "cluster": -1,
                "confidence": 0.0,
                "cluster_description": "Error",
                "error": result["error"]
            }

        # Find similar users
        similar_users = skill_model.find_similar_users(user_dict)
        result["similar_users"] = similar_users

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/sample-users", tags=["Sample Data"])
async def get_sample_users():
    """
    Get sample user data for testing the API
    """
    return {"samples": SAMPLE_USERS}


@app.get("/clusters", tags=["Information"])
async def get_clusters():
    """
    Get information about all possible clusters
    """
    clusters = {}

    # Get descriptions for all potential clusters
    for i in range(10):  # Assuming we have up to 10 clusters
        desc = skill_model.get_cluster_description(i)
        if "Group of users" not in desc:  # Skip generic descriptions
            clusters[i] = desc

    return {"clusters": clusters}


@app.get("/skills", tags=["Information"])
async def get_skills():
    """
    Get list of all skills that can be recommended
    """
    if skill_model.knn_models:
        skills = list(skill_model.knn_models.keys())
        return {"skills": skills}
    else:
        return {"skills": []}


@app.post("/batch-predict", tags=["Predictions"])
async def batch_predict(
    users: List[UserData] = Body(
        ...,
        description="Batch of user data for prediction",
        examples={
            "web_developers": {
                "summary": "A batch of web developers",
                "description": "Two web developers with different skill levels",
                "value": [
                    {
                        "user_id": "1",
                        "joinedDate": "2022-08-28",
                        "joinedCourses": "HTML, CSS, JavaScript",
                        "skills": "HTML, CSS",
                        "desired_skills": "JavaScript, React, Node.js",
                        "isVerified": False,
                        "engagement_level": "Medium"
                    },
                    {
                        "user_id": "2",
                        "joinedDate": "2023-12-04",
                        "joinedCourses": "JavaScript, React, Node.js",
                        "skills": "HTML, CSS, JavaScript, React",
                        "desired_skills": "TypeScript, Angular, Vue",
                        "isVerified": True,
                        "engagement_level": "High"
                    }
                ]
            },
            "sample_from_dataset": {
                "summary": "Samples from actual dataset",
                "description": "Two real users from the training data",
                "value": SAMPLE_USERS[:2]
            }
        }
    )
):
    """
    Make predictions for multiple users at once
    """
    results = []

    for user_data in users:
        user_dict = user_data.dict(exclude_none=True)
        try:
            prediction = skill_model.predict(user_dict)
            # Add user_id to the result if it exists
            if "user_id" in user_dict:
                prediction["user_id"] = user_dict["user_id"]
            results.append(prediction)
        except Exception as e:
            results.append({
                "user_id": user_dict.get("user_id", "unknown"),
                "error": str(e),
                "cluster": -1,
                "confidence": 0.0
            })

    return {"predictions": results}


class EngagementPredictionResponse(BaseModel):
    engagement_level: str
    engagement_probabilities: Optional[Dict[str, float]] = None
    feature_values: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.post("/predict-engagement", response_model=EngagementPredictionResponse, tags=["Predictions"])
async def predict_engagement(
    user_data: UserData = Body(
        ...,
        description="User data for engagement prediction",
        examples={
            "new_user": {
                "summary": "New user with some skills",
                "description": "A relatively new user with basic web development skills",
                "value": {
                    "joinedDate": "2023-11-30",
                    "joinedCourses": "JavaScript, CSS, HTML",
                    "skills": "HTML, CSS, JavaScript",
                    "desired_skills": "React, Node.js, MongoDB",
                    "isVerified": True
                }
            },
            "active_user": {
                "summary": "Very active user with many skills",
                "description": "A highly engaged user with diverse skill set",
                "value": {
                    "joinedDate": "2023-05-15",
                    "joinedCourses": "Python, Data Science, Machine Learning, AI, JavaScript",
                    "skills": "Python, JavaScript, HTML, CSS, SQL, Data Science",
                    "desired_skills": "Deep Learning, NLP, Computer Vision",
                    "isVerified": True
                }
            },
            "minimal_user": {
                "summary": "User with minimal activity",
                "description": "A user who has just started on the platform",
                "value": {
                    "joinedDate": "2024-01-10",
                    "joinedCourses": "Excel",
                    "skills": "Excel",
                    "desired_skills": "HTML, CSS",
                    "isVerified": False
                }
            }
        }
    )
):
    """
    Predict the engagement level of a user based on their skills, courses, and other data
    """
    # Convert Pydantic model to dict
    user_dict = user_data.dict(exclude_none=True)

    if not user_dict:
        raise HTTPException(status_code=400, detail="No user data provided")

    # Make engagement prediction
    try:
        result = skill_model.predict_engagement(user_dict)

        # If there was an error in prediction
        if "error" in result:
            return {
                "engagement_level": "Unknown",
                "error": result["error"]
            }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engagement prediction error: {str(e)}")


# Add this class to the top with other models in api.py
class CourseRecommendationRequest(BaseModel):
    skills: str = Field(..., description="Comma-separated list of user's current skills")
    top_n: Optional[int] = Field(5, description="Number of recommendations to return")


class CourseRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    error: Optional[str] = None


# Load course recommendation model
try:
    with open('models/course_recommendation_xgboost_model.pkl', 'rb') as f:
        course_recommendation_model = pickle.load(f)
    with open('models/course_recommendation_features.pkl', 'rb') as f:
        course_recommendation_features = pickle.load(f)
    print("Course recommendation model loaded successfully")
except Exception as e:
    print(f"Error loading course recommendation model: {e}")
    course_recommendation_model = None
    course_recommendation_features = None


# Add this endpoint
@app.post("/recommend-courses", response_model=CourseRecommendationResponse, tags=["Recommendations"])
async def recommend_courses(
        request: CourseRecommendationRequest = Body(
            ...,
            description="Request for course recommendations",
            examples={
                "web_developer": {
                    "summary": "Web Developer Skills",
                    "description": "A user with basic web development skills",
                    "value": {
                        "skills": "HTML, CSS, JavaScript",
                        "top_n": 5
                    }
                },
                "data_scientist": {
                    "summary": "Data Scientist Skills",
                    "description": "A user with data science skills",
                    "value": {
                        "skills": "Python, SQL, Statistics, Machine Learning",
                        "top_n": 3
                    }
                }
            }
        )
):
    """Recommend new courses based on a user's current skills"""
    if not course_recommendation_model or not course_recommendation_features:
        return {
            "recommendations": [],
            "error": "Course recommendation model not available"
        }

    try:
        # Parse skills
        user_skills = [s.strip() for s in request.skills.split(',') if s.strip()]

        if not user_skills:
            return {
                "recommendations": [],
                "error": "No valid skills provided"
            }

        # Create one-hot encoding for user skills
        feature_names = course_recommendation_features['feature_names']
        target_names = course_recommendation_features['target_names']

        user_vector = np.zeros(len(feature_names))

        # Map user skills to features
        for skill in user_skills:
            for i, feature in enumerate(feature_names):
                if feature.lower() == skill.lower():
                    user_vector[i] = 1
                    break

        # Reshape for prediction
        user_vector = user_vector.reshape(1, -1)

        # Make prediction
        predictions = course_recommendation_model.predict_proba(user_vector)

        # Process predictions
        recommendations = []
        for i, class_probs in enumerate(predictions):
            # Get probability of positive class
            if len(class_probs[0]) > 1:
                prob = class_probs[0][1]
            else:
                prob = class_probs[0][0]

            skill_name = target_names[i]

            # Only include skills that are not already known
            if prob > 0.2 and skill_name.lower() not in [s.lower() for s in user_skills]:
                recommendations.append({
                    "skill": skill_name,
                    "confidence": round(float(prob), 4)
                })

        # Sort by confidence and limit to top_n
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        recommendations = recommendations[:request.top_n]

        return {
            "recommendations": recommendations
        }

    except Exception as e:
        import traceback
        return {
            "recommendations": [],
            "error": f"Error generating recommendations: {str(e)}",
            "traceback": traceback.format_exc()
        }
def start():
    """Start the API server"""
    uvicorn.run("api:app", host="127.0.0.1", port=9100, reload=True)


if __name__ == "__main__":
    start()