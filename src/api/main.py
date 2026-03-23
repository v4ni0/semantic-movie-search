import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


from src.recommender import MovieRecommender
from src.config import TOP_K

class RecommendationRequest(BaseModel):
    description: str
    top_k: int = TOP_K

class RecommendationResponse(BaseModel):
    id: int
    title: str
    score: float

app = FastAPI(
    title="Movie Recommendation System API",
    description="An API for recommending movies based on user description.",
)

recommender = MovieRecommender()

@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    recommendations = recommender.recommend(request.description, request.top_k)
    return {"recommendations": recommendations}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
