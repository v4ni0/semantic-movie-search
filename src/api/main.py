import uvicorn
from fastapi import FastAPI, HTTPException
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
    try:
        recommendations = recommender.recommend(request.description, request.top_k)
        return {"recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
