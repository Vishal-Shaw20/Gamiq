from fastapi import FastAPI
from recommender.api import router as recommender_router

app = FastAPI()

app.include_router(recommender_router)
