from fastapi import FastAPI
from app.routes import health,verify

app = FastAPI()

# Register health route
app.include_router(health.router)
app.include_router(verify.router)