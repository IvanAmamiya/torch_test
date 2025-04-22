from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI ML Service"}

@app.get("/health")
def health_check():
    return {"status": "ok"}