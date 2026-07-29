import uvicorn
from fastapi import FastAPI
from src.router.router import router   

app = FastAPI(title="CXR Multi-label API")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7008)