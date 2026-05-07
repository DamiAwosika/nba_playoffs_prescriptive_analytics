from fastapi import FastAPI

from api.routes import predict

app = FastAPI(title="NBA Predict", version="0.1.0")
app.include_router(predict.router, prefix="/v1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
