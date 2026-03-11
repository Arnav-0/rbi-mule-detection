"""FastAPI application for RBI Mule Account Detection."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import RequestLoggingMiddleware
from src.api.dependencies import startup_load_model
from src.api.schemas import HealthResponse
from src.api.routes import predict, account, model, dashboard, fairness, benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_load_model()
    yield


app = FastAPI(
    title="RBI Mule Account Detection API",
    description="ML-powered mule account detection with SHAP explanations and fairness auditing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging
app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(predict.router)
app.include_router(account.router)
app.include_router(model.router)
app.include_router(dashboard.router)
app.include_router(fairness.router)
app.include_router(benchmark.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    from src.api.dependencies import model_service, _db_initialized
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model_service.is_loaded,
        db_connected=_db_initialized,
    )
