from dotenv import load_dotenv
load_dotenv()

from app.core.secrets import apply_secrets_to_environment
apply_secrets_to_environment()

from fastapi import FastAPI
from app.api.routes_jobs import router as jobs_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_inference import router as inference_router
from app.api.routes_wandb import router as wandb_router
from app.api.routes_registry import router as registry_router
from app.api.routes_model import router as model_router
from app.core.platform_config import load_platform_config

app = FastAPI(title="FineTuneIT Lite")

app.include_router(jobs_router)
app.include_router(evaluation_router)
app.include_router(inference_router)
app.include_router(wandb_router)
app.include_router(registry_router)
app.include_router(model_router)

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "FineTuneIT Lite",
        "workflow": "train → live metrics → experiment registry → evaluate → inference",
        "platform_config": load_platform_config().get("platform", {})
    }
