from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import numpy as np
from math import log, sqrt, exp, erf


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# -----------------------------
# Utility: Black-Scholes pricing
# -----------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call_price(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


def bs_vega(S: float, K: float, r: float, T: float, sigma: float) -> float:
    # Vega = S * sqrt(T) * N'(d1), where N' is standard normal pdf
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    # standard normal pdf
    pdf = (1.0 / sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1 * d1)
    return S * sqrt(T) * pdf


# -----------------------------
# Pydantic models
# -----------------------------

class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatusCheckCreate(BaseModel):
    client_name: str


class SimulateRequest(BaseModel):
    n: int = 200
    S0: float = 100.0
    mu: float = 0.05
    sigma_true: float = 0.20
    K: Optional[float] = None  # defaults to S0 if None
    r: float = 0.02  # aligns with choice 2A
    T: float = 0.25  # 3 months
    dt: float = 1.0 / 252.0
    obs_noise_std: float = 0.5
    seed: Optional[int] = 42
    save: bool = True  # respects choice 3A by default


class SimulateResponse(BaseModel):
    run_id: str
    n: int
    params: Dict[str, Any]
    S: List[float]
    true_vol: List[float]
    call_price_clean: List[float]
    call_price_obs: List[float]


class FitRequest(BaseModel):
    run_id: str
    sigma_init: float = 0.20
    process_var: float = 1e-4
    meas_var: Optional[float] = None  # if None, use (obs_noise_std)^2 from run


class FitResponse(BaseModel):
    run_id: str
    n: int
    est_vol: List[float]
    kalman_gain: List[float]
    residuals: List[float]
    call_price_est: List[float]
    params: Dict[str, Any]


# -----------------------------
# Baseline routes
# -----------------------------

@api_router.get("/")
async def root():
    return {"message": "Hello World"}


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]


# -----------------------------
# Kalman Filter - Synthetic Pipeline
# -----------------------------

@api_router.post("/kalman/simulate", response_model=SimulateResponse)
async def simulate_series(req: SimulateRequest):
    n = req.n
    S0 = req.S0
    mu = req.mu
    sigma_true = req.sigma_true
    K = req.K if req.K is not None else S0
    r = req.r
    T = req.T
    dt = req.dt
    obs_noise_std = req.obs_noise_std
    seed = req.seed

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Simulate GBM for spot
    S = np.zeros(n)
    S[0] = S0
    for t in range(1, n):
        z = rng.standard_normal()
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma_true ** 2) * dt + sigma_true * np.sqrt(dt) * z)

    # True call prices (clean)
    call_clean = np.array([bs_call_price(float(S[t]), K, r, T, sigma_true) for t in range(n)])
    # Observed prices with additive noise
    noise = rng.normal(0.0, obs_noise_std, size=n)
    call_obs = call_clean + noise

    run_id = str(uuid.uuid4())
    params = {
        "S0": S0,
        "mu": mu,
        "sigma_true": sigma_true,
        "K": K,
        "r": r,
        "T": T,
        "dt": dt,
        "obs_noise_std": obs_noise_std,
        "seed": seed,
    }

    doc = {
        "id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "params": params,
        "series": {
            "S": S.tolist(),
            "true_vol": [sigma_true] * n,
            "call_price_clean": call_clean.tolist(),
            "call_price_obs": call_obs.tolist(),
        },
    }

    if req.save:
        await db.kalman_runs.insert_one(doc)

    return SimulateResponse(
        run_id=run_id,
        n=n,
        params=params,
        S=S.tolist(),
        true_vol=[sigma_true] * n,
        call_price_clean=call_clean.tolist(),
        call_price_obs=call_obs.tolist(),
    )


@api_router.get("/kalman/run/{run_id}")
async def get_run(run_id: str):
    doc = await db.kalman_runs.find_one({"id": run_id})
    if not doc:
        return {"error": "run_id not found"}
    return doc


@api_router.post("/kalman/fit", response_model=FitResponse)
async def kalman_fit(req: FitRequest):
    # Fetch run
    doc = await db.kalman_runs.find_one({"id": req.run_id})
    if not doc:
        # For resilience: return a nice message rather than 500
        return FitResponse(
            run_id=req.run_id,
            n=0,
            est_vol=[],
            kalman_gain=[],
            residuals=[],
            call_price_est=[],
            params={"error": "run_id not found"},
        )

    series = doc["series"]
    params = doc["params"]
    S = np.array(series["S"], dtype=float)
    call_obs = np.array(series["call_price_obs"], dtype=float)
    n = len(S)

    K = float(params["K"])  # strike
    r = float(params["r"])  # rate
    T = float(params["T"])  # keep constant maturity for demo

    # EKF setup in log-vol space: x = log(sigma)
    sigma_init = max(req.sigma_init, 1e-6)
    x = log(sigma_init)
    P = 0.04  # initial variance for state (vol) uncertainty

    q = req.process_var  # process variance
    if req.meas_var is None:
        R = float(params.get("obs_noise_std", 0.5)) ** 2
    else:
        R = req.meas_var

    est_vol = []
    kalman_gain = []
    residuals = []
    call_price_est = []

    for t in range(n):
        # Predict
        x_pred = x  # random walk
        P_pred = P + q

        sigma_pred = max(exp(x_pred), 1e-8)
        # Non-linear measurement h(x) = BS_call(S_t, K, r, T, sigma)
        h = bs_call_price(S[t], K, r, T, sigma_pred)
        # Jacobian H = d h / d x = vega * d sigma / d x = vega * exp(x)
        vega = bs_vega(S[t], K, r, T, sigma_pred)
        H = vega * sigma_pred  # since d sigma / d x = exp(x) = sigma

        # Innovation
        y = call_obs[t] - h
        S_cov = H * P_pred * H + R
        K_gain = P_pred * H / S_cov if S_cov > 1e-12 else 0.0

        # Update
        x = x_pred + K_gain * y
        P = (1.0 - K_gain * H) * P_pred

        sigma_upd = max(exp(x), 1e-8)
        est_vol.append(float(sigma_upd))
        kalman_gain.append(float(K_gain))
        residuals.append(float(y))
        call_price_est.append(float(bs_call_price(S[t], K, r, T, sigma_upd)))

    # Save results back to run document
    update_doc = {
        "results": {
            "est_vol": est_vol,
            "kalman_gain": kalman_gain,
            "residuals": residuals,
            "call_price_est": call_price_est,
            "fitted_at": datetime.utcnow().isoformat(),
            "config": {
                "sigma_init": sigma_init,
                "process_var": q,
                "meas_var": R,
            },
        }
    }
    await db.kalman_runs.update_one({"id": req.run_id}, {"$set": update_doc})

    out_params = dict(params)
    out_params.update(update_doc["results"]["config"])  # include fit config

    return FitResponse(
        run_id=req.run_id,
        n=n,
        est_vol=est_vol,
        kalman_gain=kalman_gain,
        residuals=residuals,
        call_price_est=call_price_est,
        params=out_params,
    )


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()