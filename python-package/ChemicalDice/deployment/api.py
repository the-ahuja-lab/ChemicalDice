"""
FastAPI Server Application.
Hosts endpoints bridging external `ChemicalDice` Tier 1 calls into live HuggingFace/PyTorch inference streams.
"""

import os
import io
import time
import asyncio
import logging
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Header, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Assume modular Tier 2 dependencies are explicitly available since 'deployment' overlaps closely
try:
    from ChemicalDice.training.gen_model import SmiSsedPredictor
    from rdkit import Chem
except ImportError as e:
    raise ImportError(
        "Deployment tier relies on structural training objects to host predictions. "
        "Make sure to install via `pip install ChemicalDice[deployment]`."
    ) from e

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration Environment ---
class Settings(BaseSettings):
    api_key: str = "ajci8JYskz5FulkeXaczeQmVTYF1cABnP7pdfUFDBgjuCVJZ6R7YjA"
    mamba_dir: str = "/mnt/home/casusers/ajay/Chemical_Dice_Genralised/smi_ssed_model/"
    model_weights: str = "generalised_smi_ssed_model.pth"
    device_str: str = 'cuda' if torch.cuda.is_available() else 'cpu'

settings = Settings()
app = FastAPI(title="ChemicalDice Integrator Cloud API", version="1.0.0")

# --- Global Physics State ---
inference_model = None

# --- Startup Lifecycles ---
@app.on_event("startup")
async def load_model():
    """Mounts the incredibly heavy Mamba state space engine strictly once per server boot."""
    global inference_model
    logger.info("Initializing HuggingFace Smi-SSED Inference Topology...")
    
    if os.path.exists(settings.model_weights) and os.path.exists(settings.mamba_dir):
        try:
            inference_model = SmiSsedPredictor(settings.mamba_dir, target_dim=8192)
            device = torch.device(settings.device_str)
            
            # Mount custom linear state dict mapping explicitly over frozen Mamba encoders
            state_dict = torch.load(settings.model_weights, map_location=device)
            inference_model.load_state_dict(state_dict, strict=False)
            
            inference_model.to(device)
            inference_model.eval()
            logger.info("Generalised Model securely mapped to GPU VRAM and locked into Eval Mode.")
        except Exception as e:
            logger.error(f"Failed to bootstrap Inference topology: {e}")
    else:
        logger.warning("Deployment booted blindly without persistent weights. Inference routes will automatically 503.")


# --- Authentication Dep ---
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        logger.warning(f"Rejected malicious API request. Provided Key: {x_api_key}")
        raise HTTPException(status_code=401, detail="Unauthorized ChemicalDice execution route.")
    return x_api_key

# --- Request Models ---
class SMILESRequest(BaseModel):
    smiles: str

class FeatureResponse(BaseModel):
    smiles: str
    features: list

# --- Endpoint Routes ---

@app.post("/predict", response_model=FeatureResponse)
async def predict_single_smiles(request: SMILESRequest, api_key: str = Depends(verify_api_key)):
    """
    Ingests a raw canonical string and passes it aggressively through frozen State-Space attention.
    """
    global inference_model
    if inference_model is None:
        raise HTTPException(status_code=503, detail="Inference cluster offline. Weights unresolved.")

    smiles = request.smiles
    if not Chem.MolFromSmiles(smiles):
        raise HTTPException(status_code=400, detail="Catastrophic RDKit validation failure on input SMILES.")

    device = torch.device(settings.device_str)
    
    # Pre-Flight tokenizer bounds
    inputs = inference_model.smi_ssed_tokenizer(
        [smiles], 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Real-Time Inference
    with torch.no_grad():
        try:
            output = inference_model(input_ids, attention_mask)
            feature_vector = output.squeeze(0).cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Inference execution fault: {e}")
            raise HTTPException(status_code=500, detail="Computational pipeline collapsed during forward pass.")

    return FeatureResponse(smiles=smiles, features=feature_vector)


@app.post("/stream-features-from-csv")
async def stream_features_endpoint(
    file: UploadFile = File(...), 
    batch_size: int = Form(32),
    api_key: str = Depends(verify_api_key)
):
    """
    Asynchronous CSV streamer designed specifically to pair cleanly with ChemicalDice.core.api_client.
    Pushes 8192-D vectors in literal floating-point byte chunks identically.
    """
    global inference_model
    if inference_model is None:
        raise HTTPException(status_code=503, detail="Inference cluster offline. Weights unresolved.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if 'SMILES' not in df.columns:
            raise HTTPException(status_code=400, detail="Malformed structure mapping. 'SMILES' target missing.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload transmission: {str(e)}")

    smiles_list = df['SMILES'].tolist()
    total_smiles = len(smiles_list)
    logger.info(f"Accepted {total_smiles} molecules for streaming projection mapping.")
    device = torch.device(settings.device_str)

    # Isolated generator for strict chunk-level ASGI yield
    async def feature_generator():
        for i in range(0, total_smiles, batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            current_batch_size = len(batch_smiles)

            inputs = inference_model.smi_ssed_tokenizer(
                batch_smiles, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                output = inference_model(input_ids, attention_mask).cpu().numpy().astype(np.float32)

            # Symmetrical byte-padding enforcing static network structures back to the client
            if current_batch_size < batch_size:
                padding_size = batch_size - current_batch_size
                padding = np.zeros((padding_size, output.shape[1]), dtype=np.float32)
                output = np.vstack([output, padding])

            yield output.tobytes()
            # Explicit ASGI asynchronous yield back to networking I/O buffers
            await asyncio.sleep(0.001)

    return StreamingResponse(feature_generator(), media_type="application/octet-stream")
