"""
FastAPI Bridge Service - Exposes Python ML models as REST API
Bridges sizing_pipeline.py with NestJS backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn

# Import our existing AR sizing system
from .sizing_pipeline import (
    SizingPipeline, 
    BodyMeasurements,
    GarmentSpec,
    FitDecision
)
from .style_recommender import StyleRecommender, BodyMeasurementProfile
from .sku_learning_system import SKUCorrectionLearner, SKUCorrection
from .multi_garment_system import LayerManager, ProductCategory, LayerType
import numpy as np

app = FastAPI(title="Chic India ML Service", version="1.0.0")

# CORS for backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:19006"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
style_recommender = StyleRecommender()
sku_learner = SKUCorrectionLearner()

# Load SKU corrections
try:
    sku_learner.load()
    print("✓ Loaded SKU corrections")
except:
    print("⚠ No SKU corrections found")


# Request/Response models
class MeasurementInput(BaseModel):
    shoulder_width_cm: float
    chest_width_cm: float
    torso_length_cm: float
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    inseam_cm: Optional[float] = None


class GarmentSpecInput(BaseModel):
    shoulder_width_cm: Optional[float] = None
    chest_width_cm: Optional[float] = None
    length_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    inseam_cm: Optional[float] = None
    tight_tolerance: float = 1.0
    loose_tolerance: float = 2.0


class SKUCorrectionInput(BaseModel):
    shoulder_bias_cm: float = 0.0
    chest_bias_cm: float = 0.0
    waist_bias_cm: float = 0.0
    tight_tolerance: Optional[float] = None
    loose_tolerance: Optional[float] = None


class FitPredictionRequest(BaseModel):
    measurements: MeasurementInput
    garment_specs: GarmentSpecInput
    sku_correction: Optional[SKUCorrectionInput] = None
    product_category: str = "SHIRT"


class FitPredictionResponse(BaseModel):
    decision: str
    confidence: float
    details: Dict
    style_recommendations: List[Dict]


@app.get("/")
def root():
    return {
        "service": "Chic India ML Service",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "fit_prediction": "/predict-fit",
            "fit_size": "/fit-size",
            "style_recommendations": "/style-recommendations",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "sku_corrections_loaded": len(sku_learner.corrections)}


@app.post("/predict-fit", response_model=FitPredictionResponse)
def predict_fit(request: FitPredictionRequest):
    """
    Main fit prediction endpoint
    Takes measurements + garment specs, returns fit decision
    """
    try:
        # 1. Convert measurements
        meas = request.measurements
        specs = request.garment_specs
        
        # 2. Determine fit for each dimension
        decisions = {}
        
        # Upper body fit
        if specs.shoulder_width_cm:
            shoulder_diff = meas.shoulder_width_cm - specs.shoulder_width_cm
            if shoulder_diff < -specs.tight_tolerance:
                decisions['shoulder'] = 'TIGHT'
            elif shoulder_diff > specs.loose_tolerance:
                decisions['shoulder'] = 'LOOSE'
            else:
                decisions['shoulder'] = 'GOOD'
        
        if specs.chest_width_cm:
            chest_diff = meas.chest_width_cm - specs.chest_width_cm
            if chest_diff < -specs.tight_tolerance:
                decisions['chest'] = 'TIGHT'
            elif chest_diff > specs.loose_tolerance:
                decisions['chest'] = 'LOOSE'
            else:
                decisions['chest'] = 'GOOD'
        
        # Lower body fit
        if specs.waist_cm and meas.waist_cm:
            waist_diff = meas.waist_cm - specs.waist_cm
            if waist_diff < -specs.tight_tolerance:
                decisions['waist'] = 'TIGHT'
            elif waist_diff > specs.loose_tolerance:
                decisions['waist'] = 'LOOSE'
            else:
                decisions['waist'] = 'GOOD'
        
        if specs.hip_cm and meas.hip_cm:
            hip_diff = meas.hip_cm - specs.hip_cm
            if hip_diff < -specs.tight_tolerance:
                decisions['hip'] = 'TIGHT'
            elif hip_diff > specs.loose_tolerance:
                decisions['hip'] = 'LOOSE'
            else:
                decisions['hip'] = 'GOOD'
        
        # 3. Apply SKU correction if available
        if request.sku_correction:
            # Adjust thresholds based on learned bias
            # This is where PHASE 2 learning improves accuracy
            pass
        
        # 4. Overall decision (worst case across dimensions)
        if 'TIGHT' in decisions.values():
            overall_decision = 'TIGHT'
            confidence = 0.75
        elif 'LOOSE' in decisions.values():
            overall_decision = 'LOOSE'
            confidence = 0.75
        else:
            overall_decision = 'GOOD'
            confidence = 0.85
        
        # 5. Get style recommendations
        profile = BodyMeasurementProfile(
            shoulder_width_cm=meas.shoulder_width_cm,
            chest_width_cm=meas.chest_width_cm,
            torso_length_cm=meas.torso_length_cm,
            waist_cm=meas.waist_cm,
            hip_cm=meas.hip_cm,
            inseam_cm=meas.inseam_cm
        )
        
        style_recs = style_recommender.recommend(profile)
        
        return FitPredictionResponse(
            decision=overall_decision,
            confidence=confidence,
            details=decisions,
            style_recommendations=[
                {
                    "category": rec.category,
                    "subcategory": rec.subcategory,
                    "reason": rec.reason,
                    "color": rec.color_suggestion.value,
                    "confidence": rec.confidence
                }
                for rec in style_recs[:3]
            ]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/style-recommendations")
def get_style_recommendations(measurements: MeasurementInput):
    """Get personalized style recommendations based on body measurements"""
    try:
        profile = BodyMeasurementProfile(
            shoulder_width_cm=measurements.shoulder_width_cm,
            chest_width_cm=measurements.chest_width_cm,
            torso_length_cm=measurements.torso_length_cm,
            waist_cm=measurements.waist_cm,
            hip_cm=measurements.hip_cm,
            inseam_cm=measurements.inseam_cm
        )
        
        recommendations = style_recommender.recommend(profile)
        
        return {
            "body_shape": profile.body_shape if hasattr(profile, 'body_shape') else "unknown",  # type: ignore
            "recommendations": [
                {
                    "category": rec.category,
                    "subcategory": rec.subcategory,
                    "reason": rec.reason,
                    "color": rec.color_suggestion.value,
                    "confidence": rec.confidence
                }
                for rec in recommendations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style recommendation failed: {str(e)}")


# ---------------------------------------------------------------------------
# FitEngine — POST /fit-size
# ---------------------------------------------------------------------------

class FitSizeRequest(BaseModel):
    front_image_b64: str          # base64-encoded JPEG/PNG, front view
    side_image_b64: str           # base64-encoded JPEG/PNG, side view
    height_cm: float              # user-entered height in cm
    brand: str = "generic"        # size-chart brand key
    session_id: Optional[str] = None  # PilotLogger correlation ID


class FitSizeResponse(BaseModel):
    collar: str
    jacket: str
    trouser: str
    confidence: str               # "heuristic" | "High" | "Medium" | "Low"
    alignment_ok: bool
    alignment_hint: Optional[str]
    session_id: str


_fit_engine_pipeline = None


def _get_fit_engine(brand: str = "generic"):
    """Lazy-initialise FitEnginePipeline (one per worker process)."""
    global _fit_engine_pipeline
    if _fit_engine_pipeline is None:
        # Deferred import so existing /predict-fit path is unaffected if
        # fitengine optional deps (rtmlib, onnxruntime, …) are not installed.
        from fitengine.pipeline import FitEnginePipeline
        _fit_engine_pipeline = FitEnginePipeline(chart=brand)
    return _fit_engine_pipeline


@app.post("/fit-size", response_model=FitSizeResponse)
def fit_size(request: FitSizeRequest):
    """
    FitEngine size recommendation endpoint (Phase 1 — heuristic).

    Accepts two base64-encoded images (front + side view), user height,
    and an optional brand key.  Returns collar / jacket / trouser size
    recommendation with confidence label and alignment status.

    This endpoint is independent of /predict-fit; the NestJS backend
    can call both during migration without touching existing code.
    """
    import base64
    import uuid
    import numpy as np
    import cv2

    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Decode images -------------------------------------------------------
        def _b64_to_bgr(b64: str) -> np.ndarray:
            data = base64.b64decode(b64)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            return img

        front_bgr = _b64_to_bgr(request.front_image_b64)
        side_bgr = _b64_to_bgr(request.side_image_b64)

        # Run FitEnginePipeline -----------------------------------------------
        pipeline = _get_fit_engine(request.brand)
        result, alignment_ok, alignment_hint = pipeline.predict(
            front_bgr=front_bgr,
            side_bgr=side_bgr,
            height_cm=request.height_cm,
            session_id=session_id,
            skip_alignment_check=False,
        )

        return FitSizeResponse(
            collar=result["collar"],
            jacket=result["jacket"],
            trouser=result["trouser"],
            confidence=result.get("confidence", "heuristic"),
            alignment_ok=alignment_ok,
            alignment_hint=alignment_hint,
            session_id=session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fit-size failed: {str(e)}")


# ---------------------------------------------------------------------------
# End FitEngine block
# ---------------------------------------------------------------------------


@app.post("/train-sku-corrections")
def train_sku_corrections(logs: List[Dict]):
    """
    Train SKU corrections from measurement logs
    Called periodically by backend (PHASE 2)
    """
    try:
        # This will be called by a scheduled job
        # Trains corrections from accumulated fit feedback
        correction_count = len(logs)
        
        return {
            "status": "training_complete",
            "logs_processed": correction_count,
            "corrections_learned": 0  # Placeholder
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CHIC INDIA ML SERVICE")
    print("="*60)
    print("Starting FastAPI server...")
    print("Endpoints available at http://localhost:8000")
    print("Documentation at http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
