
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal, List
import joblib
import numpy as np

app = FastAPI(title="Phishing Detection API (webOut vs webIn)")

# 모델 로드
model_out = joblib.load("rf_model_webOut01.pkl")
model_in  = joblib.load("rf_model_webIn01.pkl")

# webOut(14개) 스키마
class WebOutFeatures(BaseModel):
    having_IP_Address: int = Field(..., ge=0, le=1)
    URL_Length: int = Field(..., ge=0, le=1)
    Shortining_Service: int = Field(..., ge=0, le=1)
    having_At_Symbol: int = Field(..., ge=0, le=1)
    double_slash_redirecting: int = Field(..., ge=0, le=1)
    Prefix_Suffix: int = Field(..., ge=0, le=1)
    having_Sub_Domain: int = Field(..., ge=0, le=1)
    SSLfinal_State: int = Field(..., ge=0, le=1)
    Domain_registeration_length: int = Field(..., ge=0, le=1)
    Favicon: int = Field(..., ge=0, le=1)
    port: int = Field(..., ge=0, le=1)
    HTTPS_token: int = Field(..., ge=0, le=1)
    age_of_domain: int = Field(..., ge=0, le=1)
    DNSRecord: int = Field(..., ge=0, le=1)

# webIn(25개) 스키마
class WebInFeatures(WebOutFeatures):
    Request_URL: int = Field(..., ge=0, le=1)
    URL_of_Anchor: int = Field(..., ge=0, le=1)
    Links_in_tags: int = Field(..., ge=0, le=1)
    SFH: int = Field(..., ge=0, le=1)
    Submitting_to_email: int = Field(..., ge=0, le=1)
    Abnormal_URL: int = Field(..., ge=0, le=1)
    Redirect: int = Field(..., ge=0, le=1)
    on_mouseover: int = Field(..., ge=0, le=1)
    RightClick: int = Field(..., ge=0, le=1)
    popUpWidnow: int = Field(..., ge=0, le=1)
    Iframe: int = Field(..., ge=0, le=1)

@app.get("/")
def health_check():
    return {"status":"ok","models":["webOut","webIn"]}

@app.post("/predict")
def predict(
    mode: Literal["webOut","webIn"] = Query(..., description="webOut=14, webIn=25"),
    features: dict = None
):
    try:
        if mode=="webOut":
            data = WebOutFeatures(**features)
            m = model_out
            keys = list(WebOutFeatures.__fields__.keys())
        else:
            data = WebInFeatures(**features)
            m = model_in
            keys = list(WebInFeatures.__fields__.keys())
    except Exception as e:
        raise HTTPException(422, detail=str(e))

    x = np.array([[ getattr(data, k) for k in keys ]])
    try:
        pred = m.predict(x)[0]
        proba = m.predict_proba(x)[0].tolist()
    except Exception as e:
        raise HTTPException(500, detail=f"prediction error: {e}")

    return {"mode":mode, "prediction":int(pred), "probabilities":proba}
