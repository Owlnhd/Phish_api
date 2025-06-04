
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests

from feature_extractor import (
    Prefix_Suffix, double_slash_redirecting, having_At_Symbol,
    Shortining_Service, URL_Length, having_IP_Address,
    having_Sub_Domain, SSLfinal_State, Domain_registeration_length,
    age_of_domain, DNSRecord, Favicon, port,
    HTTPS_token, Request_URL, URL_of_Anchor,
    Links_in_tags, SFH, Submitting_to_email,
    Abnormal_URL, Redirect, on_mouseover,
    RightClick, popUpWindow, Iframe
)

app = FastAPI(title="Phishing Detection API (webOut vs webIn)")

# 14개 URL-only 모델
model_out = joblib.load("rf_model_webOut01.pkl")
# 25개 URL+내부 모델
model_in  = joblib.load("rf_model_webIn01.pkl")

# 14개 피처 이름
OUT_COLS = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token",
    "age_of_domain", "DNSRecord"
]

# 25개 피처 이름 (14개 + 11개)
IN_COLS = OUT_COLS + [
    "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
    "Submitting_to_email", "Abnormal_URL", "Redirect", "on_mouseover",
    "RightClick", "popUpWindow", "Iframe"
]

class URLInput(BaseModel):
    url: HttpUrl

@app.get("/")
def health_check():
    return {"status": "ok", "models": ["webOut", "webIn"]}


@app.post("/predict")
def predict(
    inp: URLInput,
    mode: str = Query(..., description="webOut=URL-only(14), webIn=URL+내부(25)")
):
    u = str(inp.url)

    # 1) 한 번만 HTTP GET 요청 + HTML 파싱
    try:
        resp = requests.get(u, timeout=5)
        html = resp.text
        parsed = BeautifulSoup(html, "html.parser")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL 요청/파싱 실패: {e}")

    # 2) URL-only 14개 피처 계산 (사전 형태)
    out_feats = {
        "having_IP_Address": having_IP_Address(u),
        "URL_Length": URL_Length(u),
        "Shortining_Service": Shortining_Service(u),
        "having_At_Symbol": having_At_Symbol(u),
        "double_slash_redirecting": double_slash_redirecting(u),
        "Prefix_Suffix": Prefix_Suffix(u),
        "having_Sub_Domain": having_Sub_Domain(u),
        "SSLfinal_State": SSLfinal_State(u, response=resp),
        "Domain_registeration_length": Domain_registeration_length(u),
        "Favicon": Favicon(u, parsed=parsed),
        "port": port(u),
        "HTTPS_token": HTTPS_token(u),
        "age_of_domain": age_of_domain(u),
        "DNSRecord": DNSRecord(u)
    }

    # 3) URL+내부 25개 피처 계산 (사전 덮어쓰기)
    in_feats = {
        **out_feats,
        "Request_URL": Request_URL(u, parsed=parsed),
        "URL_of_Anchor": URL_of_Anchor(u, parsed=parsed),
        "Links_in_tags": Links_in_tags(u, parsed=parsed),
        "SFH": SFH(u, parsed=parsed),
        "Submitting_to_email": Submitting_to_email(u, parsed=parsed),
        "Abnormal_URL": Abnormal_URL(u),
        "Redirect": Redirect(u, response=resp),
        "on_mouseover": on_mouseover(u, html=html),
        "RightClick": RightClick(u, html=html),
        "popUpWindow": popUpWindow(u, html=html),
        "Iframe": Iframe(u, parsed=parsed)
    }

    # 4) pandas.DataFrame으로 변환 → 컬럼명 지정
    if mode == "webOut":
        df_input = pd.DataFrame([out_feats], columns=OUT_COLS)
        m = model_out
    elif mode == "webIn":
        df_input = pd.DataFrame([in_feats], columns=IN_COLS)
        m = model_in
    else:
        raise HTTPException(status_code=422, detail="mode는 반드시 'webOut' 또는 'webIn'이어야 합니다.")

    # 5) 예측 및 확률 계산
    try:
        pred = int(m.predict(df_input)[0])
        proba = m.predict_proba(df_input)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류: {e}")

    return {
        "mode": mode,
        "prediction": bool(pred),
        "confidence": proba[pred],
        "probabilities": proba
    }
