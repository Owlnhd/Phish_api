
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
import joblib, requests, numpy as np
from bs4 import BeautifulSoup

# 25개 피처 함수 import
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

# 모델 로드
model_out = joblib.load("rf_model_webOut01.pkl")   # 14개 URL-only 모델
model_in  = joblib.load("rf_model_webIn01.pkl")    # 25개 URL+내부 모델

# 입력 스키마: URL 하나만 받음
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

    # 한 번만 HTTP GET 수행
    try:
        response = requests.get(u, timeout=5)
        html = response.text
        parsed = BeautifulSoup(html, "html.parser")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL 요청 실패: {e}")

    # URL-only 14개 피처
    out_feats = [
        having_IP_Address(u),
        URL_Length(u),
        Shortining_Service(u),
        having_At_Symbol(u),
        double_slash_redirecting(u),
        Prefix_Suffix(u),
        having_Sub_Domain(u),
        SSLfinal_State(u, response=response),
        Domain_registeration_length(u),
        Favicon(u, parsed=parsed),
        port(u),
        HTTPS_token(u),
        age_of_domain(u),
        DNSRecord(u)
    ]

    # URL+내부 25개 피처: out_feats + 추가 11개
    in_feats = out_feats + [
        Request_URL(u, parsed=parsed),
        URL_of_Anchor(u, parsed=parsed),
        Links_in_tags(u, parsed=parsed),
        SFH(u, parsed=parsed),
        Submitting_to_email(u, parsed=parsed),
        Abnormal_URL(u),
        Redirect(u, response=response),
        on_mouseover(u, html=html),
        RightClick(u, html=html),
        popUpWindow(u, html=html),
        Iframe(u, parsed=parsed)
    ]

    # mode에 따라 모델 및 피처 벡터 선택
    if mode == "webOut":
        x = np.array([out_feats])
        m = model_out
    elif mode == "webIn":
        x = np.array([in_feats])
        m = model_in
    else:
        raise HTTPException(status_code=422, detail="mode must be 'webOut' or 'webIn'")

    # 예측 및 확률
    try:
        pred = int(m.predict(x)[0])
        proba = m.predict_proba(x)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction error: {e}")

    return {
        "mode": mode,
        "prediction": bool(pred),
        "confidence": proba[pred],
        "probabilities": proba
    }
