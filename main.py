
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
import joblib
import requests
import numpy as np
import pandas as pd
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

# out-features(14개) 컬럼 이름 리스트
OUT_COLS = [
    "having_IP_Address","URL_Length","Shortining_Service","having_At_Symbol",
    "double_slash_redirecting","Prefix_Suffix","having_Sub_Domain","SSLfinal_State",
    "Domain_registeration_length","Favicon","port","HTTPS_token","age_of_domain","DNSRecord"
]

# in-features(25개) 컬럼 이름 리스트 (out-cols + 추가 11개)
IN_COLS = OUT_COLS + [
    "Request_URL","URL_of_Anchor","Links_in_tags","SFH",
    "Submitting_to_email","Abnormal_URL","Redirect","on_mouseover",
    "RightClick","popUpWindow","Iframe"
]

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

    # URL-only 14개 피처 계산
    out_feats = {
        "having_IP_Address": having_IP_Address(u),
        "URL_Length": URL_Length(u),
        "Shortining_Service": Shortining_Service(u),
        "having_At_Symbol": having_At_Symbol(u),
        "double_slash_redirecting": double_slash_redirecting(u),
        "Prefix_Suffix": Prefix_Suffix(u),
        "having_Sub_Domain": having_Sub_Domain(u),
        "SSLfinal_State": SSLfinal_State(u, response=response),
        "Domain_registeration_length": Domain_registeration_length(u),
        "Favicon": Favicon(u, parsed=parsed),
        "port": port(u),
        "HTTPS_token": HTTPS_token(u),
        "age_of_domain": age_of_domain(u),
        "DNSRecord": DNSRecord(u)
    }

    # URL+내부 25개 피처: 사전(Dictionary)으로 확장
    in_feats = { **out_feats,
        "Request_URL": Request_URL(u, parsed=parsed),
        "URL_of_Anchor": URL_of_Anchor(u, parsed=parsed),
        "Links_in_tags": Links_in_tags(u, parsed=parsed),
        "SFH": SFH(u, parsed=parsed),
        "Submitting_to_email": Submitting_to_email(u, parsed=parsed),
        "Abnormal_URL": Abnormal_URL(u),
        "Redirect": Redirect(u, response=response),
        "on_mouseover": on_mouseover(u, html=html),
        "RightClick": RightClick(u, html=html),
        "popUpWindow": popUpWindow(u, html=html),
        "Iframe": Iframe(u, parsed=parsed)
    }

    # mode에 따라 모델 및 DataFrame 선택
    if mode == "webOut":
        df_in = pd.DataFrame([out_feats], columns=OUT_COLS)
        m = model_out
    elif mode == "webIn":
        df_in = pd.DataFrame([in_feats], columns=IN_COLS)
        m = model_in
    else:
        raise HTTPException(status_code=422, detail="mode must be 'webOut' or 'webIn'")

    # 예측 및 확률
    try:
        pred = int(m.predict(df_in)[0])
        proba = m.predict_proba(df_in)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction error: {e}")

    return {
        "mode": mode,
        "prediction": bool(pred),
        "confidence": proba[pred],
        "probabilities": proba
    }
