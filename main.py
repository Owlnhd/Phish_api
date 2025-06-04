
# main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup

# (1) feature_extractor 모듈을 필요 시에 로컬 임포트
#     → whois, bs4 등의 부담을 최대한 늦게 로드
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

app = FastAPI(title="Phishing Detection API (메모리 최적화 버전)")

# ----------------- (2) 글로벌 세션 및 모델 로드 -----------------
# requests.Session()을 전역으로 생성 → 커넥션 재사용, 메모리/CPU 절약
session = requests.Session()

# 모델 파일은 애플리케이션 시작 시 한 번만 로드
model_out = joblib.load("rf_model_webOut01.pkl")  # 14개 URL-only 모델
model_in  = joblib.load("rf_model_webIn01.pkl")   # 25개 URL+내부 모델

# (3) 컬럼 순서를 리스트로 정의 (numpy array 생성 시 사용)
OUT_COLS = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token",
    "age_of_domain", "DNSRecord"
]
IN_COLS = OUT_COLS + [
    "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
    "Submitting_to_email", "Abnormal_URL", "Redirect", "on_mouseover",
    "RightClick", "popUpWindow", "Iframe"
]

# 입력 스키마: 사용자로부터 URL 하나만 받음
class URLInput(BaseModel):
    url: HttpUrl

@app.get("/")
def health_check():
    return {"status": "ok", "models": ["webOut", "webIn"]}


@app.post("/predict")
def predict(
    inp: URLInput,
    mode: str = Query(..., description="webOut=14개 URL-only, webIn=25개 URL+내부")
):
    u = str(inp.url)

    # 1) 한 번만 HTTP GET 요청 → response + html + parsed(BeautifulSoup)
    try:
        resp = session.get(u, timeout=5)
        html = resp.text
        parsed = BeautifulSoup(html, "html.parser")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL 요청 혹은 파싱 실패: {e}")

    # 2) URL-only 14개 피처 → 리스트 형태로 즉시 계산
    out_list = [
        having_IP_Address(u),
        URL_Length(u),
        Shortining_Service(u),
        having_At_Symbol(u),
        double_slash_redirecting(u),
        Prefix_Suffix(u),
        having_Sub_Domain(u),
        SSLfinal_State(u, response=resp),
        Domain_registeration_length(u),
        Favicon(u, parsed=parsed),
        port(u),
        HTTPS_token(u),
        age_of_domain(u),
        DNSRecord(u)
    ]

    # 3) URL+내부 25개 피처 → out_list + 추가 11개 항목
    in_list = out_list + [
        Request_URL(u, parsed=parsed),
        URL_of_Anchor(u, parsed=parsed),
        Links_in_tags(u, parsed=parsed),
        SFH(u, parsed=parsed),
        Submitting_to_email(u, parsed=parsed),
        Abnormal_URL(u),
        Redirect(u, response=resp),
        on_mouseover(u, html=html),
        RightClick(u, html=html),
        popUpWindow(u, html=html),
        Iframe(u, parsed=parsed)
    ]

    # 4) mode에 따라 numpy array로 변환 (pandas 사용 완전 제거)
    if mode == "webOut":
        x = np.array([out_list])
        m = model_out
    elif mode == "webIn":
        x = np.array([in_list])
        m = model_in
    else:
        raise HTTPException(status_code=422, detail="mode는 'webOut' 혹은 'webIn'이어야 합니다.")

    # 5) 예측 & 확률 산출
    try:
        pred = int(m.predict(x)[0])
        proba = m.predict_proba(x)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {e}")

    # 6) 참조 해제(가비지 컬렉션 가속)
    del resp, html, parsed

    return {
        "mode": mode,
        "prediction": bool(pred),
        "confidence": proba[pred],
        "probabilities": proba
    }
