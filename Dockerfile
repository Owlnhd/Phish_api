FROM python:3.10-slim
WORKDIR /app

# 시스템 의존성: gcc가 필수라면 남기되, 설치/캐시 정리
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY main.py feature_extractor.py ./

# 모델 파일 명시적 복사
COPY rf_model_webOut01.pkl rf_model_webIn01.pkl ./

# 서버 실행
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","5000"]