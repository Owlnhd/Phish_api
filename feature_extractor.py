
# feature_extractor.py
# feature_extractor.py
import re
import socket
from urllib.parse import urlparse
from datetime import datetime
import whois
import requests
from bs4 import BeautifulSoup

# 1) Prefix_Suffix (도메인에 "-" 포함 여부)
#   -1 = 하이픈이 있다 (피싱 의심)
#    1 = 하이픈이 없다 (정상)
def Prefix_Suffix(url):
    netloc = urlparse(url).netloc
    return -1 if '-' in netloc else 1


# 2) double_slash_redirecting (URL 중 "//" 횟수)
#   -1 = "//"이 두 번 이상 (e.g. http://example.com//somewhere → 피싱 의심)
#    1 = 그 외
def double_slash_redirecting(url):
    # “//”를 제외하고, 도메인 뒤 슬래시가 두 번 이상 있으면 피싱 의심
    if url.count('//') > 1:
        return -1
    return 1


# 3) having_At_Symbol ("@" 기호 포함 여부)
#   -1 = URL에 "@"가 있으면 피싱 의심
#    1 = 없으면 정상
def having_At_Symbol(url):
    return -1 if '@' in url else 1


# 4) Shortining_Service (shortening service 사용 여부)
#   -1 = bit.ly, tinyurl.com 등 URL 단축 서비스 사용 → 의심
#    1 = 그 외 정상
def Shortining_Service(url):
    # 대표적인 단축 도메인 패턴 (정규식)
    pattern = r"(bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|t\.co|tiny\.cc|bitly\.com|lc\.chat)"
    return -1 if re.search(pattern, url) else 1


# 5) URL_Length (URL 길이에 따른 분류)
#   임계치: < 54 → 정상(1), 54–75 → 의심(0), > 75 → 피싱(-1)
def URL_Length(url):
    length = len(url)
    if length < 54:
        return 1
    if 54 <= length <= 75:
        return 0
    return -1


# 6) having_IP_Address (도메인 대신 IP 사용 여부)
#   -1 = 도메인이 IP 형식(e.g. "192.168.0.1") → 피싱 의심
#    1 = 도메인 이름 사용 → 정상
def having_IP_Address(url):
    try:
        host = urlparse(url).netloc
        # 도메인에 포트가 붙어 있으면 분리
        host = host.split(':')[0]
        socket.inet_aton(host)
        return -1
    except:
        return 1


# 7) having_Sub_Domain (서브도메인 개수에 따른 분류)
#   도메인의 “.” 개수를 세어 서브도메인 수를 구함
#   - 0 또는 1개의 서브도메인: 정상(1)
#   - 2개의 서브도메인: 의심(0)
#   - 3개 이상 서브도메인: 피싱(-1)
def having_Sub_Domain(url):
    netloc = urlparse(url).netloc
    # 서브도메인 수 = 전체 레이블(도메인)에서 "." 개수 - 1
    parts = netloc.split('.')
    sub_count = len(parts) - 2  # "example.com" → sub_count=0, "sub.example.com"→1
    if sub_count == 0:
        return 1
    if sub_count == 1:
        return 0
    return -1


# 8) SSLfinal_State (SSL 인증서 상태)
#   1: HTTPS로 리디렉션되고 상태 코드 200 → 정상
#   0: 검증 불가(예: 타임아웃, 가져올 수 없음) → 의심
#  -1: HTTP만 사용(https가 아니거나 인증서 문제) → 피싱 의심
def SSLfinal_State(url):
    try:
        resp = requests.get(url, timeout=5)
        # 실제로 HTTPS인지 확인
        if resp.url.startswith("https://") and resp.status_code == 200:
            return 1
        return -1
    except:
        return 0


# 9) Domain_registeration_length (도메인 등록 기간)
#   -1: 등록 기간 < 365일 → 피싱 가능성
#    1: 등록 기간 >= 365일 → 비교적 안전
#    0: WHOIS 정보를 가져올 수 없는 경우 → 의심
def Domain_registeration_length(url):
    try:
        w = whois.whois(urlparse(url).netloc)
        # creation_date, expiration_date가 리스트일 수도 있음
        cd = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        ed = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
        # 등록 기간이 1년(365일) 미만이면 피싱 의심
        return 1 if (ed - cd).days >= 365 else -1
    except:
        return 0


# 10) age_of_domain (도메인 연령)
#   -1: 연령 < 180일 → 피싱 의심
#    0: 180일 ≤ 연령 < 365일 → 의심
#    1: 연령 ≥ 365일 → 정상
def age_of_domain(url):
    try:
        w = whois.whois(urlparse(url).netloc)
        cd = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        age = (datetime.now() - cd).days
        if age >= 365:
            return 1
        if age >= 180:
            return 0
        return -1
    except:
        return 0


# 11) DNSRecord (DNS 레코드 존재 여부)
#   1: 정상적으로 DNS가 존재함 → 정상
#   -1: DNS 조회 실패 → 피싱 의심
#    0: 예외 등으로 판단 불가 → 의심
def DNSRecord(url):
    try:
        host = urlparse(url).netloc
        socket.gethostbyname(host)
        return 1
    except socket.gaierror:
        return -1
    except:
        return 0


# 12) Favicon (파비콘 출처)
#   1: 파비콘이 동일 도메인에서 로드됨 → 정상
#   -1: 외부 도메인 파비콘 → 피싱 의심
#    0: 파비콘 태그를 찾을 수 없거나 확인 불가 → 의심
def Favicon(url):
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        f_tag = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        if not f_tag or 'href' not in f_tag.attrs:
            return 0
        fav_url = f_tag['href']
        fav_netloc = urlparse(fav_url).netloc
        site_netloc = urlparse(url).netloc
        return 1 if fav_netloc == site_netloc else -1
    except:
        return 0


# 13) port (URL에 포트 번호 사용 여부)
#   1: 기본 포트(80,443) 또는 포트 없음 → 정상
#   -1: 비표준 포트(예: 8080, 8443 등) → 피싱 의심
#    0: 포트 정보를 가져올 수 없는 경우 → 의심
def port(url):
    try:
        p = urlparse(url).port
        if p is None or p in (80, 443):
            return 1
        return -1
    except:
        return 0


# 14) HTTPS_token (URL 문자열에 "HTTPS" 단어 포함 여부)
#   -1: URL 도메인/경로에 "https"가 텍스트 형태로 포함 → 피싱 의심
#    1: 그렇지 않음 → 정상
def HTTPS_token(url):
    # domain://... 문자열 부분(스킴)만 검사
    scheme_part = url.split('//')[0].lower()
    return -1 if 'https' in scheme_part else 1


# 15) Request_URL (외부 리소스 비율)
#   ratio = (외부 도메인 src 태그 수 / 전체 src 태그 수) * 100
#   > 61% → -1 (피싱 의심)
#   22%–61% → 0 (의심)
#   < 22% → 1 (정상)
def Request_URL(url, parsed):
    try:
        base = urlparse(url).netloc
        tags = parsed.find_all(['img', 'audio', 'embed', 'iframe', 'script'])
        if not tags:
            return 1
        external = 0
        for t in tags:
            src = t.get('src', '')
            if src and urlparse(src).netloc and urlparse(src).netloc != base:
                external += 1
        ratio = (external / len(tags)) * 100
        if ratio < 22:
            return 1
        if ratio <= 61:
            return 0
        return -1
    except:
        return 0


# 16) URL_of_Anchor (외부 앵커 태그 비율)
#   안전한 앵커: href가 "#" 혹은 내부 도메인
#   > 67% → -1 (피싱 의심), 31%–67% → 0 (의심), < 31% → 1 (정상)
def URL_of_Anchor(url, parsed):
    try:
        base = urlparse(url).netloc
        anchors = parsed.find_all('a')
        if not anchors:
            return 1
        unsafe = 0
        for a in anchors:
            href = a.get('href', '')
            dom = urlparse(href).netloc
            if dom and dom != base:
                unsafe += 1
        ratio = (unsafe / len(anchors)) * 100
        if ratio < 31:
            return 1
        if ratio <= 67:
            return 0
        return -1
    except:
        return 0


# 17) Links_in_tags (meta/script/link 외부 링크 비율)
#   > 81% → -1 (피싱 의심), 17%–81% → 0 (의심), < 17% → 1 (정상)
def Links_in_tags(url, parsed):
    try:
        tags = parsed.find_all(['meta', 'script', 'link'])
        if not tags:
            return 1
        external = 0
        for t in tags:
            href = t.get('href', '') or t.get('src', '')
            if href and urlparse(href).netloc:
                external += 1
        ratio = (external / len(tags)) * 100
        if ratio < 17:
            return 1
        if ratio <= 81:
            return 0
        return -1
    except:
        return 0


# 18) SFH (Server Form Handler)
#   -1: action="" 혹은 action이 외부 도메인 → 피싱 의심
#    1: action이 내부 페이지(url 포함) → 정상
#    0: form 태그가 없거나 판단 불가 → 의심
def SFH(url, parsed):
    try:
        form = parsed.find('form')
        if not form:
            return 0
        action = form.get('action', '')
        if action == "" or url not in action:
            return -1
        return 1
    except:
        return 0


# 19) Submitting_to_email (form action에 mailto: 포함 여부)
#   -1: mailto:가 있으면 피싱 의심
#    1: 아니면 정상
def Submitting_to_email(url, parsed):
    try:
        form = parsed.find('form')
        if not form:
            return 0
        action = form.get('action', '')
        return -1 if 'mailto:' in action.lower() else 1
    except:
        return 0


# 20) Abnormal_URL (URL 구조가 이상한지 여부)
#   -1: 도메인(netloc) 정보가 URL 문자열에 포함되지 않으면 피싱 의심
#    1: 정상
def Abnormal_URL(url):
    netloc = urlparse(url).netloc
    return -1 if netloc not in url else 1


# 21) Redirect (리디렉션 횟수)
#   history list 길이(=리디렉션 횟수)
#   0–1회 → 정상(1), 2–4회 → 의심(0), >4회 → 피싱 의심(-1)
def Redirect(url, response):
    try:
        hops = len(response.history)
        if hops <= 1:
            return 1
        if hops <= 4:
            return 0
        return -1
    except:
        return 0


# 22) on_mouseover (onmouseover 이벤트 스크립트 존재 여부)
#   -1: HTML 내에 “onmouseover” 문자열이 있으면 피싱 의심
#    1: 없으면 정상
def on_mouseover(url, html):
    try:
        return -1 if 'onmouseover' in html.lower() else 1
    except:
        return 0


# 23) RightClick (우클릭 차단 스크립트 존재 여부)
#   -1: “event.button==2” 또는 “contextmenu” 코드가 있으면 피싱 의심
#    1: 아니면 정상
def RightClick(url, html):
    try:
        t = html.lower()
        return -1 if ('event.button==2' in t or 'contextmenu' in t) else 1
    except:
        return 0


# 24) popUpWindow (팝업 창 스크립트 존재 여부)
#   -1: “window.open”가 있으면 피싱 의심
#    1: 아니면 정상
def popUpWindow(url, html):
    try:
        return -1 if 'window.open' in html.lower() else 1
    except:
        return 0


# 25) Iframe (<iframe> 태그 존재 여부)
#   -1: iframe 태그가 하나라도 있으면 피싱 의심
#    1: 없으면 정상
#    0: 파싱 불가 시
def Iframe(url, parsed):
    try:
        frames = parsed.find_all('iframe')
        return -1 if len(frames) > 0 else 1
    except:
        return 0
