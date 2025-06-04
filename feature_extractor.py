
# feature_extractor.py
import re
import socket
from urllib.parse import urlparse
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from functools import lru_cache

# -------------- (1) WHOIS 조회 캐싱 ----------------
@lru_cache(maxsize=256)
def _cached_whois_lookup(domain: str):
    
    # 도메인 WHOIS 정보를 lru_cache로 캐싱.
    # Returns: whois 결과 객체 또는 None
    
    try:
        import whois
        w = whois.whois(domain)
        return w
    except:
        return None


# -------------- (2) URL 기반 14개 피처 ----------------
def Prefix_Suffix(url, **kwargs):
    netloc = urlparse(url).netloc
    return -1 if '-' in netloc else 1

def double_slash_redirecting(url, **kwargs):
    return -1 if url.count('//') > 1 else 1

def having_At_Symbol(url, **kwargs):
    return -1 if '@' in url else 1

def Shortining_Service(url, **kwargs):
    # 단축서비스 도메인 패턴
    pattern = r"(bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|t\.co|tiny\.cc|bitly\.com|lc\.chat)"
    return -1 if re.search(pattern, url) else 1

def URL_Length(url, **kwargs):
    length = len(url)
    if length < 54:
        return 1
    if length <= 75:
        return 0
    return -1

def having_IP_Address(url, **kwargs):
    try:
        host = urlparse(url).netloc.split(':')[0]
        socket.inet_aton(host)  # IP인지 체크
        return -1
    except:
        return 1

def having_Sub_Domain(url, **kwargs):
    netloc = urlparse(url).netloc
    parts = netloc.split('.')
    sub_count = len(parts) - 2
    if sub_count <= 0:
        return 1
    if sub_count == 1:
        return 0
    return -1

def SSLfinal_State(url, response=None, **kwargs):
    
    # response가 None이면 자체적으로 요청 -> 파싱
    
    try:
        if response is None:
            response = requests.get(url, timeout=5)
        # HTTPS && status_code == 200 → 정상
        if response.url.startswith("https://") and response.status_code == 200:
            return 1
        return -1
    except:
        return 0

def Domain_registeration_length(url, **kwargs):
    
    # WHOIS 정보로 도메인 등록 기간 계산
    
    dom = urlparse(url).netloc
    w = _cached_whois_lookup(dom)
    if w is None:
        return 0
    cd = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
    ed = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
    if not cd or not ed:
        return 0
    return 1 if (ed - cd).days >= 365 else -1

def age_of_domain(url, **kwargs):
    dom = urlparse(url).netloc
    w = _cached_whois_lookup(dom)
    if w is None:
        return 0
    cd = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
    if not cd:
        return 0
    age_days = (datetime.now() - cd).days
    if age_days >= 365:
        return 1
    if age_days >= 180:
        return 0
    return -1

def DNSRecord(url, **kwargs):
    try:
        host = urlparse(url).netloc
        socket.gethostbyname(host)
        return 1
    except socket.gaierror:
        return -1
    except:
        return 0

def Favicon(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            resp = requests.get(url, timeout=5)
            parsed = BeautifulSoup(resp.text, 'html.parser')
        f_tag = parsed.find('link', rel=lambda x: x and 'icon' in x.lower())
        if not f_tag or 'href' not in f_tag.attrs:
            return 0
        fav_url = f_tag['href']
        fav_netloc = urlparse(fav_url).netloc
        site_netloc = urlparse(url).netloc
        return 1 if fav_netloc == site_netloc else -1
    except:
        return 0

def port(url, **kwargs):
    try:
        p = urlparse(url).port
        if p is None or p in (80, 443):
            return 1
        return -1
    except:
        return 0

def HTTPS_token(url, **kwargs):
    scheme_part = url.split('//')[0].lower()
    return -1 if 'https' in scheme_part else 1

# -------------- (3) 웹페이지 기반 추가 11개 피처 ----------------
def Request_URL(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            resp = requests.get(url, timeout=5)
            parsed = BeautifulSoup(resp.text, 'html.parser')
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

def URL_of_Anchor(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            resp = requests.get(url, timeout=5)
            parsed = BeautifulSoup(resp.text, 'html.parser')
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

def Links_in_tags(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            resp = requests.get(url, timeout=5)
            parsed = BeautifulSoup(resp.text, 'html.parser')
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

def SFH(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            resp = requests.get(url, timeout=5)
            parsed = BeautifulSoup(resp.text, 'html.parser')
        form = parsed.find('form')
        if not form:
            return 0
        action = form.get('action', '')
        if action == "" or url not in action:
            return -1
        return 1
    except:
        return 0

def Submitting_to_email(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            resp = requests.get(url, timeout=5)
            parsed = BeautifulSoup(resp.text, 'html.parser')
        form = parsed.find('form')
        if not form:
            return 0
        action = form.get('action', '')
        return -1 if 'mailto:' in action.lower() else 1
    except:
        return 0

def Abnormal_URL(url, **kwargs):
    netloc = urlparse(url).netloc
    return -1 if netloc not in url else 1

def Redirect(url, response=None, **kwargs):
    try:
        if response is None:
            response = requests.get(url, timeout=5)
        hops = len(response.history)
        if hops <= 1:
            return 1
        if hops <= 4:
            return 0
        return -1
    except:
        return 0

def on_mouseover(url, html=None, **kwargs):
    try:
        if html is None:
            resp = requests.get(url, timeout=5)
            html = resp.text
        return -1 if 'onmouseover' in html.lower() else 1
    except:
        return 0

def RightClick(url, html=None, **kwargs):
    try:
        if html is None:
            resp = requests.get(url, timeout=5)
            html = resp.text
        t = html.lower()
        return -1 if ('event.button==2' in t or 'contextmenu' in t) else 1
    except:
        return 0

def popUpWindow(url, html=None, **kwargs):
    try:
        if html is None:
            resp = requests.get(url, timeout=5)
            html = resp.text
        return -1 if 'window.open' in html.lower() else 1
    except:
        return 0

def Iframe(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            resp = requests.get(url, timeout=5)
            parsed = BeautifulSoup(resp.text, 'html.parser')
        frames = parsed.find_all('iframe')
        return -1 if len(frames) > 0 else 1
    except:
        return 0
