
# feature_extractor.py
import re
import socket
from urllib.parse import urlparse
from datetime import datetime
import whois
import requests
from bs4 import BeautifulSoup

def Prefix_Suffix(url, **kwargs):
    
    # 도메인(netloc)에 하이픈('-')이 있으면 피싱 의심 → -1, 없으면 정상 → 1
    
    netloc = urlparse(url).netloc
    return -1 if '-' in netloc else 1


def double_slash_redirecting(url, **kwargs):
    
    # URL 문자열에 '//'가 두 번 이상 나타나면 피싱 의심 → -1, 그 외 → 1
    
    return -1 if url.count('//') > 1 else 1


def having_At_Symbol(url, **kwargs):
    
    # URL에 '@'가 있으면 피싱 의심 → -1, 없으면 정상 → 1
    
    return -1 if '@' in url else 1


def Shortining_Service(url, **kwargs):
    
    # URL 단축 서비스(bit.ly, tinyurl.com 등) 사용 여부
    # - 만약 단축 도메인 패턴이 들어 있으면 피싱 의심 → -1
    # - 아니면 정상 → 1
    
    pattern = r"(bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|t\.co|tiny\.cc|bitly\.com|lc\.chat)"
    return -1 if re.search(pattern, url) else 1


def URL_Length(url, **kwargs):
    
    # URL 길이에 따라 세 단계로 분류
    #     length < 54    → 정상(1)
    #     54 <= length <= 75 → 의심(0)
    #     length > 75    → 피싱 의심(-1)
    
    length = len(url)
    if length < 54:
        return 1
    if length <= 75:
        return 0
    return -1


def having_IP_Address(url, **kwargs):
    
    # 도메인 대신 IP 문자열을 사용하는지 검사
    # - IP 형식일 경우 피싱 의심 → -1
    # - 그렇지 않으면 정상 → 1
    
    try:
        host = urlparse(url).netloc.split(':')[0]
        socket.inet_aton(host)  # IP 형식이면 예외 없이 통과
        return -1
    except:
        return 1


def having_Sub_Domain(url, **kwargs):

    # 서브도메인 개수에 따라 세 단계 분류
    #     sub_count = (# of '.') - 1
    #     sub_count == 0 → 정상(1)
    #     sub_count == 1 → 의심(0)
    #     sub_count >= 2 → 피싱 의심(-1)
    
    netloc = urlparse(url).netloc
    parts = netloc.split('.')
    sub_count = len(parts) - 2
    if sub_count <= 0:
        return 1
    if sub_count == 1:
        return 0
    return -1


def SSLfinal_State(url, response=None, **kwargs):
    
    # SSL 상태 검사 (HTTPS 연결 여부)
    # - response 인자로 requests.get 결과를 받으면, response.url로 https 확인
    # - response가 None이면 내부에서 GET 요청을 보내서 상태 확인
    # 리턴값:
    #     1  → response.url이 'https://' 로 시작하고 status_code==200 (정상)
    #     -1 → HTTPS가 아니거나 status_code!=200 (피싱 의심)
    #     0  → 요청 실패 등으로 확인 불가 (의심)
    
    try:
        if response is None:
            response = requests.get(url, timeout=5)
        if response.url.startswith("https://") and response.status_code == 200:
            return 1
        return -1
    except:
        return 0


def Domain_registeration_length(url, **kwargs):
    
    # 도메인 등록 기간(만료일 - 생성일) 확인
    # - 등록 기간 >= 365일 → 1
    # - 등록 기간 < 365일 → -1
    # - WHOIS 정보 조회 실패 시 → 0
    
    try:
        w = whois.whois(urlparse(url).netloc)
        cd = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        ed = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
        return 1 if (ed - cd).days >= 365 else -1
    except:
        return 0


def age_of_domain(url, **kwargs):
    
    # 도메인 연령을 일수로 계산
    #     age >= 365 → 1
    #     180 <= age < 365 → 0
    #     age < 180 → -1
    # 조회 실패 시 → 0
    
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


def DNSRecord(url, **kwargs):
    
    # DNS 레코드가 존재하는지 확인
    # - 조회 성공 → 1
    # - 조회 실패(socket.gaierror) → -1
    # - 기타 예외 → 0
    
    try:
        host = urlparse(url).netloc
        socket.gethostbyname(host)
        return 1
    except socket.gaierror:
        return -1
    except:
        return 0


def Favicon(url, parsed=None, **kwargs):
    
    # 파비콘이 동일 도메인에서 로드되는지 확인
    # - 동일 도메인 → 1
    # - 외부 도메인 → -1
    # - 파비콘 태그 없음 또는 예외 → 0
    
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
    
    # URL에서 포트 번호를 확인
    # - None 또는 80,443 → 1
    # - 그 외 비표준 포트 → -1
    # - 예외 시 → 0
    
    try:
        p = urlparse(url).port
        if p is None or p in (80, 443):
            return 1
        return -1
    except:
        return 0


def HTTPS_token(url, **kwargs):
    
    # URL 문자열에 "https"가 포함되는지 확인 (피싱 경우가 있음)
    # - scheme 부분(“http://” 등) 이외에 도메인/경로에 “https”가 있으면 → -1
    # - 아니면 → 1
    
    scheme_part = url.split('//')[0].lower()
    return -1 if 'https' in scheme_part else 1


def Request_URL(url, parsed=None, **kwargs):
    
    # 웹페이지 내에 포함된 img, script, iframe 등 태그(src 속성) 중 “외부 도메인” 비율 계산
    #   ratio = (외부 도메인 src 태그 수 / 전체 src 태그 수) * 100
    #   ratio < 22   → 1
    #   22 <= ratio <= 61 → 0
    #   ratio > 61  → -1
    # parsed (BeautifulSoup 객체)를 전달하면 page_source 재요청 없이 사용
    # 예외 시 → 0
    
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
    
    # 앵커(<a>) 태그 중 외부 도메인 링크 비율 계산
    #   ratio = (외부 앵커 / 전체 앵커) * 100
    #   ratio < 31   → 1
    #   31 <= ratio <= 67 → 0
    #   ratio > 67  → -1
    # parsed(BeautifulSoup) 전달 시 재요청 생략 가능
    # 예외 시 → 0
    
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
    
    # meta/script/link 태그 중 외부 링크 비율 계산
    #   ratio = (외부 링크 태그 수 / 전체 태그 수) * 100
    #   ratio < 17   → 1
    #   17 <= ratio <= 81 → 0
    #   ratio > 81  → -1
    # parsed(BeautifulSoup) 전달 시 재요청 생략 가능
    # 예외 시 → 0
    
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
    
    # Server Form Handler (form action)
    #   - action이 비어 있거나 외부 도메인일 경우 → -1
    #   - action이 내부 URL(자기 자신 포함)일 경우 → 1
    #   - form 태그 아예 없거나 예외 시 → 0
    
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
    
    # form action에 mailto: 포함 여부
    #   - 'mailto:' 있으면 → -1
    #   - 없으면 → 1
    #   - form 태그 없거나 예외 시 → 0
    
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
    
    # URL 구조 검사
    #   - 도메인(netloc)이 URL 문자열에 포함되지 않으면 → -1
    #   - 그 외 → 1
    
    netloc = urlparse(url).netloc
    return -1 if netloc not in url else 1


def Redirect(url, response=None, **kwargs):
    
    # 리디렉션 횟수(history 길이)로 세 단계 분류
    #   hops <= 1   → 1
    #   2 <= hops <= 4 → 0
    #   hops > 4   → -1
    # response(=requests.get 결과)가 없으면 내부에서 GET 요청 수행
    # 예외 시 → 0
    
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
    
    # HTML 소스에 “onmouseover” 문자열 존재 여부
    #   - 있으면 → -1
    #   - 없으면 → 1
    #   - 예외 시 → 0
    
    try:
        if html is None:
            resp = requests.get(url, timeout=5)
            html = resp.text
        return -1 if 'onmouseover' in html.lower() else 1
    except:
        return 0


def RightClick(url, html=None, **kwargs):
    
    # HTML 소스에 우클릭 차단 스크립트(“event.button==2” 또는 “contextmenu”) 존재 여부
    #   - 있으면 → -1
    #   - 없으면 → 1
    #   - 예외 시 → 0
    
    try:
        if html is None:
            resp = requests.get(url, timeout=5)
            html = resp.text
        t = html.lower()
        return -1 if ('event.button==2' in t or 'contextmenu' in t) else 1
    except:
        return 0


def popUpWindow(url, html=None, **kwargs):
    
    # HTML 소스에 “window.open” 문자열 존재 여부
    #   - 있으면 → -1
    #   - 없으면 → 1
    #   - 예외 시 → 0
    
    try:
        if html is None:
            resp = requests.get(url, timeout=5)
            html = resp.text
        return -1 if 'window.open' in html.lower() else 1
    except:
        return 0


def Iframe(url, parsed=None, **kwargs):
    
    # HTML 내 <iframe> 태그 존재 여부
    #   - 있으면 → -1
    #   - 없으면 → 1
    #   - 예외 시 → 0
    
    try:
        if parsed is None:
            resp = requests.get(url, timeout=5)
            parsed = BeautifulSoup(resp.text, 'html.parser')
        frames = parsed.find_all('iframe')
        return -1 if len(frames) > 0 else 1
    except:
        return 0

