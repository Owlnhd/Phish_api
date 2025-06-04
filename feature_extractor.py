
# feature_extractor.py
import re, socket
from urllib.parse import urlparse
from datetime import datetime
from bs4 import BeautifulSoup

def Prefix_Suffix(url, **kwargs):
    return int('-' in urlparse(url).netloc)

def double_slash_redirecting(url, **kwargs):
    return int(url.count('//') > 1)

def having_At_Symbol(url, **kwargs):
    return int('@' in url)

def Shortining_Service(url, **kwargs):
    # 백슬래시 없이 정규식 패턴만 쓴다
    pattern = r"bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|t\.co|tiny\.cc"
    return int(bool(re.search(pattern, url)))

def URL_Length(url, **kwargs):
    return int(len(url) > 75)

def having_IP_Address(url, **kwargs):
    try:
        host = urlparse(url).netloc
        socket.inet_aton(host)
        return 1
    except:
        return 0

def having_Sub_Domain(url, **kwargs):
    return int(len(urlparse(url).netloc.split('.')) - 2 > 0)

def SSLfinal_State(url, response=None, **kwargs):
    try:
        if response is None:
            return 0
        return int(response.url.startswith('https://') and response.status_code == 200)
    except:
        return 0

def Domain_registeration_length(url, **kwargs):
    try:
        import whois
        w = whois.whois(url)
        cd = w.creation_date if not isinstance(w.creation_date, list) else w.creation_date[0]
        ed = w.expiration_date if not isinstance(w.expiration_date, list) else w.expiration_date[0]
        return int((ed - cd).days < 365)
    except:
        return 1

def age_of_domain(url, **kwargs):
    try:
        import whois
        w = whois.whois(url)
        cd = w.creation_date if not isinstance(w.creation_date, list) else w.creation_date[0]
        return (datetime.now() - cd).days
    except:
        return 0

def DNSRecord(url, **kwargs):
    try:
        socket.gethostbyname(urlparse(url).netloc)
        return 1
    except:
        return 0

def Favicon(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            return 0
        f = parsed.find('link', rel=lambda x: x and 'icon' in x.lower())
        return int(urlparse(f['href']).netloc == urlparse(url).netloc)
    except:
        return 0

def port(url, **kwargs):
    p = urlparse(url).port
    return int(p not in (80, 443, None))

def HTTPS_token(url, **kwargs):
    return int('https' in url.split('//')[0].lower())

def Request_URL(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            return 0
        base = urlparse(url).netloc
        tags = parsed.find_all(['img','audio','embed','iframe','script'])
        external = sum(1 for t in tags if urlparse(t.get('src','')).netloc and urlparse(t.get('src')).netloc != base)
        return int(external/len(tags)*100) if tags else 0
    except:
        return 0

def URL_of_Anchor(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            return 0
        base = urlparse(url).netloc
        anchors = parsed.find_all('a')
        unsafe = sum(1 for a in anchors if urlparse(a.get('href','')).netloc and urlparse(a.get('href')).netloc != base)
        return int(unsafe/len(anchors)*100) if anchors else 0
    except:
        return 0

def Links_in_tags(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            return 0
        tags = parsed.find_all(['meta','script','link'])
        external = sum(1 for t in tags if t.get('href') and urlparse(t['href']).netloc)
        return int(external/len(tags)*100) if tags else 0
    except:
        return 0

def SFH(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            return 1
        form = parsed.find('form')
        action = form.get('action','') if form else ''
        return int(action == '' or url not in action)
    except:
        return 1

def Submitting_to_email(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            return 0
        f = parsed.find('form')
        return int('mailto:' in f.get('action','')) if f else 0
    except:
        return 0

def Abnormal_URL(url, **kwargs):
    return int(urlparse(url).netloc not in url)

def Redirect(url, response=None, **kwargs):
    try:
        if response is None:
            return 0
        return len(response.history)
    except:
        return 0

def on_mouseover(url, html=None, **kwargs):
    try:
        if html is None:
            return 0
        return int('onmouseover' in html.lower())
    except:
        return 0

def RightClick(url, html=None, **kwargs):
    try:
        if html is None:
            return 0
        t = html.lower()
        return int('event.button==2' in t or 'contextmenu' in t)
    except:
        return 0

def popUpWindow(url, html=None, **kwargs):
    try:
        if html is None:
            return 0
        return int('window.open' in html)
    except:
        return 0

def Iframe(url, parsed=None, **kwargs):
    try:
        if parsed is None:
            return 0
        return int(bool(parsed.find_all('iframe')))
    except:
        return 0
