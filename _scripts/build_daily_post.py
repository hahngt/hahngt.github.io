import os, re, json, time, hashlib, datetime as dt
from dateutil import tz
import requests
from bs4 import BeautifulSoup
from tenacity import retry, wait_exponential, stop_after_attempt
from urllib.parse import urljoin, urlparse
from openai import OpenAI

# 전역
import math, openai

RPM_LIMIT = int(os.getenv("OPENAI_RPM_LIMIT","3"))
_rpm_win = time.monotonic(); _rpm_used = 0

def rpm_guard(n=1):
    global _rpm_win, _rpm_used
    now = time.monotonic()
    if now - _rpm_win >= 60:
        _rpm_win = now; _rpm_used = 0
    if _rpm_used + n > RPM_LIMIT:
        sleep = 60 - (now - _rpm_win) + 0.5
        if sleep > 0: time.sleep(sleep)
        _rpm_win = time.monotonic(); _rpm_used = 0
    _rpm_used += n

KST = tz.gettz("Asia/Seoul")
TODAY_KST = dt.datetime.now(KST).date()
DATE_STR = os.getenv("DATE", str(TODAY_KST))  # 재실행/백필 시 DATE=YYYY-MM-DD 로 override
HF_BASE = "https://huggingface.co"
HF_DAY_URL = f"{HF_BASE}/papers/date/{DATE_STR}"
HEADERS = {"User-Agent": "Mozilla/5.0 (PaperBot)"}

OUT_DIR = "_posts"
os.makedirs(OUT_DIR, exist_ok=True)

# ==== Seen DB =====
SEEN_DB = os.getenv("SEEN_DB", "_scripts/seen_papers.txt")
EXCLUDE_UPLOADED = os.getenv("EXCLUDE_UPLOADED", "1") == "1"
RESET_SEEN = os.getenv("RESET_SEEN", "0") == "1"

def _load_seen(path: str) -> set[str]:
    if RESET_SEEN: return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    except FileNotFoundError:
        return set()

def _save_seen(path: str, keys: set[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(keys)) + "\n")

def _extract_arxiv_id(url: str | None) -> str | None:
    if not url: return None
    u = url.replace("/pdf/", "/abs/").replace(".pdf", "")
    m = re.search(r"arxiv\.org/(abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", u)
    return m.group(2) if m else None

def _paper_key(meta: dict) -> str | None:
    # arXiv ID 우선
    aid = _extract_arxiv_id(meta.get("arxiv_url"))
    if aid: return f"arxiv:{aid}"
    # 없으면 HF papers/<pid>
    pid = _extract_pid(meta.get("hf_url", ""))
    if pid: return f"hf:{pid}"
    return None

@retry(wait=wait_exponential(multiplier=1, max=20), stop=stop_after_attempt(4))
def get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r

def _extract_pid(href: str):
    """
    Extracts the paper id from a Hugging Face Daily link.
    Accepts hrefs like '/papers/2508.18124', '/papers/2508.18124#community', etc.
    Returns the id string (e.g., '2508.18124') or None if not matched.
    """
    if not href:
        return None
    path = urlparse(href).path  # strip query/hash
    m = re.match(r"^/papers/([^/]+)$", path)
    return m.group(1) if m else None

def parse_daily_list(html):
    """
    Hugging Face Daily page parser.
    Primary selector: 'h3 a[href^="/papers/"]' (matches the title anchor inside each card)
    Fallback selector: 'article a[href^="/papers/"]' (covers the image link at the top of each card)
    The page structure (example card) includes both an image-link and a title-link; we only keep
    the first unique paper per card by deduping on the paper id.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Primary: title anchors inside h3
    links = soup.select('h3 a[href^="/papers/"]')

    # Fallback: any anchor inside article cards that points to /papers/<id>
    if not links:
        links = soup.select('article a[href^="/papers/"]')

    seen, items = set(), []
    for a in links:
        href = a.get("href") or ""
        pid = _extract_pid(href)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        items.append(urljoin(HF_BASE, f"/papers/{pid}"))

    return items
       
def parse_paper_page(url):
    """HF 논문 페이지에서 제목, arXiv 링크 추출. 없으면 HF 링크 사용."""
    h = get(url).text
    s = BeautifulSoup(h, "html.parser")
    # 제목: 첫 번째 h1
    title = (s.find("h1").get_text(strip=True) if s.find("h1") else "").strip()
    # arXiv abs 링크: 'View arXiv page' 앵커
    arxiv_link = None
    for a in s.find_all("a", href=True):
        if "arxiv.org" in a["href"]:
            arxiv_link = a["href"]
            if "/abs/" in arxiv_link or "/pdf/" in arxiv_link:
                break
    return {"title": title or url, "hf_url": url, "arxiv_url": arxiv_link, "pdf_url":arxiv_link.replace('abs','pdf')}

def fetch_arxiv_abstract(arxiv_abs_url):
    """arXiv 초록 텍스트 추출"""
    if not arxiv_abs_url:
        return ""
    abs_url = arxiv_abs_url
    if "/pdf/" in abs_url:
        abs_url = abs_url.replace("/pdf/", "/abs/").replace(".pdf", "")
    h = get(abs_url).text
    s = BeautifulSoup(h, "html.parser")
    # arXiv의 초록은 <blockquote class="abstract"> 내 텍스트
    blk = s.select_one("blockquote.abstract")
    if not blk:
        return ""
    txt = blk.get_text(" ", strip=True)
    # "Abstract:" 접두 제거
    return re.sub(r"^\s*Abstract:\s*", "", txt, flags=re.I)

def fetch_arxiv_pdf_text(pdf_url, max_chars=120000):
    if not pdf_url:
        return ""
    url = pdf_url
    if "/abs/" in url:
        url = url.replace("/abs/", "/pdf/") + ("" if url.endswith(".pdf") else ".pdf")
    r = get(url)  # 기존 get() 재사용
    import fitz  # PyMuPDF
    doc = fitz.open(stream=r.content, filetype="pdf")
    chunks = []
    total = 0
    for page in doc:
        t = page.get_text("text")
        if not t:
            t = page.get_text()  # fallback
        take = max_chars - total
        chunks.append(t[:take])
        total += len(chunks[-1])
        if total >= max_chars:
            break
    return "".join(chunks)

def summarize_with_gpt(client, item):
    title = item["title"]
    link = item.get("arxiv_url") or item["hf_url"]
    text = item.get("source_text") or item.get("abstract","")
    STYLE_RULES = """문체 규칙:
- 한국어 학술 보고체로 작성한다.
- 모든 문장은 '~다/~된다/~이다/~였다'로 끝난다.
- 1인칭/명령형/감탄/이모지 금지.
"""
    prompt = f"""{STYLE_RULES}
아래 논문 내용을 한국어로 요약하라.
요구사항:
- Markdown 문서 형식으로 작성하되, 코드블록( ``` )은 사용하지 말 것.
- 섹션과 형식을 아래 스켈레톤에 정확히 맞출 것.
- 각 줄 수/문장 수 제한을 지킬 것. 불명확하면 '정보 부족'으로 표기.
- 제공된 본문/초록에 근거한 사실만 사용할 것. 추정/외부지식 금지.
- 총 길이 ≤ 1000자.

[입력]
제목: {title}
본문:
{text}

[출력 스켈레톤]
## Introduction
- Goal: (정확히 1문장)
- Motivation: (정확히 1문장)
- Contribution: (정확히 1문장)

## Method (최대 3문장)

## Results (대표 성과 혹은 벤치마크 실험 결과, 정확히 1문장)

## Limitations (정확히 1문장)

## Conclusion (정확히 1문장)


"""
    # r = client.responses.create(
    #     model=os.getenv("OPENAI_MODEL","gpt-4.1-mini"),
    #     input=prompt,
    # )
    # return r.output_text.strip()
    rpm_guard(1)                 # 호출 직전
    for _ in range(6):           # 최대 6회 백오프
        try:
            r = client.responses.create(
                model=os.getenv("OPENAI_MODEL","gpt-4.1-mini"),
                input=prompt,
                max_output_tokens=int(os.getenv("MAX_OUT_TOKENS","300")),
            )
            return r.output_text.strip()
        except openai.RateLimitError as e:
            time.sleep(22)       # RPM/TPM 초과 공통 대기
    raise RuntimeError("OpenAI rate limit persistent")

def main():
    # 1) Daily 목록 수집
    daily_html = get(HF_DAY_URL).text
    paper_urls = parse_daily_list(daily_html)
    if not paper_urls:
        # 당일 비어있으면 전일로 백오프
        prev = (dt.datetime.fromisoformat(DATE_STR) - dt.timedelta(days=1)).date()
        alt_url = f"{HF_BASE}/papers/date/{prev}"
        daily_html = get(alt_url).text
        paper_urls = parse_daily_list(daily_html)

    # 상한 n개
    N = int(os.getenv("N","10"))
    paper_urls = paper_urls[:N]
    # print("paper_urls : ", len(paper_urls))
    # 2) 각 논문 상세 파싱 + arXiv 초록/PDF
    items = []
    for u in paper_urls:
        meta = parse_paper_page(u)
        meta["abstract"] = fetch_arxiv_abstract(meta["arxiv_url"]) if meta.get("arxiv_url") else ""
        # PDF 있으면 본문 텍스트 사용
        pdf_text = fetch_arxiv_pdf_text(meta.get("pdf_url"), max_chars=int(os.getenv("PDF_MAX_CHARS","40000"))) if meta.get("pdf_url") else ""
        meta["source_text"] = pdf_text if pdf_text else meta["abstract"]
        items.append(meta)
        time.sleep(0.5)
    # print("items : ", len(items))
    
    # seen = _load_seen(SEEN_DB)
    # if EXCLUDE_UPLOADED:
    #     items = [it for it in items if it.get("key") and it["key"] not in seen]
    # print("filtered items : ", len(items))

    # 3) 요약
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for it in items:
        it["summary_ko"] = summarize_with_gpt(client, it)
        time.sleep(3)
    # print("filtered items : ", len(items))

    # 4) Markdown 생성
    kst_now = dt.datetime.now(KST)
    front = [
        "---",
        "layout: post",
        f'title: Daily Papers — {DATE_STR}"',
        f"date: {DATE_STR} 08:15:00",
        "tags: [papers, arxiv, ai]",
        "categories: []",
        "---",
        "",
        "",
    ]
    body = []
    for i, it in enumerate(items, 1):
        link = it.get("arxiv_url") or it["hf_url"]
        title_one = it["title"].replace("\n", " ")
        body.append(f"# {i}. [{title_one}]({link})")
        body.append("")
        body.append(it["summary_ko"])
        body.append("")
    md = "\n".join(front + body)

    # 5) 파일 저장
    slug = "daily-papers"
    out_path = f"{OUT_DIR}/{DATE_STR}-{slug}_{DATE_STR.replace('-','')}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {out_path}")
    
    # new_keys = {it["key"] for it in items if it.get("key")}
    # seen |= new_keys
    # _save_seen(SEEN_DB, seen)

if __name__ == "__main__":
    main()