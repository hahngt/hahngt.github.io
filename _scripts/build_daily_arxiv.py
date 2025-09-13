# _scripts/build_daily_post.py
import os, re, time, datetime as dt
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from dateutil import tz
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI
import openai

from tqdm import tqdm

# ===== 설정 =====
KST = tz.gettz("Asia/Seoul")
NOW_KST = dt.datetime.now(KST)
DATE_STR = os.getenv("DATE", NOW_KST.date().isoformat())  # 포스트 날짜(기본: 오늘 KST)
N_MAX = int(os.getenv("N", "20"))
PDF_MAX_CHARS = int(os.getenv("PDF_MAX_CHARS", "40000"))
RPM_LIMIT = int(os.getenv("OPENAI_RPM_LIMIT", "3"))
CATS = ("cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.MM")
# 포함/제외 키워드 설정 (정규식 허용)
# - INCLUDE_KWS: 제목/초록에 하나라도 매치되면 포함
# - EXCLUDE_KWS: 제목/초록에 하나라도 매치되면 제외 (우선순위 더 높음)
INCLUDE_KWS = [
    r"\bdiffusion",
    r"video",
    r"generati",
    r"vlm",
    r"rectified flow",
    r"mllm",
    r"\bCLIP\b",
    r"multi[-\s]?modal",
    r"flow[-\s]?matching",
    r"vision[-\s]?language",
    r"uni[-\s]?modal",
    r"image",
]
ENV_EXCLUDE = os.getenv("EXCLUDE_KWS", "").strip()
EXCLUDE_KWS = [p.strip() for p in ENV_EXCLUDE.split(",") if p.strip()]
WITHDRAWN_RE = re.compile(r"\bwithdrawn\b", re.I)

OUT_DIR = "_posts"
os.makedirs(OUT_DIR, exist_ok=True)

SEEN_DB = os.getenv("SEEN_DB", "_scripts/seen_papers.txt")
EXCLUDE_UPLOADED = os.getenv("EXCLUDE_UPLOADED", "1") == "1"
RESET_SEEN = os.getenv("RESET_SEEN", "0") == "1"

# ===== 유틸 =====
def _load_seen(path: str) -> set[str]:
    if RESET_SEEN:
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    except FileNotFoundError:
        return set()

def _save_seen(path: str, keys: set[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(keys)) + "\n")

def _arxiv_id_from_url(url: str) -> str | None:
    """
    abs URL 예: https://arxiv.org/abs/2508.12345v2  ->  2508.12345
    """
    if not url:
        return None
    u = url.replace("/pdf/", "/abs/").replace(".pdf", "")
    m = re.search(r"arxiv\.org/(abs|pdf)/([0-9]{4}\.[0-9]{4,5})", u)
    return m.group(2) if m else None

def _key_for_entry(abs_url: str) -> str | None:
    aid = _arxiv_id_from_url(abs_url)
    return f"arxiv:{aid}" if aid else None

# 간단한 RPM 가드
_rpm_win = time.monotonic()
_rpm_used = 0
def rpm_guard(n=1):
    global _rpm_win, _rpm_used
    now = time.monotonic()
    if now - _rpm_win >= 60:
        _rpm_win = now
        _rpm_used = 0
    if _rpm_used + n > RPM_LIMIT:
        sleep = 60 - (now - _rpm_win) + 0.5
        if sleep > 0:
            time.sleep(sleep)
        _rpm_win = time.monotonic()
        _rpm_used = 0
    _rpm_used += n

HEADERS = {"User-Agent": "Mozilla/5.0 (PaperBot)"}

@retry(wait=wait_exponential(multiplier=1, max=20), stop=stop_after_attempt(4))
def get(url, **kw):
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r

# ===== arXiv 수집 =====
def arxiv_query_url(max_results=100):
    # 카테고리 OR 쿼리
    # 날짜 필터는 API에서 직접 range가 제한적이므로 많이 가져와서 사후 필터
    cat_q = "+OR+".join([f"cat:{c}" for c in CATS])
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={cat_q}"
        "&sortBy=submittedDate&sortOrder=descending"
        f"&start=0&max_results={max_results}"
    )
    return url

def parse_arxiv_feed(xml_text: str):
    """
    BeautifulSoup(xml)로 arXiv ATOM 파싱.
    반환: [{title, summary, authors, abs_url, pdf_url, primary, published_dt, updated_dt}, ...]
    """
    s = BeautifulSoup(xml_text, "lxml-xml")  # lxml 설치 전제
    out = []
    for e in s.find_all("entry"):
        title = (e.title.get_text(strip=True) if e.title else "")
        summary = (e.summary.get_text(" ", strip=True) if e.summary else "")
        # authors
        authors = [a.find("name").get_text(strip=True) for a in e.find_all("author")]
        # links
        abs_url = None
        pdf_url = None
        for l in e.find_all("link"):
            rel = l.get("rel", "")
            href = l.get("href", "")
            typ = l.get("type", "")
            if rel == "alternate" and "abs" in href:
                abs_url = href
            if (typ == "application/pdf") or ("/pdf/" in href and href.endswith(".pdf")):
                pdf_url = href
        if not abs_url and e.id:
            abs_url = e.id.get_text(strip=True)
        # primary category
        prim = e.find("arxiv:primary_category")
        primary = prim.get("term") if prim else None
        # dates
        published = e.published.get_text(strip=True) if e.published else None
        updated = e.updated.get_text(strip=True) if e.updated else None
        try:
            pub_dt = dt.datetime.fromisoformat(published.replace("Z", "+00:00")) if published else None
        except Exception:
            pub_dt = None
        try:
            upd_dt = dt.datetime.fromisoformat(updated.replace("Z", "+00:00")) if updated else None
        except Exception:
            upd_dt = pub_dt

        out.append(
            {
                "title": title,
                "summary": summary,
                "authors": authors,
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "primary": primary,
                "published_dt": pub_dt,
                "updated_dt": upd_dt or pub_dt,
            }
        )
    return out

def within_last_24h(dt_utc: dt.datetime | None) -> bool:
    if not dt_utc:
        return False
    now_utc = dt.datetime.utcnow().replace(tzinfo=tz.tzutc())
    return (now_utc - dt_utc) <= dt.timedelta(days=1)

def match_keywords(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(pat, t, flags=re.I) for pat in INCLUDE_KWS)

def match_exclude(text: str) -> bool:
    if not EXCLUDE_KWS:
        return False
    t = (text or "").lower()
    return any(re.search(pat, t, flags=re.I) for pat in EXCLUDE_KWS)

def fetch_pdf_text(pdf_url: str | None, max_chars: int) -> str:
    if not pdf_url:
        return ""
    url = pdf_url
    if "/abs/" in url:
        url = url.replace("/abs/", "/pdf/") + ("" if url.endswith(".pdf") else ".pdf")
    r = get(url)
    import fitz  # PyMuPDF
    doc = fitz.open(stream=r.content, filetype="pdf")
    out, total = [], 0
    for pg in doc:
        t = pg.get_text("text") or pg.get_text()
        take = max_chars - total
        out.append(t[:take])
        total += len(out[-1])
        if total >= max_chars:
            break
    return "".join(out)

# ===== 요약 =====
def summarize_with_gpt(client: OpenAI, item: dict) -> str:
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
제목: {item['title']}
본문:
{item.get('source_text','')}

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
    rpm_guard(1)
    for _ in range(6):
        try:
            r = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                input=prompt,
                # max_output_tokens=int(os.getenv("MAX_OUT_TOKENS","300")),
            )
            return r.output_text.strip()
        except openai.RateLimitError:
            time.sleep(22)
    return "요약 실패: 레이트 리밋으로 인해 생성하지 못했다."

# ===== 메인 =====
def main():
    # 1) arXiv 최신 목록 가져와서 24h/카테고리/키워드/withdrawn 필터
    feed_url = arxiv_query_url(max_results=150)  # 넉넉히 가져와서 필터
    print(feed_url); print()
    xml = get(feed_url).text
    entries = parse_arxiv_feed(xml)
    # published_dt 기준 최신순 정렬
    entries.sort(
        key=lambda x: x.get("published_dt") or dt.datetime.min.replace(tzinfo=tz.tzutc()),
        reverse=True,
    )
    print(len(entries), "entries")
    for e in entries[:10]: print(e['title'], e['published_dt'], e['primary'])
    # 필터
    
    filt = []
    for e in entries:
        # 날짜 24h
        if not within_last_24h(e["published_dt"] or e["updated_dt"]):
            continue
        filt.append(e)
    print(len(filt), "papers in 24 hours")
    entries = filt
    filt = []
    for e in entries:
        # primary 카테고리 제한
        if e["primary"] not in CATS:
            continue
        # withdrawn 제외(제목/초록/코멘트)
        text_all = f"{e['title']} {e['summary']}"
        if WITHDRAWN_RE.search(text_all):
            continue
        # 제외 키워드 우선 적용
        if match_exclude(text_all):
            continue
        # 키워드 포함(제목+초록)
        if not match_keywords(text_all):
            continue
        filt.append(e)
    print(len(filt), "after filtering")
    
    if len(filt) == 0:
        print("No new papers found.")
        return
    
    # ID 기준 dedupe & seen DB 제외
    seen = _load_seen(SEEN_DB)
    uniq, seen_ids = [], set()
    for e in filt:
        abs_url = e["abs_url"] or ""
        aid = _arxiv_id_from_url(abs_url)
        if not aid:
            continue
        kid = f"arxiv:{aid}"
        if kid in seen_ids:
            continue
        if EXCLUDE_UPLOADED and kid in seen:
            continue
        seen_ids.add(kid)
        uniq.append(e)

    # 최신순 상한
    uniq.sort(key=lambda x: x["published_dt"] or x["updated_dt"] or dt.datetime.min.replace(tzinfo=tz.tzutc()), reverse=True)
    items = uniq[:N_MAX]

    # 2) PDF 텍스트(없으면 초록) 수집
    for it in items:
        pdf_text = fetch_pdf_text(it.get("pdf_url"), PDF_MAX_CHARS) if it.get("pdf_url") else ""
        it["source_text"] = pdf_text if pdf_text else (it.get("summary") or "")
        time.sleep(0.5)
        
    # print(len(items)); input()

    # 3) 요약
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for it in tqdm(items, desc="Paper Summarizing"):
        it["summary_ko"] = summarize_with_gpt(client, it)
        time.sleep(3)

    # 4) 마크다운 생성
    front = [
        "---",
        "layout: post",
        f'title: "Arxiv - {DATE_STR}"',
        f"date: {DATE_STR} 08:15:00",
        "tags: [arxiv, multimodal, diffusion, generative AI, vision]",
        "categories: []",
        "---",
        "",
        "",
    ]
    body = []
    for i, it in enumerate(items, 1):
        link = it.get("abs_url") or ""
        title_one = (it.get("title") or "").replace("\n", " ")
        body.append(f"# {i}. [{title_one}]({link})")
        body.append("")
        body.append(it.get("summary_ko", ""))
        body.append("")


    md = "\n".join(front + body)

    # 5) 저장 (하루 1개 파일 이름 고정)
    out_path = f"{OUT_DIR}/{DATE_STR}-Arxiv.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {out_path}")

    # 6) seen DB 갱신
    new_keys = {_key_for_entry(it.get("abs_url") or "") for it in items}
    new_keys = {k for k in new_keys if k}
    seen |= new_keys
    _save_seen(SEEN_DB, seen)

if __name__ == "__main__":
    main()
