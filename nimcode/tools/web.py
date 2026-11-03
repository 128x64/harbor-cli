"""Web tools: fetch (HTML -> markdown/text), search (Bing, no key), raw HTTP requests, downloads."""
from __future__ import annotations

import base64
import gzip
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import zlib
from html import unescape

from ..core import Registry
from ..util import bold, cyan, decode_bytes, truncate, truthy
from .. import VERSION

UA = "Mozilla/5.0 (compatible; nimcode/%s)" % VERSION


def _http(method, url, headers=None, data=None, timeout=30, max_bytes=3_000_000):
    h = {"User-Agent": UA, "Accept": "*/*"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        status = resp.status
    except urllib.error.HTTPError as e:
        resp, status = e, e.code
    raw = resp.read(max_bytes)
    enc = (resp.headers.get("Content-Encoding") or "").lower()
    try:
        if enc == "gzip":
            raw = gzip.decompress(raw)
        elif enc == "deflate":
            raw = zlib.decompress(raw)
    except Exception:
        pass
    return status, resp.headers, raw


def html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "svg", "form", "iframe"]):
            tag.decompose()
        try:
            from markdownify import markdownify
            md = markdownify(str(soup.body or soup), heading_style="ATX")
            return re.sub(r"\n{3,}", "\n\n", md).strip()
        except ImportError:
            return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()
    except ImportError:
        text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = unescape(text)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()


BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0")


def bing_unwrap(url: str) -> str:
    """Bing wraps result links as https://www.bing.com/ck/a?...&u=a1<base64url(target)>. Return the real URL."""
    url = unescape(url)
    if "bing.com/ck/a" not in url:
        return url
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    u = (qs.get("u") or [""])[0]
    if u.startswith("a1"):
        b = u[2:]
        try:
            return base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)).decode("utf-8", "replace")
        except Exception:
            pass
    return url


def parse_bing_html(html: str, n: int = 6) -> list:
    out = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a") or li.select_one("a[href]")
            if not a or not a.get("href"):
                continue
            p = li.select_one(".b_caption p") or li.select_one("p")
            out.append({"title": a.get_text(" ", strip=True), "url": bing_unwrap(a["href"]),
                        "snippet": p.get_text(" ", strip=True) if p else ""})
            if len(out) >= n:
                break
        return out
    except ImportError:
        pass
    strip = lambda s: unescape(re.sub(r"<[^>]+>", "", s or "")).strip()
    for block in re.split(r'<li class="b_algo"', html)[1:]:
        m = re.search(r'(?s)<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block)
        if not m:
            continue
        p = re.search(r"(?s)<p[^>]*>(.*?)</p>", block)
        out.append({"title": strip(m.group(2)), "url": bing_unwrap(m.group(1)), "snippet": strip(p.group(1)) if p else ""})
        if len(out) >= n:
            break
    return out


def bing_search(query: str, n: int = 6) -> list:
    """Scrape Bing's HTML results page (no API key needed). Returns [{title, url, snippet}]."""
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "setlang": "en-US", "count": max(n, 10)})
    status, _h, raw = _http("GET", url, {"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml",
                                          "Accept-Language": "en-US,en;q=0.9"})
    if status != 200:
        raise RuntimeError("Bing returned HTTP %d" % status)
    return parse_bing_html(decode_bytes(raw), n)


def register(reg: Registry) -> None:
    @reg.tool("web_search", "Search the web with Bing (no API key). Returns titles, URLs and snippets; use web_fetch to read a page.",
              {"query": {"type": "string"}, "max_results": {"type": "integer", "description": "default 6"}},
              ["query"], group="web", read_only=True, summary_key="query")
    def web_search(ctx, query, max_results=6):
        try:
            res = bing_search(query, max(1, min(int(max_results), 15)))
        except Exception as e:
            return "Error: search failed: %s" % e
        if not res:
            return "No results."
        return "\n\n".join("%d. %s\n   %s\n   %s" % (i, r["title"], r["url"], r["snippet"]) for i, r in enumerate(res, 1))

    @reg.tool("web_fetch", "Fetch a URL and return readable text (HTML converted to markdown). JSON/plain text returned as-is.",
              {"url": {"type": "string"}, "max_chars": {"type": "integer", "description": "default 15000"}},
              ["url"], group="web", ask=True, read_only=True, summary_key="url",
              preview=lambda ctx, a: str(a.get("url", "")))
    def web_fetch(ctx, url, max_chars=15000):
        if not re.match(r"^https?://", url):
            return "Error: url must start with http:// or https://"
        try:
            status, headers, raw = _http("GET", url)
        except Exception as e:
            return "Error: could not fetch: %s" % e
        ctype = (headers.get("Content-Type") or "").lower()
        text = decode_bytes(raw)
        if "html" in ctype or text.lstrip()[:15].lower().startswith(("<!doctype html", "<html")):
            text = html_to_text(text)
        return "[HTTP %d %s]\n%s" % (status, ctype.split(";")[0], truncate(text, int(max_chars)))

    @reg.tool("http_request", "Make an arbitrary HTTP request (REST APIs). Returns status, selected headers and body.",
              {"method": {"type": "string", "description": "GET, POST, PUT, PATCH, DELETE"},
               "url": {"type": "string"}, "headers": {"type": "object"},
               "json_body": {"type": "object", "description": "sent as JSON"},
               "body": {"type": "string", "description": "raw body (if not json_body)"}},
              ["method", "url"], group="web", ask=True, summary_key="url",
              preview=lambda ctx, a: "%s %s" % (a.get("method", "GET"), a.get("url", "")) +
              ("\nbody: " + json.dumps(a.get("json_body"))[:300] if a.get("json_body") else ""))
    def http_request(ctx, method, url, headers=None, json_body=None, body=""):
        if not re.match(r"^https?://", url):
            return "Error: url must start with http:// or https://"
        hdrs, data = dict(headers or {}), None
        if json_body is not None:
            data = json.dumps(json_body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        elif body:
            data = body.encode()
        try:
            status, h, raw = _http(method.upper(), url, hdrs, data)
        except Exception as e:
            return "Error: request failed: %s" % e
        keep = {k: v for k, v in h.items() if k.lower() in ("content-type", "location", "retry-after", "content-length")}
        return "HTTP %d\n%s\n\n%s" % (status, json.dumps(keep), truncate(decode_bytes(raw), 15000))

    @reg.tool("download_file", "Download a URL to a local file (max 200 MB).",
              {"url": {"type": "string"}, "path": {"type": "string"}}, ["url", "path"], group="web", ask=True,
              summary_key="url")
    def download_file(ctx, url, path):
        if not re.match(r"^https?://", url):
            return "Error: url must start with http:// or https://"
        dest = ctx.path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 200 * 1024 * 1024:
                        f.close()
                        dest.unlink(missing_ok=True)
                        return "Error: file larger than 200 MB; aborted."
                    f.write(chunk)
        except Exception as e:
            return "Error: download failed: %s" % e
        return "Downloaded %d bytes to %s" % (total, dest)
