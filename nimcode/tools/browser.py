"""Optional headless-browser tools (pip install playwright; playwright install chromium). Registered only if installed."""
from __future__ import annotations

from ..core import Registry
from ..util import have, truncate

_S: dict = {"pw": None, "browser": None, "page": None}


def _page():
    if _S["page"] is None:
        from playwright.sync_api import sync_playwright
        _S["pw"] = sync_playwright().start()
        _S["browser"] = _S["pw"].chromium.launch(headless=True)
        _S["page"] = _S["browser"].new_page(viewport={"width": 1280, "height": 900})
    return _S["page"]


def register(reg: Registry) -> None:
    if not have("playwright"):
        return

    @reg.tool("browser_open", "Open a URL in a headless Chromium browser and return the page title and visible text.",
              {"url": {"type": "string"}}, ["url"], group="browser", ask=True, summary_key="url")
    def browser_open(ctx, url):
        pg = _page()
        pg.goto(url, wait_until="domcontentloaded", timeout=45000)
        return "%s\n%s" % (pg.title(), truncate(pg.inner_text("body"), 12000))

    @reg.tool("browser_text", "Return the visible text of the current page (or of a CSS selector).",
              {"selector": {"type": "string"}}, [], group="browser", read_only=False, summary_key="selector")
    def browser_text(ctx, selector="body"):
        return truncate(_page().inner_text(selector or "body"), 12000)

    @reg.tool("browser_click", "Click an element by CSS selector or text=Label.", {"selector": {"type": "string"}}, ["selector"],
              group="browser", ask=True, summary_key="selector")
    def browser_click(ctx, selector):
        pg = _page()
        pg.click(selector, timeout=15000)
        pg.wait_for_load_state("domcontentloaded")
        return "Clicked. Now at %s (%s)" % (pg.url, pg.title())

    @reg.tool("browser_fill", "Type text into an input by CSS selector (optionally press Enter).",
              {"selector": {"type": "string"}, "text": {"type": "string"}, "submit": {"type": "boolean"}},
              ["selector", "text"], group="browser", ask=True, summary_key="selector")
    def browser_fill(ctx, selector, text, submit=False):
        pg = _page()
        pg.fill(selector, text, timeout=15000)
        if str(submit).lower() in ("true", "1"):
            pg.press(selector, "Enter")
            pg.wait_for_load_state("domcontentloaded")
        return "Filled %s" % selector

    @reg.tool("browser_eval", "Run JavaScript in the page and return the JSON result.", {"js": {"type": "string"}}, ["js"],
              group="browser", ask=True, summary_key="js")
    def browser_eval(ctx, js):
        import json
        return truncate(json.dumps(_page().evaluate(js), default=str), 8000)

    @reg.tool("browser_screenshot", "Save a screenshot of the current page (use analyze_image to look at it).",
              {"path": {"type": "string"}}, [], group="browser", summary_key="path")
    def browser_screenshot(ctx, path="browser.png"):
        out = ctx.path(path)
        _page().screenshot(path=str(out), full_page=False)
        return "Saved %s" % out

    @reg.tool("browser_close", "Close the headless browser.", {}, [], group="browser")
    def browser_close(ctx):
        try:
            if _S["browser"]:
                _S["browser"].close()
            if _S["pw"]:
                _S["pw"].stop()
        finally:
            _S.update(pw=None, browser=None, page=None)
        return "Browser closed."
