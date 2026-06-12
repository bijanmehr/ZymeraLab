"""Docs sanity: files exist, are self-contained, internal links resolve."""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parent.parent / "docs"
PAGES = ["index.html", "tutorial-env.html"]


@pytest.mark.parametrize("page", PAGES)
def test_page_exists_and_wellformed(page):
    html = (DOCS / page).read_text()
    assert "<main>" in html and "</html>" in html
    assert html.count("<pre><code>") == html.count("</code></pre>")


@pytest.mark.parametrize("page", PAGES)
def test_self_contained(page):
    """No external resources — docs must work offline."""
    html = (DOCS / page).read_text()
    for m in re.finditer(r'(?:href|src)="(http[^"]+)"', html):
        pytest.fail(f"external resource in {page}: {m.group(1)}")


@pytest.mark.parametrize("page", PAGES)
def test_internal_links_resolve(page):
    html = (DOCS / page).read_text()
    for m in re.finditer(r'(?:href|src)="([^"#][^"]*?)(?:#[^"]*)?"', html):
        target = m.group(1)
        assert (DOCS / target).exists(), f"{page} links to missing {target}"


def test_anchors_resolve():
    for page in PAGES:
        html = (DOCS / page).read_text()
        ids = set(re.findall(r'id="([^"]+)"', html))
        for m in re.finditer(rf'href="{re.escape(page)}#([^"]+)"', html):
            assert m.group(1) in ids, f"{page}: dangling anchor #{m.group(1)}"
        for m in re.finditer(r'href="#([^"]+)"', html):
            assert m.group(1) in ids
