#!/usr/bin/env python
"""Generate the small sample PDFs used by the RAGAS evaluation harness.

These fictional GreenLeaf Energy documents give the evaluation set a stable,
self-contained ground truth. Regenerate with:

    python scripts/make_sample_docs.py

Requires ``reportlab`` (dev-only; not needed at runtime since the generated
PDFs are committed under ``data/eval/docs/``).
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "docs"

DOCS: dict[str, list[str]] = {
    "greenleaf_solar_panel.pdf": [
        "GreenLeaf SunMax 400 Solar Panel - Product Guide",
        "GreenLeaf Energy manufactures the SunMax 400, a residential solar "
        "panel built around monocrystalline silicon cells.",
        "The SunMax 400 has a rated power output of 400 watts and a module "
        "efficiency of 21.5 percent.",
        "Each panel weighs 21 kilograms and measures 1.7 meters by 1.0 meters.",
        "The SunMax 400 operates reliably across a temperature range of minus "
        "40 degrees Celsius to 85 degrees Celsius.",
        "GreenLeaf backs the SunMax 400 with a 25 year performance warranty.",
        "The panel is designed to pair directly with the GreenLeaf PowerCell "
        "10 home battery for whole-home backup.",
    ],
    "greenleaf_battery.pdf": [
        "GreenLeaf PowerCell 10 Home Battery - Technical Specification",
        "The PowerCell 10 is a lithium iron phosphate (LiFePO4) home battery "
        "produced by GreenLeaf Energy.",
        "It has a nominal capacity of 10 kilowatt-hours and a usable capacity "
        "of 9.2 kilowatt-hours.",
        "The PowerCell 10 delivers a maximum continuous power of 5 kilowatts.",
        "Its round-trip efficiency is 94 percent.",
        "GreenLeaf provides a 10 year warranty on the PowerCell 10.",
        "The battery is compatible with the SunMax 400 solar panel and can be "
        "stacked in multiples for larger storage needs.",
    ],
    "greenleaf_company.pdf": [
        "GreenLeaf Energy - Company Overview and Sustainability Report",
        "GreenLeaf Energy was founded in 2015 and is headquartered in Austin, "
        "Texas.",
        "As of 2024 the company employs 320 people and reported annual revenue "
        "of 85 million US dollars.",
        "GreenLeaf's mission is to make affordable clean energy accessible to "
        "every household.",
        "Since 2020 the company has reduced its manufacturing carbon footprint "
        "by 40 percent.",
        "GreenLeaf recycles 95 percent of its manufacturing waste.",
        "The company now sells its products in 12 countries across three "
        "continents.",
    ],
}


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    for filename, paragraphs in DOCS.items():
        path = OUT_DIR / filename
        doc = SimpleDocTemplate(str(path), pagesize=LETTER)
        story = []
        story.append(Paragraph(paragraphs[0], styles["Title"]))
        story.append(Spacer(1, 18))
        for para in paragraphs[1:]:
            story.append(Paragraph(para, styles["BodyText"]))
            story.append(Spacer(1, 10))
        doc.build(story)
        print(f"Wrote {path}")


if __name__ == "__main__":
    build()
