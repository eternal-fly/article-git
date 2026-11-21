"""Non-rasterizing structure checks and an ordered PDF bundle of Python figures."""
import re
import xml.etree.ElementTree as ET
import fitz
from common import *


def main():
    folder = HERE / "figures"
    files = sorted(folder.glob("0[1-7]_*.pdf"))
    assert len(files) == 7, f"Expected seven finished figures, found {len(files)}"
    records = []
    combined = fitz.open()
    for path in files:
        with fitz.open(path) as doc:
            assert len(doc) == 1
            page = doc[0]
            text = page.get_text()
            assert len(text) > 80 and "\ufffd" not in text
            assert not page.get_images(), f"Unexpected raster in data figure {path}"
            fonts = page.get_fonts(full=True)
            assert fonts
            for font in fonts:
                assert len(doc.extract_font(font[0])[-1]) > 0, f"Font not embedded: {font}"
            root = ET.parse(path.with_suffix(".svg")).getroot()
            ns = {"svg": "http://www.w3.org/2000/svg"}
            texts = root.findall(".//svg:text", ns)
            assert len(texts) >= 10 and not root.findall(".//svg:image", ns)
            qa = json.loads(path.with_name(path.stem + "_qa.json").read_text())
            assert qa["full_frames"] and not qa["text_outside_canvas"]
            records.append(dict(figure=path.stem, pages=1, live_pdf_text_characters=len(text),
                svg_text_elements=len(texts), embedded_font_count=len(fonts), raster_images=0,
                canvas_points=[page.rect.width, page.rect.height], full_frames=True,
                text_outside_canvas=[]))
            combined.insert_pdf(doc)
    combined.save(folder / "SUPPLEMENT_FIGURES.pdf")
    combined.close()
    write_json(folder / "export_audit.json", dict(complete=True, figures=records,
        visual_review="PNG previews opened and visually inspected by the assistant; structural checks do not establish statistical correctness."))
    # Reader-facing Markdown links must resolve to existing local files.
    missing = []
    for name in ("REPORT_CN.md", "README_CN.md"):
        body = (HERE / name).read_text()
        for target in re.findall(r"\]\((/[^)]+)\)", body):
            if not Path(target).exists():
                missing.append(dict(document=name, missing=target))
    assert not missing, missing
    import hashlib
    manifest = []
    for path in sorted(HERE.rglob("*")):
        relative = path.relative_to(HERE)
        if not path.is_file() or "__pycache__" in relative.parts or ".mplconfig" in relative.parts:
            continue
        if path.name == "artifact_manifest.csv":
            continue
        assert ".partial." not in path.name, f"Unfinished output: {path}"
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                sha.update(chunk)
        manifest.append(dict(path=str(relative), bytes=path.stat().st_size, sha256=sha.hexdigest()))
    write_csv(HERE / "artifact_manifest.csv", manifest)
    print("PASS: seven vector figures, embedded fonts, live text, no raster data panels; report links valid.", flush=True)


if __name__ == "__main__":
    main()
