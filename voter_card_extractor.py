#!/usr/bin/env python3
"""
Voter Card Extractor - Single File
===================================
Extracts voter cards from electoral roll PDFs using:
  - OpenCV Hough line detection to detect card boundaries
  - ONNX OCR API (https://dev-ai.zoop.one/paddle/onnx/base) for text extraction
  - Regex-based field parsing to extract structured data

Usage:
    python voter_card_extractor.py                     # process pages 3-42, output to voter_cards_output/
    python voter_card_extractor.py --start 5 --end 10  # process pages 5-10
    python voter_card_extractor.py --page 7            # process single page
    python voter_card_extractor.py --pdf myfile.pdf    # use different PDF

Output:
    voter_cards_output/page_XXXX_voter_cards.txt       # formatted voter cards per page
    voter_cards_output/extraction_summary.json         # timing + stats for all pages
"""

import asyncio
import argparse
import base64
import json
import re
import os
import shutil
import tempfile
import time

import cv2
import httpx
import numpy as np
from pdf2image import convert_from_path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ONNX_API_URL = "https://dev-ai.zoop.one/paddle/onnx/base"
ONNX_API_KEY = "***REMOVED***"
PDF_DPI      = 300
DEFAULT_PDF  = "2024-FC-EROLLGEN-S13-206-FinalRoll-Revision2-ENG-10-WI.pdf"
OUTPUT_DIR   = "voter_cards_output"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: PDF → IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def pdf_page_to_image(pdf_path: str, page_num: int, out_path: str) -> str:
    """Convert a single PDF page to a PNG image at PDF_DPI resolution."""
    images = convert_from_path(
        pdf_path,
        dpi=PDF_DPI,
        first_page=page_num,
        last_page=page_num,
    )
    images[0].save(out_path, "PNG")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: OPENCV CARD DETECTION
# Detects the bounding box of each voter card on the page using Hough lines.
# Electoral roll pages have a 3-column × 10-row grid of cards separated by
# printed border lines.  We filter those long lines and build the grid.
# ─────────────────────────────────────────────────────────────────────────────

def _group_nearby(values: list, threshold: int = 20) -> list:
    """Cluster pixel coordinates that are within `threshold` px of each other."""
    if not values:
        return []
    grouped, bucket = [], [values[0]]
    for v in values[1:]:
        if v - bucket[-1] < threshold:
            bucket.append(v)
        else:
            grouped.append(int(np.mean(bucket)))
            bucket = [v]
    grouped.append(int(np.mean(bucket)))
    return grouped


def _merge_vertical_boxes(boxes: list, min_card_height: int = 80, max_gap: int = 5) -> list:
    """
    Merge vertically adjacent partial boxes in the same column.

    Some pages have a 'Photo Available' divider that splits one card into two
    thin boxes.  We merge them back into a single card box whenever:
      - the gap between them is <= max_gap pixels, AND
      - at least one of them is shorter than min_card_height pixels.
    """
    by_col: dict = {}
    for box in boxes:
        by_col.setdefault(box["col"], []).append(box)

    merged = []
    for col_boxes in by_col.values():
        col_boxes.sort(key=lambda b: b["y"])
        i = 0
        while i < len(col_boxes):
            cur = col_boxes[i].copy()
            if i + 1 < len(col_boxes):
                nxt = col_boxes[i + 1]
                gap = nxt["y"] - (cur["y"] + cur["height"])
                is_partial = cur["height"] < min_card_height or nxt["height"] < min_card_height
                if gap <= max_gap and is_partial:
                    cur = {
                        "x":      min(cur["x"], nxt["x"]),
                        "y":      cur["y"],
                        "width":  max(cur["x"] + cur["width"], nxt["x"] + nxt["width"])
                                  - min(cur["x"], nxt["x"]),
                        "height": (nxt["y"] + nxt["height"]) - cur["y"],
                        "row":    cur["row"],
                        "col":    cur["col"],
                    }
                    i += 2
                    merged.append(cur)
                    continue
            merged.append(cur)
            i += 1
    return merged


def detect_card_boxes(image_path: str) -> list:
    """
    Return a list of card bounding-box dicts  {x, y, width, height, row, col}
    detected from the page image via filtered Hough line transform.

    Algorithm
    ---------
    1. Canny edge detection on greyscale image.
    2. Probabilistic Hough line transform → raw line segments.
    3. Keep only:
       - Horizontal lines spanning >= 70 % of page width  (card row borders)
       - Vertical   lines spanning >= 50 % of page height (card column borders)
    4. Group nearby parallel lines (within 20 px) → grid coordinates.
    5. Build bounding boxes from every grid cell.
    6. Keep only the 3 widest columns (the actual card columns).
    7. Merge partial boxes caused by internal separators.
    8. Cap at 30 boxes (max cards per page).
    """
    img    = cv2.imread(image_path)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape

    # --- edge & line detection ---
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=50,
    )
    if lines is None:
        return []

    h_ys, v_xs = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10 and abs(x2 - x1) >= w * 0.70:   # long horizontal
            h_ys.append((y1 + y2) // 2)
        elif abs(x2 - x1) < 10 and abs(y2 - y1) >= h * 0.50: # long vertical
            v_xs.append((x1 + x2) // 2)

    h_lines = _group_nearby(sorted(set(h_ys)))
    v_lines = _group_nearby(sorted(set(v_xs)))

    # --- build grid cells ---
    boxes = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            bw = v_lines[j + 1] - v_lines[j]
            bh = h_lines[i + 1] - h_lines[i]
            if bw > 50 and bh > 50:
                boxes.append({
                    "x": v_lines[j], "y": h_lines[i],
                    "width": bw, "height": bh,
                    "row": i + 1, "col": j + 1,
                })

    # --- keep the 3 widest columns ---
    if len(v_lines) > 3:
        col_widths: dict = {}
        for b in boxes:
            col_widths.setdefault(b["col"], []).append(b["width"])
        avg_w = {c: sum(ws) / len(ws) for c, ws in col_widths.items()}
        top3  = sorted(avg_w, key=avg_w.get, reverse=True)[:3]
        boxes = [b for b in boxes if b["col"] in top3]

    # --- merge partial / split boxes ---
    boxes = _merge_vertical_boxes(boxes, min_card_height=80, max_gap=5)

    # --- sort and cap ---
    boxes.sort(key=lambda b: (b["y"], b["x"]))
    return boxes[:30]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: ONNX OCR API CALL
# ─────────────────────────────────────────────────────────────────────────────

def _parse_onnx_response(data: dict, x_offset: int = 0, y_offset: int = 0) -> list:
    """
    Parse raw ONNX API response into a flat list of text-block dicts.
    x_offset / y_offset are added to bbox coordinates — used when the image
    sent was a crop so we can map coordinates back to full-page space.
    """
    blocks = []
    for item in data.get("result", []):
        if len(item) < 2:
            continue
        bbox_pts, text_info = item
        text       = text_info[0] if isinstance(text_info, list) else text_info
        confidence = text_info[1] if isinstance(text_info, list) and len(text_info) > 1 else 0.0
        if bbox_pts and len(bbox_pts) >= 4:
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            blocks.append({
                "text":       text.strip(),
                "bbox":       [min(xs) + x_offset, min(ys) + y_offset,
                               max(xs) + x_offset, max(ys) + y_offset],
                "confidence": confidence,
            })
    return blocks


async def onnx_ocr(image_path: str) -> list:
    """POST the full page image to ONNX OCR. Returns text-block dicts."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            ONNX_API_URL,
            headers={"API-KEY": ONNX_API_KEY, "Content-Type": "application/json"},
            json={"image": b64, "detect": True},
        )
        resp.raise_for_status()

    return _parse_onnx_response(resp.json())


async def onnx_ocr_crop(image_path: str, card: dict) -> list:
    """
    Crop just the card region from the page image and OCR it.

    When the full-page OCR misses a text line (e.g. faint Husband's Name),
    sending a focused crop of just that card lets ONNX detect it reliably.

    Coordinates in the returned blocks are offset back to full-page space
    so they are interchangeable with full-page OCR blocks.
    """
    pad = 10
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    x  = max(0, card["x"] - pad)
    y  = max(0, card["y"] - pad)
    x2 = min(w_img, card["x"] + card["width"]  + pad)
    y2 = min(h_img, card["y"] + card["height"] + pad)

    crop = img[y:y2, x:x2]
    _, buf = cv2.imencode(".png", crop)
    b64 = base64.b64encode(buf.tobytes()).decode()

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            ONNX_API_URL,
            headers={"API-KEY": ONNX_API_KEY, "Content-Type": "application/json"},
            json={"image": b64, "detect": True},
        )
        resp.raise_for_status()

    # Offset coords back to full-page space
    return _parse_onnx_response(resp.json(), x_offset=x, y_offset=y)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: MAP TEXT BLOCKS → CARD REGIONS
# ─────────────────────────────────────────────────────────────────────────────

def texts_in_card(card: dict, all_blocks: list) -> list:
    """
    Return text blocks whose centre falls inside the card's bounding box,
    sorted top-to-bottom then left-to-right.
    """
    cx, cy, cw, ch = card["x"], card["y"], card["width"], card["height"]
    inside = []
    for blk in all_blocks:
        x0, y0, x1, y1 = blk["bbox"]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if cx <= mx <= cx + cw and cy <= my <= cy + ch:
            inside.append({"text": blk["text"], "x": x0, "y": y0})
    inside.sort(key=lambda b: (b["y"], b["x"]))
    return inside


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: FIELD EXTRACTION VIA REGEX
#
# Each voter card on the printed roll has this layout (translated to English):
#
#   <serial>  <VoterID>
#   Name      : <Full Name>
#   Father's / Husband's Name : <Relation Name>
#   House Number : <HouseNo>          Photo
#   Age : <age>   Gender : <gender>   Available
#
# The OCR text from ONNX comes back as separate tokens per text block.
# We join all tokens for a card into one string and run regex patterns.
# ─────────────────────────────────────────────────────────────────────────────

# Voter ID  — 2-4 uppercase letters followed by 6-10 digits
_RE_VOTER_ID  = re.compile(r'\b([A-Z]{2,4}\d{6,10})\b')

# Name field — "Name :" then the name up to the next labelled field
_RE_NAME      = re.compile(
    r'Name\s*:\s*([A-Z][A-Za-z\s]+?)(?=\s+(?:Father|Husband|Mother|House|Age|Photo)|$)',
    re.IGNORECASE,
)

# Father's name
_RE_FATHER    = re.compile(
    r"Father'?s?\s+Name\s*:\s*([A-Za-z\s]+?)(?=\s+House|\s+Photo|\s+Age|$)",
    re.IGNORECASE,
)

# Husband's name
_RE_HUSBAND   = re.compile(
    r"Husband'?s?\s+Name\s*:\s*([A-Za-z\s]+?)(?=\s+House|\s+Photo|\s+Age|$)",
    re.IGNORECASE,
)

# Age — "Age : <digits>"
_RE_AGE       = re.compile(r'Age\s*:\s*(\d{1,3})', re.IGNORECASE)

# Gender
_RE_GENDER_M  = re.compile(r'Gender\s*:\s*Male',   re.IGNORECASE)
_RE_GENDER_F  = re.compile(r'Gender\s*:\s*Female', re.IGNORECASE)

# House number — value after "House Number :" that isn't a keyword
_RE_HOUSE     = re.compile(r'House\s+Number\s*:\s*([A-Z0-9/\-]+)', re.IGNORECASE)
_HOUSE_SKIP   = {"Photo", "Available", "Age", "Gender", "Name"}


def parse_card(texts: list) -> dict:
    """
    Parse one card's sorted text-block list into a structured dict.

    Returns
    -------
    {voter_id, serial_no, name, relation_type, relation_name,
     house_number, age, gender}
    """
    joined = " ".join(t["text"] for t in texts)

    card = {
        "voter_id":      "",
        "serial_no":     "",
        "name":          "",
        "relation_type": "",
        "relation_name": "",
        "house_number":  "",
        "age":           "",
        "gender":        "",
    }

    # --- voter ID ---
    m = _RE_VOTER_ID.search(joined)
    if m:
        card["voter_id"] = m.group(1)

    # --- serial number (digit(s) just before the voter ID) ---
    if card["voter_id"]:
        pat = re.compile(r'\b(\d{1,2})\s+' + re.escape(card["voter_id"]))
        m = pat.search(joined)
        if m:
            card["serial_no"] = m.group(1)

    # --- name ---
    m = _RE_NAME.search(joined)
    if m:
        name = re.sub(r'\s+(Father|Husband|Mother).*$', '', m.group(1), flags=re.IGNORECASE)
        card["name"] = name.strip()

    # --- relation ---
    m = _RE_FATHER.search(joined)
    if m:
        card["relation_type"] = "Father"
        card["relation_name"] = m.group(1).strip()
    else:
        m = _RE_HUSBAND.search(joined)
        if m:
            card["relation_type"] = "Husband"
            card["relation_name"] = m.group(1).strip()

    # --- age ---
    m = _RE_AGE.search(joined)
    if m:
        card["age"] = m.group(1)

    # --- gender ---
    if _RE_GENDER_M.search(joined):
        card["gender"] = "Male"
    elif _RE_GENDER_F.search(joined):
        card["gender"] = "Female"

    # --- house number ---
    m = _RE_HOUSE.search(joined)
    if m and m.group(1).strip() not in _HOUSE_SKIP:
        card["house_number"] = m.group(1).strip()

    return card


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: FORMAT OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def format_page(cards: list, page_num: int) -> tuple:
    """
    Render the list of card dicts as a formatted text string.

    Returns (formatted_text, valid_card_count)
    """
    lines = [
        "=" * 80,
        f"VOTER CARDS - PAGE {page_num}",
        "=" * 80,
        "",
    ]

    valid = 0
    for i, card in enumerate(cards, 1):
        if not card["voter_id"]:
            continue
        valid += 1
        rel_label = f"  {card['relation_type']}'s Name" if card["relation_type"] else None

        lines += [
            "━" * 80,
            f"CARD #{i:02d}",
            "━" * 80,
            f"  Voter ID         : {card['voter_id']}",
        ]
        if card["serial_no"]:
            lines.append(f"  Serial No        : {card['serial_no']}")
        lines.append(f"  Name             : {card['name']}")
        if rel_label:
            lines.append(f"{rel_label:<20}: {card['relation_name']}")
        lines += [
            f"  House Number     : {card['house_number'] or 'N/A'}",
            f"  Age              : {card['age']}",
            f"  Gender           : {card['gender']}",
            f"  Photo Available  : Yes",
            "",
        ]

    lines += ["=" * 80, f"TOTAL CARDS: {valid}", "=" * 80]
    return "\n".join(lines), valid


# ─────────────────────────────────────────────────────────────────────────────
# PAGE PROCESSOR  (all steps combined)
# ─────────────────────────────────────────────────────────────────────────────

async def process_page(pdf_path: str, page_num: int, output_dir: str, tmp_dir: str) -> dict:
    """
    Full pipeline for one page.  Returns a result dict with timing and counts.
    """
    t = {}
    t0 = time.time()
    result = {"page": page_num, "success": False, "total_cards": 0, "timing": t, "error": None}

    try:
        image_path = os.path.join(tmp_dir, f"page_{page_num:04d}.png")

        # 1 — PDF → image
        s = time.time()
        pdf_page_to_image(pdf_path, page_num, image_path)
        t["pdf_to_image"] = time.time() - s

        # 2 — card detection
        s = time.time()
        card_boxes = detect_card_boxes(image_path)
        t["card_detection"] = time.time() - s

        # 3 — ONNX OCR
        s = time.time()
        text_blocks = await onnx_ocr(image_path)
        t["onnx_ocr"] = time.time() - s

        # 4 — match text → cards + parse fields
        s = time.time()
        cards = [parse_card(texts_in_card(box, text_blocks)) for box in card_boxes]
        t["text_matching_and_parsing"] = time.time() - s

        # 4b — fallback crop OCR for cards with missing relation_name
        #      Full-page OCR can drop faint text lines; cropping the individual
        #      card and re-sending it to ONNX reliably recovers them.
        s = time.time()
        fallback_count = 0
        for i, (card, box) in enumerate(zip(cards, card_boxes)):
            if not card["relation_name"]:
                crop_blocks = await onnx_ocr_crop(image_path, box)
                crop_texts  = texts_in_card(box, crop_blocks)
                reparsed    = parse_card(crop_texts)
                if reparsed["relation_name"]:
                    # keep all other fields from original parse if better
                    cards[i]["relation_type"] = reparsed["relation_type"]
                    cards[i]["relation_name"] = reparsed["relation_name"]
                    fallback_count += 1
        t["fallback_crop_ocr"] = time.time() - s
        t["fallback_count"]    = fallback_count

        # 5 — format + save
        s = time.time()
        formatted, valid = format_page(cards, page_num)
        out_path = os.path.join(output_dir, f"page_{page_num:04d}_voter_cards.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(formatted)
        t["format_and_save"] = time.time() - s

        t["total"] = time.time() - t0
        result.update({"success": True, "total_cards": valid})

    except Exception as exc:
        t["total"] = time.time() - t0
        result["error"] = str(exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  — batch processing
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Extract voter cards from electoral roll PDF.")
    parser.add_argument("--pdf",   default=DEFAULT_PDF,   help="Path to PDF file")
    parser.add_argument("--start", type=int, default=3,   help="First page to process (default: 3)")
    parser.add_argument("--end",   type=int, default=42,  help="Last page to process (default: 42)")
    parser.add_argument("--page",  type=int, default=None,help="Process a single page only")
    parser.add_argument("--out",   default=OUTPUT_DIR,    help="Output directory")
    args = parser.parse_args()

    pdf_path   = args.pdf
    out_dir    = args.out
    start_page = args.page if args.page else args.start
    end_page   = args.page if args.page else args.end

    # Prepare output directory
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    tmp_dir = tempfile.mkdtemp(prefix="voter_extractor_")

    total_pages = end_page - start_page + 1
    print("=" * 80)
    print("VOTER CARD EXTRACTOR — ONNX OCR + OPENCV")
    print("=" * 80)
    print(f"PDF    : {pdf_path}")
    print(f"Pages  : {start_page} → {end_page}  ({total_pages} pages)")
    print(f"Output : {out_dir}/")
    print("=" * 80)

    overall_start = time.time()
    results = []

    for page_num in range(start_page, end_page + 1):
        print(f"\n[{page_num}/{end_page}] Processing page {page_num} ...", end=" ", flush=True)
        result = await process_page(pdf_path, page_num, out_dir, tmp_dir)
        results.append(result)

        if result["success"]:
            t = result["timing"]
            fb = t.get("fallback_count", 0)
            fb_str = f"  fallback {fb}" if fb else ""
            print(
                f"✓  {result['total_cards']} cards  |  "
                f"total {t['total']:.1f}s  (ocr {t['onnx_ocr']:.1f}s{fb_str})"
            )
        else:
            print(f"✗  FAILED — {result['error']}")

    shutil.rmtree(tmp_dir)
    overall_time = time.time() - overall_start

    # ── summary ──────────────────────────────────────────────────────────────
    ok  = [r for r in results if r["success"]]
    bad = [r for r in results if not r["success"]]

    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Pages processed    : {len(results)}")
    print(f"Successful         : {len(ok)}")
    print(f"Failed             : {len(bad)}")

    if ok:
        total_cards  = sum(r["total_cards"] for r in ok)
        avg_total    = sum(r["timing"]["total"]    for r in ok) / len(ok)
        avg_ocr      = sum(r["timing"]["onnx_ocr"] for r in ok) / len(ok)
        avg_cv       = sum(r["timing"]["card_detection"] for r in ok) / len(ok)
        avg_fb       = sum(r["timing"].get("fallback_crop_ocr", 0) for r in ok) / len(ok)
        total_fb     = sum(r["timing"].get("fallback_count", 0) for r in ok)
        print(f"Total voter cards  : {total_cards}")
        print(f"Fallback crop OCR  : {total_fb} cards recovered")
        print()
        print(f"{'Stage':<35} {'Avg time':>10}")
        print("-" * 47)
        print(f"{'PDF → Image':<35} {avg_total - avg_ocr - avg_cv - avg_fb:>9.2f}s")
        print(f"{'Card Detection (OpenCV)':<35} {avg_cv:>9.2f}s")
        print(f"{'OCR API Call (ONNX, full page)':<35} {avg_ocr:>9.2f}s")
        print(f"{'Fallback Crop OCR (ONNX)':<35} {avg_fb:>9.2f}s")
        print(f"{'Text Matching + Parsing':<35} {'< 0.01':>9}s")
        print("-" * 47)
        print(f"{'Total per page':<35} {avg_total:>9.2f}s")
        print()

    print(f"Wall-clock time    : {overall_time:.1f}s ({overall_time / 60:.1f} min)")

    if bad:
        print("\nFailed pages:")
        for r in bad:
            print(f"  page {r['page']}: {r['error']}")

    # save summary
    summary_path = os.path.join(out_dir, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "pdf":              pdf_path,
            "start_page":       start_page,
            "end_page":         end_page,
            "total_pages":      len(results),
            "successful_pages": len(ok),
            "failed_pages":     len(bad),
            "total_cards":      sum(r["total_cards"] for r in ok) if ok else 0,
            "overall_time_s":   overall_time,
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
            "pages":            results,
        }, f, indent=2)

    print(f"\n✓ Summary  → {summary_path}")
    print(f"✓ Cards    → {out_dir}/page_XXXX_voter_cards.txt")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
