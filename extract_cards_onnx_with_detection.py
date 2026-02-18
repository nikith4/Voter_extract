#!/usr/bin/env python3
"""
Extract voter cards using ONNX OCR with OpenCV card detection.
Hybrid approach: Card detection (like Google Vision) + ONNX OCR for text.
"""

import asyncio
import base64
import json
import re
import cv2
import numpy as np
import tempfile
import httpx
import time
import logging
from pdf2image import convert_from_path
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_voter_boxes_opencv_improved import extract_cards_by_filtered_lines


async def _call_onnx_api_with_retry(url, headers, payload, max_retries=5, base_delay=2.0):
    """
    Call ONNX API with retry logic and exponential backoff.

    Handles:
    - Rate limiting (429) with exponential backoff
    - Network errors with retry
    - Timeout errors with retry

    Args:
        url: API endpoint
        headers: Request headers
        payload: Request payload
        max_retries: Maximum retry attempts (default: 5)
        base_delay: Base delay in seconds, doubles each retry (default: 2.0s)

    Returns:
        API response JSON
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)

                # Check for rate limiting
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"⚠️  ONNX API rate limited (429). "
                            f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"❌ Rate limit exceeded after {max_retries} attempts")
                        response.raise_for_status()

                # Raise for other HTTP errors
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"⚠️  ONNX API timeout. "
                    f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"❌ ONNX API timeout after {max_retries} attempts")
                raise

        except httpx.NetworkError as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"⚠️  Network error: {e}. "
                    f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"❌ Network error after {max_retries} attempts: {e}")
                raise

        except httpx.HTTPStatusError as e:
            # Retry 5xx server errors (502, 503, 504, etc.)
            if e.response.status_code >= 500:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"⚠️  ONNX API server error ({e.response.status_code}). "
                        f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ Server error {e.response.status_code} after {max_retries} attempts")
                    raise
            else:
                # 4xx client errors (except 429) don't retry
                logger.error(f"❌ ONNX API HTTP error: {e.response.status_code}")
                raise

        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"⚠️  ONNX API error: {type(e).__name__}. "
                    f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"❌ ONNX API failed after {max_retries} attempts: {e}")
                raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


async def call_onnx_ocr(image_path):
    """Call ONNX OCR API with retry logic and return raw results."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    url = "https://dev-ai.zoop.one/paddle/onnx/base"
    headers = {
        'API-KEY': os.getenv('ONNX_OCR_API_KEY'),
        'Content-Type': 'application/json'
    }
    payload = {
        "image": image_base64,
        "detect": True
    }

    return await _call_onnx_api_with_retry(url, headers, payload)


async def call_onnx_ocr_crop(image_path, card_box):
    """
    Crop just the card region and OCR it.

    Fallback method when full-page OCR misses text (e.g. faint relation_name).
    Sending a focused crop of just that card lets ONNX detect it reliably.

    Args:
        image_path: Path to full page image
        card_box: Card bounding box dict with {x, y, width, height}

    Returns:
        Text blocks with coordinates offset back to full-page space
    """
    pad = 10
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    # Extract card bounds
    if isinstance(card_box, dict):
        card_x = card_box['x']
        card_y = card_box['y']
        card_w = card_box['width']
        card_h = card_box['height']
    else:
        card_x, card_y, card_w, card_h = card_box

    # Calculate crop region with padding
    x = max(0, card_x - pad)
    y = max(0, card_y - pad)
    x2 = min(w_img, card_x + card_w + pad)
    y2 = min(h_img, card_y + card_h + pad)

    # Crop and encode
    crop = img[y:y2, x:x2]
    _, buf = cv2.imencode('.png', crop)
    crop_base64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    # Call ONNX API on crop with retry logic
    url = "https://dev-ai.zoop.one/paddle/onnx/base"
    headers = {
        'API-KEY': os.getenv('ONNX_OCR_API_KEY'),
        'Content-Type': 'application/json'
    }
    payload = {
        "image": crop_base64,
        "detect": True
    }

    crop_result = await _call_onnx_api_with_retry(url, headers, payload)

    # Parse result and offset coordinates back to full-page space
    text_blocks = []
    if 'result' in crop_result:
        for item in crop_result['result']:
            if len(item) >= 2:
                bbox, text_info = item
                text = text_info[0] if isinstance(text_info, list) else text_info
                confidence = text_info[1] if isinstance(text_info, list) and len(text_info) > 1 else 0.0

                if bbox and len(bbox) >= 4:
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]

                    # Offset back to full-page coordinates
                    text_blocks.append({
                        'text': text.strip(),
                        'bbox': [
                            min(x_coords) + x,
                            min(y_coords) + y,
                            max(x_coords) + x,
                            max(y_coords) + y
                        ],
                        'confidence': confidence
                    })

    return text_blocks


def get_text_blocks_from_onnx(onnx_result):
    """Extract text blocks with bounding boxes from ONNX result."""
    text_blocks = []

    if 'result' not in onnx_result:
        return text_blocks

    for item in onnx_result['result']:
        if len(item) >= 2:
            bbox, text_info = item
            text = text_info[0] if isinstance(text_info, list) else text_info
            confidence = text_info[1] if isinstance(text_info, list) and len(text_info) > 1 else 0.0

            # Convert bbox to simpler format [x_min, y_min, x_max, y_max]
            if bbox and len(bbox) >= 4:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)

                text_blocks.append({
                    'text': text.strip(),
                    'bbox': [x_min, y_min, x_max, y_max],
                    'confidence': confidence
                })

    return text_blocks


def get_text_in_card(card_box, text_blocks):
    """Get all text blocks within a card boundary."""
    card_texts = []

    # card_box is a dictionary with keys: x, y, width, height, row, col
    if isinstance(card_box, dict):
        if 'x' not in card_box:
            print(f"Warning: card_box missing 'x' key: {card_box}")
            return []
        card_x = card_box['x']
        card_y = card_box['y']
        card_w = card_box['width']
        card_h = card_box['height']
    elif isinstance(card_box, (list, tuple)) and len(card_box) == 4:
        # Format: [x, y, w, h]
        card_x, card_y, card_w, card_h = card_box
    else:
        print(f"Warning: Unexpected card_box format: {type(card_box)}, {card_box}")
        return []

    for block in text_blocks:
        # block['bbox'] format: [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = block['bbox']

        # Get center point of text block
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Check if center is within card
        if (card_x <= center_x <= card_x + card_w and
            card_y <= center_y <= card_y + card_h):
            card_texts.append({
                'text': block['text'],
                'y': y_min,
                'x': x_min
            })

    # Sort by position (top to bottom, left to right)
    card_texts.sort(key=lambda b: (b['y'], b['x']))

    return card_texts


def parse_card_fields(card_texts):
    """Extract structured fields from card texts."""
    # Combine all text
    all_text = ' '.join([t['text'] for t in card_texts])

    card = {
        'name': '',
        'relation_type': '',
        'relation_name': '',
        'house_number': '',
        'age': '',
        'gender': '',
        'voter_id': '',
        'serial_no': ''
    }

    # Extract voter ID
    voter_id_pattern = re.compile(r'\b([A-Z]{2,4}\d{6,10})\b')
    vid_match = voter_id_pattern.search(all_text)
    if vid_match:
        card['voter_id'] = vid_match.group(1)

    # Extract serial number (1-2 digits before OR after voter ID)
    if card['voter_id']:
        # Try serial before voter_id: "1 SKX3772639"
        serial_match = re.search(r'\b(\d{1,2})\s+' + re.escape(card['voter_id']), all_text)
        if serial_match:
            card['serial_no'] = serial_match.group(1)
        else:
            # Try serial after voter_id: "SKX3772639 2"
            serial_match = re.search(re.escape(card['voter_id']) + r'\s+(\d{1,2})\b', all_text)
            if serial_match:
                card['serial_no'] = serial_match.group(1)

    # Extract name
    name_match = re.search(r'Name\s*:\s*([A-Z][A-Za-z\s]+?)(?:\s+(?:Father|Husband|Mother|House|Age)|$)', all_text, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        # Clean up name
        name = re.sub(r'\s+(Father|Husband|Mother).*$', '', name, flags=re.IGNORECASE)
        card['name'] = name.strip()

    # Extract relation
    father_match = re.search(r"Father'?s?\s+Name\s*:\s*([A-Za-z\s]+?)(?:\s+House|\s+Photo|\s+Age|$)", all_text, re.IGNORECASE)
    husband_match = re.search(r"Husband'?s?\s+Name\s*:\s*([A-Za-z\s]+?)(?:\s+House|\s+Photo|\s+Age|$)", all_text, re.IGNORECASE)

    if father_match:
        card['relation_type'] = "Father"
        card['relation_name'] = father_match.group(1).strip()
    elif husband_match:
        card['relation_type'] = "Husband"
        card['relation_name'] = husband_match.group(1).strip()

    # Extract age
    age_match = re.search(r'Age\s*:\s*(\d{1,3})', all_text, re.IGNORECASE)
    if age_match:
        card['age'] = age_match.group(1)

    # Extract gender
    if re.search(r'Gender\s*:\s*Male', all_text, re.IGNORECASE):
        card['gender'] = 'Male'
    elif re.search(r'Gender\s*:\s*Female', all_text, re.IGNORECASE):
        card['gender'] = 'Female'

    # Extract house number
    house_match = re.search(r'House\s+Number\s*:\s*([A-Z0-9/\-]+)', all_text, re.IGNORECASE)
    if house_match:
        house_num = house_match.group(1).strip()
        if house_num not in ['Photo', 'Available', 'Age', 'Gender', 'Name']:
            card['house_number'] = house_num

    return card


async def extract_cards_with_fallback(image_path, card_boxes, text_blocks):
    """
    Extract cards with fallback crop OCR for missing data.

    When full-page OCR misses text (serial_no, name, relation_name, etc.),
    this function crops just that card and re-OCRs it to recover missing fields.

    Args:
        image_path: Path to page image
        card_boxes: List of card bounding boxes
        text_blocks: Text blocks from full-page OCR

    Returns:
        Tuple of (cards_list, fallback_count)
    """
    cards = []
    fallback_count = 0

    for card_box in card_boxes:
        # Initial parse with full-page OCR
        card_texts = get_text_in_card(card_box, text_blocks)
        card = parse_card_fields(card_texts)

        # Check if any critical fields are missing (but has voter_id)
        # Note: house_number is often legitimately empty, so we don't check it
        missing_fields = []
        if card['voter_id']:
            if not card['serial_no']:
                missing_fields.append('serial_no')
            if not card['name']:
                missing_fields.append('name')
            if not card['relation_name']:
                missing_fields.append('relation_name')
            if not card['age']:
                missing_fields.append('age')
            if not card['gender']:
                missing_fields.append('gender')

        # Fallback: If any critical fields are missing, try crop OCR
        if missing_fields:
            try:
                crop_blocks = await call_onnx_ocr_crop(image_path, card_box)
                crop_texts = get_text_in_card(card_box, crop_blocks)
                reparsed = parse_card_fields(crop_texts)

                # Update any fields that were missing but are now found
                recovered = []
                if not card['serial_no'] and reparsed['serial_no']:
                    card['serial_no'] = reparsed['serial_no']
                    recovered.append('serial_no')
                if not card['name'] and reparsed['name']:
                    card['name'] = reparsed['name']
                    recovered.append('name')
                if not card['relation_name'] and reparsed['relation_name']:
                    card['relation_type'] = reparsed['relation_type']
                    card['relation_name'] = reparsed['relation_name']
                    recovered.append('relation_name')
                if not card['age'] and reparsed['age']:
                    card['age'] = reparsed['age']
                    recovered.append('age')
                if not card['gender'] and reparsed['gender']:
                    card['gender'] = reparsed['gender']
                    recovered.append('gender')
                if not card['house_number'] and reparsed['house_number']:
                    card['house_number'] = reparsed['house_number']
                    recovered.append('house_number')

                if recovered:
                    fallback_count += 1

            except Exception as e:
                # Fallback failed, keep original card
                pass

        cards.append(card)

    return cards, fallback_count


def format_cards(cards):
    """Format cards as neat output."""
    lines = []

    lines.append("=" * 80)
    lines.append("VOTER CARDS - PAGE 3 (ONNX OCR + OpenCV Card Detection)")
    lines.append("=" * 80)
    lines.append("")

    for i, card in enumerate(cards, 1):
        lines.append("━" * 80)
        lines.append(f"CARD #{i:02d}")
        lines.append("━" * 80)
        lines.append(f"  Voter ID         : {card['voter_id']}")
        if card['serial_no']:
            lines.append(f"  Serial No        : {card['serial_no']}")
        lines.append(f"  Name             : {card['name']}")

        if card['relation_type']:
            lines.append(f"  {card['relation_type']}'s Name    : {card['relation_name']}")

        lines.append(f"  House Number     : {card['house_number'] or 'N/A'}")
        lines.append(f"  Age              : {card['age']}")
        lines.append(f"  Gender           : {card['gender']}")
        lines.append(f"  Photo Available  : Yes")
        lines.append("")

    lines.append("=" * 80)
    lines.append(f"TOTAL CARDS: {len(cards)}")
    lines.append("=" * 80)

    return '\n'.join(lines)


async def main():
    """Main function."""
    pdf_path = "2024-FC-EROLLGEN-S13-206-FinalRoll-Revision2-ENG-10-WI.pdf"
    page_num = 3

    print("=" * 80)
    print("EXTRACT VOTER CARDS - ONNX OCR + OPENCV CARD DETECTION (WITH TIMING)")
    print("=" * 80)
    print(f"\nPDF: {pdf_path}")
    print(f"Page: {page_num}\n")

    # Track timing
    timing = {}
    total_start = time.time()

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix=f'page{page_num}_hybrid_')

    try:
        # Step 1: Convert PDF to image
        print("STEP 1: Convert PDF to image (300 DPI)")
        print("-" * 80)

        stage_start = time.time()
        pil_images = convert_from_path(
            pdf_path,
            dpi=300,
            first_page=page_num,
            last_page=page_num
        )

        image_path = f"{temp_dir}/page_{page_num:04d}.png"
        pil_images[0].save(image_path, 'PNG')
        stage_end = time.time()
        timing['pdf_to_image'] = stage_end - stage_start

        print(f"✓ Saved to: {image_path}")
        print(f"⏱️  Time: {timing['pdf_to_image']:.3f} seconds\n")

        # Step 2: Detect card boundaries using OpenCV
        print("STEP 2: Detect card boundaries using OpenCV")
        print("-" * 80)

        stage_start = time.time()
        img = cv2.imread(image_path)
        card_boxes_raw = extract_cards_by_filtered_lines(img, method='approach1')

        # Flatten if nested (extract_cards_by_filtered_lines returns list of lists)
        card_boxes = []
        for item in card_boxes_raw:
            if isinstance(item, list):
                card_boxes.extend(item)
            else:
                card_boxes.append(item)

        stage_end = time.time()
        timing['card_detection'] = stage_end - stage_start

        print(f"✓ Detected {len(card_boxes)} card boundaries")
        print(f"⏱️  Time: {timing['card_detection']:.3f} seconds\n")

        # Step 3: Call ONNX OCR
        print("STEP 3: Extract text using ONNX OCR")
        print("-" * 80)

        stage_start = time.time()
        onnx_result = await call_onnx_ocr(image_path)
        text_blocks = get_text_blocks_from_onnx(onnx_result)
        stage_end = time.time()
        timing['onnx_ocr'] = stage_end - stage_start

        print(f"✓ Extracted {len(text_blocks)} text blocks")
        print(f"⏱️  Time: {timing['onnx_ocr']:.3f} seconds\n")

        # Step 4: Match text to cards
        print("STEP 4: Match text blocks to card regions and parse fields")
        print("-" * 80)

        stage_start = time.time()
        cards = []
        for i, card_box in enumerate(card_boxes, 1):
            card_texts = get_text_in_card(card_box, text_blocks)

            # Parse fields
            card_data = parse_card_fields(card_texts)
            cards.append(card_data)

        stage_end = time.time()
        timing['text_matching_and_parsing'] = stage_end - stage_start

        print(f"✓ Parsed {len(cards)} cards")
        print(f"⏱️  Time: {timing['text_matching_and_parsing']:.3f} seconds\n")

        # Step 5: Format and save output
        print("STEP 5: Format and save output")
        print("-" * 80)

        stage_start = time.time()
        formatted = format_cards(cards)

        # Save outputs
        output_txt = "page_0003_voter_cards_onnx_opencv.txt"
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(formatted)

        output_json = "page_0003_voter_cards_onnx_opencv.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump({
                'page': page_num,
                'total_cards': len(cards),
                'cards': cards
            }, f, indent=2, ensure_ascii=False)

        stage_end = time.time()
        timing['format_and_save'] = stage_end - stage_start

        print(f"✓ Saved to: {output_txt}")
        print(f"✓ Saved to: {output_json}")
        print(f"⏱️  Time: {timing['format_and_save']:.3f} seconds\n")

        # Calculate total time
        total_end = time.time()
        timing['total'] = total_end - total_start

        # Display timing summary
        print("=" * 80)
        print("TIMING SUMMARY")
        print("=" * 80)
        print()
        print(f"{'Stage':<45} {'Time (seconds)':<15} {'Percentage':<10}")
        print("-" * 80)
        print(f"{'1. PDF to Image Conversion':<45} {timing['pdf_to_image']:>12.3f}s    {timing['pdf_to_image']/timing['total']*100:>6.1f}%")
        print(f"{'2. OpenCV Card Detection':<45} {timing['card_detection']:>12.3f}s    {timing['card_detection']/timing['total']*100:>6.1f}%")
        print(f"{'3. ONNX OCR API Call':<45} {timing['onnx_ocr']:>12.3f}s    {timing['onnx_ocr']/timing['total']*100:>6.1f}%")
        print(f"{'4. Text Matching & Field Parsing':<45} {timing['text_matching_and_parsing']:>12.3f}s    {timing['text_matching_and_parsing']/timing['total']*100:>6.1f}%")
        print(f"{'5. Format & Save Output':<45} {timing['format_and_save']:>12.3f}s    {timing['format_and_save']/timing['total']*100:>6.1f}%")
        print("-" * 80)
        print(f"{'TOTAL TIME':<45} {timing['total']:>12.3f}s    {100:>6.1f}%")
        print("=" * 80)
        print()

        # Save timing data
        timing_json = f"page_{page_num:04d}_onnx_opencv_timing.json"
        with open(timing_json, 'w') as f:
            json.dump({
                'page': page_num,
                'total_cards': len(cards),
                'timing': timing,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"✓ Timing data saved to: {timing_json}\n")

        # Display formatted cards
        print(formatted)

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Extracted {len(cards)} voter cards")
    print(f"✓ Total time: {timing['total']:.3f} seconds")
    print(f"✓ ONNX OCR API time: {timing['onnx_ocr']:.3f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
