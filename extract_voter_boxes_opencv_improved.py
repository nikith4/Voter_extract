#!/usr/bin/env python3
"""
Improved OpenCV-based card extraction with filtered line detection.
Implements Approach 1: Filter lines by length to get only card boundaries.
"""

import cv2
import numpy as np
import sys
import os
import json
from datetime import datetime


def extract_cards_by_filtered_lines(img, method='approach1', min_line_length=100):
    """
    Extract card bounding boxes using filtered line detection.
    
    Args:
        img: Input image (BGR format)
        method: 'approach1' (70% width filter) or 'user_suggestion' (1/3 width filter)
        min_line_length: Minimum line length to consider
        
    Returns:
        tuple: (card_boxes, stats_dict) where stats contains line counts and filtering info
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=50
    )
    
    if lines is None:
        return [], {'total_lines': 0, 'h_lines_before': 0, 'v_lines_before': 0}
    
    # Separate horizontal and vertical lines (before filtering)
    h_lines_raw = []
    v_lines_raw = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length_x = abs(x2 - x1)
        line_length_y = abs(y2 - y1)
        
        # Check if horizontal (small y difference)
        if abs(y2 - y1) < 10:
            h_lines_raw.append({
                'y': (y1 + y2) // 2,
                'x1': min(x1, x2),
                'x2': max(x1, x2),
                'length': line_length_x
            })
        
        # Check if vertical (small x difference)
        elif abs(x2 - x1) < 10:
            v_lines_raw.append({
                'x': (x1 + x2) // 2,
                'y1': min(y1, y2),
                'y2': max(y1, y2),
                'length': line_length_y
            })
    
    stats = {
        'total_lines': len(lines),
        'h_lines_before': len(h_lines_raw),
        'v_lines_before': len(v_lines_raw)
    }
    
    # ============================================================
    # APPROACH 1: Filter by length (70% width for horizontal, 50% height for vertical)
    # ============================================================
    if method == 'approach1':
        # Filter horizontal lines: must span at least 70% of page width
        h_lines_filtered = []
        for line in h_lines_raw:
            if line['length'] >= width * 0.7:
                h_lines_filtered.append(line['y'])
        
        # Filter vertical lines: must span at least 50% of page height
        v_lines_filtered = []
        for line in v_lines_raw:
            if line['length'] >= height * 0.5:
                v_lines_filtered.append(line['x'])
        
        stats['h_lines_after_filter'] = len(h_lines_filtered)
        stats['v_lines_after_filter'] = len(v_lines_filtered)
        stats['filter_criteria'] = 'Horizontal: >=70% width, Vertical: >=50% height'
    
    # ============================================================
    # USER SUGGESTION: Filter by length (1/3 width for horizontal, 1/10 height for vertical)
    # ============================================================
    elif method == 'user_suggestion':
        # Filter horizontal lines: must span at least 1/3 of page width
        h_lines_filtered = []
        for line in h_lines_raw:
            if line['length'] >= width / 3:
                h_lines_filtered.append(line['y'])
        
        # Filter vertical lines: must span at least 1/10 of page height
        v_lines_filtered = []
        for line in v_lines_raw:
            if line['length'] >= height / 10:
                v_lines_filtered.append(line['x'])
        
        stats['h_lines_after_filter'] = len(h_lines_filtered)
        stats['v_lines_after_filter'] = len(v_lines_filtered)
        stats['filter_criteria'] = 'Horizontal: >=1/3 width, Vertical: >=1/10 height'
    
    else:
        # No filtering (original method)
        h_lines_filtered = [line['y'] for line in h_lines_raw]
        v_lines_filtered = [line['x'] for line in v_lines_raw]
        stats['filter_criteria'] = 'No filtering'
    
    # Sort and remove duplicates
    h_lines_filtered = sorted(set(h_lines_filtered))
    v_lines_filtered = sorted(set(v_lines_filtered))
    
    # Group nearby lines (within 20 pixels)
    def group_lines(lines, threshold=20):
        if not lines:
            return []
        grouped = []
        current_group = [lines[0]]
        for line in lines[1:]:
            if line - current_group[-1] < threshold:
                current_group.append(line)
            else:
                grouped.append(int(np.mean(current_group)))
                current_group = [line]
        grouped.append(int(np.mean(current_group)))
        return grouped
    
    h_lines_before_group = len(h_lines_filtered)
    v_lines_before_group = len(v_lines_filtered)
    
    h_lines = group_lines(h_lines_filtered)
    v_lines = group_lines(v_lines_filtered)
    
    stats['h_lines_after_group'] = len(h_lines)
    stats['v_lines_after_group'] = len(v_lines)
    stats['h_lines_grouped_from'] = h_lines_before_group
    stats['v_lines_grouped_from'] = v_lines_before_group
    
    # Generate bounding boxes from grid
    card_boxes = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            x = v_lines[j]
            y = h_lines[i]
            w = v_lines[j + 1] - v_lines[j]
            h = h_lines[i + 1] - h_lines[i]
            
            # Filter out very small boxes
            if w > 50 and h > 50:
                card_boxes.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'row': i + 1,
                    'col': j + 1
                })
    
    # Filter to keep only the 3 main columns (cards are typically in 3 columns)
    # Find columns with reasonable width (cards should be ~300-400px wide)
    if len(v_lines) > 3:
        # Group boxes by column and calculate average width per column
        col_widths = {}
        for box in card_boxes:
            col = box['col']
            if col not in col_widths:
                col_widths[col] = []
            col_widths[col].append(box['width'])
        
        # Calculate average width per column
        avg_col_widths = {col: sum(widths) / len(widths) for col, widths in col_widths.items()}
        
        # Find the 3 columns with largest average width (these are likely the card columns)
        sorted_cols = sorted(avg_col_widths.items(), key=lambda x: x[1], reverse=True)
        top_3_cols = [col for col, _ in sorted_cols[:3]]
        top_3_cols.sort()  # Sort by column number
        
        # Filter boxes to keep only these 3 columns
        card_boxes = [box for box in card_boxes if box['col'] in top_3_cols]
        stats['columns_filtered'] = f"Kept columns: {top_3_cols} (from {len(col_widths)} total)"
    else:
        stats['columns_filtered'] = f"No filtering needed ({len(v_lines)-1} columns)"
    
    # Merge vertically adjacent boxes in the same column to form complete cards
    # This handles cases where "photo available" separator creates multiple boxes per card
    def merge_vertical_boxes(boxes, min_card_height=100, max_gap=30):
        """
        Merge small boxes (likely partial cards) with adjacent boxes in the same column.
        min_card_height: Minimum height for a complete card. Boxes smaller than this are merged.
        max_gap: Maximum gap between boxes to merge them.
        """
        # Group boxes by column
        by_column = {}
        for box in boxes:
            col = box['col']
            if col not in by_column:
                by_column[col] = []
            by_column[col].append(box)
        
        merged_boxes = []
        for col, col_boxes in by_column.items():
            # Sort by y-coordinate (top to bottom)
            col_boxes.sort(key=lambda b: b['y'])
            
            i = 0
            while i < len(col_boxes):
                current_box = col_boxes[i].copy()
                
                # Check if we can merge with next box
                if i + 1 < len(col_boxes):
                    next_box = col_boxes[i + 1]
                    
                    # Calculate gap
                    gap = next_box['y'] - (current_box['y'] + current_box['height'])
                    
                    # Merge if:
                    # 1. Gap is very small (likely internal separator like "photo available")
                    # 2. AND at least one box is smaller than expected (suggesting partial card)
                    # This prevents merging two complete cards that happen to touch
                    avg_height = (current_box['height'] + next_box['height']) / 2
                    is_partial_card = (current_box['height'] < min_card_height) or (next_box['height'] < min_card_height)
                    should_merge = (gap <= max_gap) and is_partial_card
                    
                    if should_merge:
                        # Merge: combine both boxes
                        merged_box = {
                            'x': min(current_box['x'], next_box['x']),
                            'y': current_box['y'],
                            'width': max(current_box['x'] + current_box['width'], 
                                       next_box['x'] + next_box['width']) - min(current_box['x'], next_box['x']),
                            'height': (next_box['y'] + next_box['height']) - current_box['y'],
                            'row': current_box['row'],
                            'col': col,
                            'merged_from': 2
                        }
                        merged_boxes.append(merged_box)
                        i += 2  # Skip both boxes
                        continue
                
                # Can't merge, keep as is
                merged_boxes.append(current_box)
                i += 1
        
        return merged_boxes
    
    boxes_before_merge = len(card_boxes)
    # Merge boxes with very small or zero gaps where at least one is a partial card
    # min_card_height: Boxes smaller than this are considered partial and should be merged
    # This catches "photo available" separators that create small partial boxes
    card_boxes = merge_vertical_boxes(card_boxes, min_card_height=80, max_gap=5)
    boxes_after_merge = len(card_boxes)
    stats['boxes_before_merge'] = boxes_before_merge
    stats['boxes_after_merge'] = boxes_after_merge
    stats['boxes_merged'] = boxes_before_merge - boxes_after_merge
    
    # Sort by position (row, then column)
    card_boxes.sort(key=lambda b: (b['y'], b['x']))
    
    # Limit to max 30 boxes
    # After merging, we should have close to 30 boxes (10 rows × 3 columns)
    if len(card_boxes) > 30:
        # Sort by position and take first 30
        card_boxes = card_boxes[:30]
        stats['boxes_limited'] = True
    else:
        stats['boxes_limited'] = False
    
    stats['final_box_count'] = len(card_boxes)
    
    return card_boxes, stats


def visualize_filtered_extraction(image_path, method='approach1', output_dir=None):
    """
    Extract cards and create visualization showing filtered lines and boxes.
    """
    print("=" * 80)
    print(f"EXTRACTING CARDS WITH FILTERED LINES - {method.upper()}")
    print("=" * 80)
    print(f"\n📸 Image: {image_path}\n")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image")
        return None, None
    
    height, width = img.shape[:2]
    print(f"Image dimensions: {width}x{height}\n")
    
    # Extract boxes
    card_boxes, stats = extract_cards_by_filtered_lines(img, method=method)
    
    # Print statistics
    print("FILTERING STATISTICS:")
    print("-" * 80)
    print(f"Total lines detected: {stats['total_lines']}")
    print(f"Horizontal lines (before filter): {stats['h_lines_before']}")
    print(f"Vertical lines (before filter): {stats['v_lines_before']}")
    print(f"\nFilter criteria: {stats['filter_criteria']}")
    print(f"Horizontal lines (after filter): {stats['h_lines_after_filter']}")
    print(f"Vertical lines (after filter): {stats['v_lines_after_filter']}")
    print(f"\nAfter grouping:")
    print(f"  Horizontal: {stats['h_lines_after_group']} (from {stats['h_lines_grouped_from']})")
    print(f"  Vertical: {stats['v_lines_after_group']} (from {stats['v_lines_grouped_from']})")
    print(f"\nFinal boxes: {stats['final_box_count']}")
    if stats['boxes_limited']:
        print("  (Limited to 30 boxes)")
    print("-" * 80)
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create visualization
    vis_img = img.copy()
    
    # Draw boxes
    for i, box in enumerate(card_boxes):
        x = box['x']
        y = box['y']
        w = box['width']
        h = box['height']
        
        # Draw rectangle
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label
        label = f"R{box['row']}C{box['col']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        label_thickness = 1
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, label_thickness)
        
        # Draw background for text
        cv2.rectangle(vis_img, (x, y - text_height - 5), 
                     (x + text_width + 5, y), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(vis_img, label, (x + 2, y - 5), 
                   font, font_scale, (0, 0, 0), label_thickness)
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{base_name}_filtered_{method}_boxes.png")
    cv2.imwrite(output_path, vis_img)
    print(f"\n✓ Visualization saved to: {output_path}\n")
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{base_name}_filtered_{method}_boxes.json")
    output_data = {
        'source_image': image_path,
        'method': method,
        'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'statistics': stats,
        'total_boxes': len(card_boxes),
        'boxes': card_boxes
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON saved to: {json_path}\n")
    
    return card_boxes, stats


def main():
    """Main function - test both approaches."""
    if len(sys.argv) < 2:
        image_path = "/Users/muskan/Documents/ZOOP/zoop-ml-challan-ocr/voter/Pimpri/2024-FC-EROLLGEN-S13-206-FinalRoll-Revision2-ENG-3-WI/page_0007.png"
    else:
        image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print("\n" + "="*80)
    print("TESTING APPROACH 1 (70% width filter)")
    print("="*80 + "\n")
    boxes1, stats1 = visualize_filtered_extraction(image_path, method='approach1')
    
    print("\n" + "="*80)
    print("TESTING USER SUGGESTION (1/3 width, 1/10 height filter)")
    print("="*80 + "\n")
    boxes2, stats2 = visualize_filtered_extraction(image_path, method='user_suggestion')
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Approach 1: {stats1['final_box_count']} boxes")
    print(f"User Suggestion: {stats2['final_box_count']} boxes")
    print("="*80)


if __name__ == "__main__":
    main()
