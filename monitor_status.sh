#!/bin/bash
# Monitor voter extraction pipeline status

echo "=================================="
echo "VOTER EXTRACTION PIPELINE STATUS"
echo "=================================="
echo ""

# Check if process is running
if pm2 list | grep -q "voter-extraction-pipeline.*online"; then
    echo "✓ Status: RUNNING"
else
    echo "✗ Status: STOPPED"
fi

echo ""
echo "=== LATEST PROGRESS ==="
tail -20 voter_extraction_status.log | grep -E "PROGRESS UPDATE|Total cards|Completed ZIP|errors" | tail -10

echo ""
echo "=== CHECKPOINT STATUS ==="
if [ -f "extraction_checkpoint.json" ]; then
    python3 -c "
import json
with open('extraction_checkpoint.json') as f:
    cp = json.load(f)
    print(f'  PDFs completed: {len(cp.get(\"completed_pdfs\", []))}')
    print(f'  ZIPs completed: {len(cp.get(\"completed_zips\", []))}')
    print(f'  Current ZIP: {cp.get(\"current_zip\", \"None\")}')
    print(f'  Last updated: {cp.get(\"last_updated\", \"Never\")}')
"
else
    echo "  No checkpoint found"
fi

echo ""
echo "=== RECENT ERRORS ==="
tail -100 voter_extraction_detailed.log | grep -i "error" | tail -5 || echo "  No recent errors"

echo ""
echo "=== MEMORY USAGE ==="
pm2 show voter-extraction-pipeline 2>/dev/null | grep -E "memory|cpu" || echo "  Process not running"

echo ""
echo "=================================="
echo "Commands:"
echo "  pm2 logs voter-extraction-pipeline  # View live logs"
echo "  pm2 restart voter-extraction-pipeline  # Restart"
echo "  pm2 stop voter-extraction-pipeline  # Stop"
echo "=================================="
