# Voter Card Extraction Pipeline

Memory-efficient pipeline for extracting voter registration data from PDF files stored in S3.

## Features

- ✅ **Smart S3 Streaming** - Processes ZIPs without downloading entire files
- ✅ **Memory Efficient** - Only keeps active PDF in memory, immediate cleanup
- ✅ **Checkpoint/Resume** - Automatically resumes from failures
- ✅ **Parallel Processing** - Can be extended to process multiple PDFs concurrently
- ✅ **Organized Output** - CSVs organized by district in S3
- ✅ **Progress Tracking** - Detailed logging and statistics

## Architecture

```
S3 (voter-pdf-dump)
  ├── S1301_Nandurbar.zip (~5GB, 400 PDFs)
  ├── S1302_Dhule.zip
  └── ... (36 ZIPs total)
       ↓
   [Pipeline]
   - Stream ZIP from S3
   - Extract PDF one-at-a-time
   - Detect cards (OpenCV)
   - Extract text (ONNX OCR)
   - Convert to CSV
   - Upload to S3
   - Delete local files
       ↓
S3 (voter-pdf-output)
  ├── S1301_Nandurbar/
  │   ├── pdf_001.csv
  │   ├── pdf_002.csv
  │   └── ...
  └── S1302_Dhule/
      └── ...
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

Create or update `.env` file:

```env
ACCESS_KEY_ID=your_access_key_id
SECRET_ACCESS_KEY=your_secret_access_key
```

### 3. Verify S3 Access

```bash
python test_pipeline.py list
```

This should list any existing files in `voter-pdf-output` bucket.

## Usage

### Local Testing (Recommended First)

Test with 1 ZIP and 2 PDFs:

```bash
python test_pipeline.py test
```

This will:
- Process the first ZIP file
- Extract only 2 PDFs (for quick testing)
- Upload CSVs to S3
- Create checkpoint file
- Show detailed logs

### Test Resume Functionality

Run the same command again to test checkpoint/resume:

```bash
python test_pipeline.py resume
```

Already processed PDFs will be skipped.

### Production Run (All ZIPs)

Edit `voter_extraction_pipeline.py` and modify the `main()` function:

```python
async def main():
    pipeline = VoterExtractionPipeline()
    # Remove limits for production
    await pipeline.run()  # No limit_zips or limit_pdfs
```

Then run:

```bash
python voter_extraction_pipeline.py
```

### Monitor Progress

```bash
# Watch logs in real-time
tail -f voter_extraction.log

# Check checkpoint status
cat extraction_checkpoint.json

# List S3 outputs
python test_pipeline.py list
```

## File Structure

```
voter_id/
├── .env                                    # AWS credentials
├── requirements.txt                        # Dependencies
├── README.md                              # This file
├── extraction_checkpoint.json             # Progress tracking
├── voter_extraction.log                   # Detailed logs
│
├── csv_converter.py                       # JSON to CSV converter
├── voter_extraction_pipeline.py           # Main pipeline
├── test_pipeline.py                       # Testing utilities
│
├── extract_voter_boxes_opencv_improved.py # Card detection
└── extract_cards_onnx_with_detection.py   # ONNX OCR integration
```

## Output Format

### CSV Structure

Each PDF generates a CSV with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `serial_no` | Serial number on card | `1` |
| `voter_id` | Voter ID number | `ABC1234567` |
| `name` | Voter name | `John Doe` |
| `relation_type` | Father/Husband/Mother | `Father` |
| `relation_name` | Relation's name | `James Doe` |
| `house_number` | House number | `123` |
| `age` | Age | `35` |
| `gender` | Male/Female | `Male` |

### S3 Output Structure

```
voter-pdf-output/
├── S1301_Nandurbar/
│   ├── 2024-FC-EROLLGEN-page-001.csv
│   ├── 2024-FC-EROLLGEN-page-002.csv
│   └── ... (~400 CSVs)
├── S1302_Dhule/
│   └── ...
└── ... (36 folders)
```

## Memory Management

The pipeline is designed to be memory-efficient:

- **ZIP files**: Streamed from S3 (not downloaded)
- **PDFs**: Extracted one-at-a-time, deleted after processing
- **CSVs**: Uploaded to S3 then deleted
- **Peak memory**: ~100-200 MB per worker
- **Disk usage**: <1 GB at any time

Perfect for `m4.xlarge` EC2 instance (16 GB RAM).

## Checkpoint & Resume

The pipeline automatically saves progress in `extraction_checkpoint.json`:

```json
{
  "completed_pdfs": [
    "S1301_Nandurbar/pdf_001.pdf",
    "S1301_Nandurbar/pdf_002.pdf"
  ],
  "completed_zips": [],
  "current_zip": "S1301_Nandurbar.zip",
  "last_updated": "2026-02-17T10:30:00"
}
```

If the pipeline crashes or is interrupted:
1. Simply run it again
2. It will skip already processed PDFs
3. Resume from where it left off

## Error Handling

- Failed PDFs are logged but don't stop the pipeline
- Checkpoint is saved after each successful PDF
- Detailed error logs in `voter_extraction.log`
- Statistics show total errors at the end

## Transferring to EC2

### 1. Create EC2 Instance

- **Type**: `m4.xlarge` (4 vCPUs, 16 GB RAM)
- **OS**: Ubuntu 22.04 LTS
- **Storage**: 20 GB (sufficient for streaming approach)

### 2. Setup EC2

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3-pip -y

# Install system dependencies for OpenCV and PDF processing
sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils

# Create project directory
mkdir ~/voter_extraction
cd ~/voter_extraction
```

### 3. Transfer Code

```bash
# From your local machine
scp -i your-key.pem -r /Users/nikithmajeti/Desktop/voter_id/* ubuntu@your-ec2-ip:~/voter_extraction/
```

### 4. Install Dependencies on EC2

```bash
# On EC2
cd ~/voter_extraction
pip3 install -r requirements.txt
```

### 5. Create .env File on EC2

```bash
# On EC2
cat > .env << EOF
ACCESS_KEY_ID=your_access_key_id
SECRET_ACCESS_KEY=your_secret_access_key
EOF
```

### 6. Test on EC2

```bash
# Quick test
python3 test_pipeline.py test

# Check logs
tail -f voter_extraction.log
```

### 7. Run Production (Background)

```bash
# Run in background with nohup
nohup python3 voter_extraction_pipeline.py > output.log 2>&1 &

# Monitor progress
tail -f voter_extraction.log
tail -f output.log

# Check if running
ps aux | grep voter_extraction_pipeline
```

## Performance Estimates

Based on m4.xlarge EC2:

- **Per PDF**: ~30-60 seconds (depends on pages)
- **Per ZIP**: ~4-8 hours (400 PDFs)
- **All 36 ZIPs**: ~144-288 hours (6-12 days)

To speed up:
- Use larger instance (more RAM for parallel processing)
- Implement multiprocessing (process multiple ZIPs simultaneously)

## Troubleshooting

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### AWS Credentials Issues

Verify `.env` file exists and has correct credentials:
```bash
cat .env
python test_pipeline.py list
```

### Memory Issues

If running out of memory:
- Check that files are being deleted (logs should show cleanup)
- Reduce concurrent processing
- Use larger EC2 instance

### ONNX API Issues

If OCR API fails:
- Check network connectivity
- Verify API key is valid
- Check API rate limits

## License

Internal use only.
