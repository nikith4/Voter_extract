#!/usr/bin/env python3
"""
Voter Card Extraction Pipeline - Production Version
Streams ZIP files from S3, extracts voter data, converts to CSV, and uploads to S3.
Memory-efficient with checkpoint/resume capability.
"""

import os
import sys
import zipfile
import tempfile
import shutil
import json
import asyncio
import logging
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import traceback

import boto3
from smart_open import open as s_open
from tqdm import tqdm
from dotenv import load_dotenv

# Import existing extraction modules
from extract_voter_boxes_opencv_improved import extract_cards_by_filtered_lines
from extract_cards_onnx_with_detection import (
    call_onnx_ocr,
    get_text_blocks_from_onnx,
    extract_cards_with_fallback
)
from csv_converter import voter_cards_to_csv

# Load environment variables
load_dotenv()

# Configure logging with separate files for different purposes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Main logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Detailed log (everything)
detailed_handler = logging.FileHandler('voter_extraction_detailed.log')
detailed_handler.setLevel(logging.DEBUG)
detailed_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Status log (important events only)
status_handler = logging.FileHandler('voter_extraction_status.log')
status_handler.setLevel(logging.INFO)
status_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# Console output (clean, essential info)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

logger.addHandler(detailed_handler)
logger.addHandler(status_handler)
logger.addHandler(console_handler)


class VoterExtractionPipeline:
    """
    Main pipeline for extracting voter data from S3 ZIP files.
    """

    def __init__(
        self,
        input_bucket: str = 'voter-pdf-dump',
        output_bucket: str = 'voter-pdf-output',
        checkpoint_file: str = 'extraction_checkpoint.json',
        max_concurrent_pdfs: int = 12
    ):
        """
        Initialize the extraction pipeline.

        Args:
            input_bucket: S3 bucket containing ZIP files
            output_bucket: S3 bucket for CSV outputs
            checkpoint_file: Local file to track progress
            max_concurrent_pdfs: Maximum PDFs to process in parallel (default: 6)
        """
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.checkpoint_file = checkpoint_file
        self.max_concurrent_pdfs = max_concurrent_pdfs
        self.shutdown_requested = False

        # AWS credentials from .env
        self.aws_access_key = os.getenv('ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('SECRET_ACCESS_KEY')

        if not self.aws_access_key or not self.aws_secret_key:
            raise ValueError("AWS credentials not found in .env file")

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
        )

        # Load checkpoint
        self.checkpoint = self._load_checkpoint()

        # Lock for thread-safe checkpoint updates (prevents race conditions)
        self.checkpoint_lock = asyncio.Lock()

        # Semaphores for controlling concurrency
        self.pdf_semaphore = asyncio.Semaphore(max_concurrent_pdfs)
        # Note: page_semaphore is created per-PDF in process_single_pdf()

        # Statistics
        self.stats = {
            'total_zips_processed': 0,
            'total_pdfs_processed': 0,
            'total_cards_extracted': 0,
            'total_errors': 0,
            'start_time': None,
            'end_time': None
        }

    def _load_checkpoint(self) -> Dict:
        """
        Load checkpoint from S3 (primary) or local file (fallback).

        Priority:
        1. Try loading from S3 (persistent across EC2 restarts)
        2. Fallback to local file (for backwards compatibility)
        3. Return empty checkpoint if neither exists
        """
        s3_key = 'checkpoints/extraction_checkpoint.json'

        # Try loading from S3 first (primary source)
        try:
            logger.info(f"Loading checkpoint from S3: s3://{self.output_bucket}/{s3_key}")
            response = self.s3_client.get_object(Bucket=self.output_bucket, Key=s3_key)
            checkpoint = json.loads(response['Body'].read().decode('utf-8'))

            # Also save to local file for quick access
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            logger.info(f"✓ Loaded checkpoint from S3: {len(checkpoint.get('completed_pdfs', []))} PDFs, "
                       f"{len(checkpoint.get('completed_zips', []))} ZIPs already processed")
            return checkpoint

        except self.s3_client.exceptions.NoSuchKey:
            logger.info("No checkpoint found in S3, checking local file...")
        except Exception as e:
            logger.warning(f"Could not load checkpoint from S3: {e}, trying local file...")

        # Fallback to local file
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    logger.info(f"✓ Loaded checkpoint from local file: {len(checkpoint.get('completed_pdfs', []))} PDFs already processed")
                    return checkpoint
            except Exception as e:
                logger.warning(f"Could not load checkpoint from local file: {e}")

        # Return empty checkpoint
        logger.info("No checkpoint found, starting fresh")
        return {
            'completed_pdfs': [],
            'completed_zips': [],
            'current_zip': None,
            'last_updated': None
        }

    async def _save_checkpoint(self):
        """
        Save current progress to S3 (primary) and local file (backup).

        Thread-safe with asyncio.Lock to prevent race conditions when multiple
        PDFs complete simultaneously.

        Saves to both locations to ensure checkpoint survives EC2 crashes/terminations.
        """
        async with self.checkpoint_lock:
            self.checkpoint['last_updated'] = datetime.now().isoformat()
            s3_key = 'checkpoints/extraction_checkpoint.json'

            # Save to local file first (atomic write using temp file)
            try:
                temp_file = f"{self.checkpoint_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(self.checkpoint, f, indent=2)

                # Atomic rename (prevents corruption if interrupted)
                os.replace(temp_file, self.checkpoint_file)

            except Exception as e:
                logger.error(f"Failed to save checkpoint to local file: {e}")

            # Save to S3 (primary persistent storage)
            try:
                checkpoint_json = json.dumps(self.checkpoint, indent=2)
                self.s3_client.put_object(
                    Bucket=self.output_bucket,
                    Key=s3_key,
                    Body=checkpoint_json.encode('utf-8'),
                    ContentType='application/json'
                )
                logger.debug(f"✓ Checkpoint saved to S3 and local: {len(self.checkpoint['completed_pdfs'])} PDFs, "
                            f"{len(self.checkpoint['completed_zips'])} ZIPs completed")

            except Exception as e:
                logger.error(f"Failed to save checkpoint to S3: {e}")
                logger.warning("⚠️  Checkpoint only saved locally - may be lost if EC2 terminates!")

    def list_zip_files(self) -> List[str]:
        """
        List all ZIP files in the input S3 bucket.

        Returns:
            List of ZIP file keys
        """
        logger.info(f"Listing ZIP files in s3://{self.input_bucket}/")

        zip_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.input_bucket):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.zip'):
                    zip_files.append(key)

        logger.info(f"Found {len(zip_files)} ZIP files")
        return sorted(zip_files)

    def get_zip_folder_name(self, zip_key: str) -> str:
        """
        Extract folder name from ZIP key.

        Args:
            zip_key: S3 key of ZIP file (e.g., 'S1301_Nandurbar.zip')

        Returns:
            Folder name (e.g., 'S1301_Nandurbar')
        """
        return Path(zip_key).stem

    async def process_pdf_page(self, image_path: str) -> tuple:
        """
        Process a single PDF page and extract voter cards with fallback OCR.

        Args:
            image_path: Path to the page image (PNG)

        Returns:
            Tuple of (cards_list, fallback_count)
        """
        try:
            import cv2

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return [], 0

            # Step 1: Detect card boundaries using OpenCV
            card_boxes, stats = extract_cards_by_filtered_lines(img, method='approach1')

            if not card_boxes:
                logger.warning(f"No card boxes detected in {image_path}")
                return [], 0

            # Step 2: Extract text using ONNX OCR
            onnx_result = await call_onnx_ocr(image_path)
            text_blocks = get_text_blocks_from_onnx(onnx_result)

            # Step 3: Match text to cards, parse fields, and apply fallback OCR
            # This automatically crops and re-OCRs cards with missing relation_name
            cards, fallback_count = await extract_cards_with_fallback(
                image_path, card_boxes, text_blocks
            )

            return cards, fallback_count

        except Exception as e:
            logger.error(f"Error processing PDF page {image_path}: {e}")
            logger.debug(traceback.format_exc())
            return [], 0

    async def _process_page_from_pdf_with_semaphore(
        self,
        pdf_path: str,
        page_num: int,
        total_pages: int,
        temp_dir: str,
        pdf_stem: str,
        page_semaphore: asyncio.Semaphore,
        pdf_index: int
    ) -> tuple:
        """
        Convert and process a single page from PDF (memory efficient).

        Converts ONLY the requested page instead of entire PDF to save memory.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            total_pages: Total pages in PDF
            temp_dir: Temporary directory
            pdf_stem: PDF file stem for naming
            page_semaphore: Semaphore for this specific PDF (NOT shared)
            pdf_index: PDF index for logging (1-based)

        Returns:
            Tuple of (page_num, cards_list, fallback_count)
        """
        logger.debug(f"  📄 [PDF {pdf_index}] Page {page_num}/{total_pages}: Waiting for page semaphore...")
        async with page_semaphore:
            logger.debug(f"  ✓ [PDF {pdf_index}] Page {page_num}/{total_pages}: Acquired page semaphore, converting page...")
            try:
                # Convert ONLY this page (not entire PDF!) - saves massive memory
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path, dpi=300, first_page=page_num, last_page=page_num)
                img = images[0]  # Only one page returned

                # Save page as image
                image_path = os.path.join(temp_dir, f"{pdf_stem}_page_{page_num:04d}.png")
                img.save(image_path, 'PNG')

                # Extract cards from page
                page_cards, fallback_count = await self.process_pdf_page(image_path)

                # Delete page image immediately
                os.remove(image_path)

                fallback_str = f" (fallback: {fallback_count})" if fallback_count > 0 else ""
                logger.info(f"  ✅ [PDF {pdf_index}] Page {page_num}/{total_pages}: {len(page_cards)} cards extracted{fallback_str}")

                return (page_num, page_cards, fallback_count)

            except Exception as e:
                logger.error(f"  ❌ [PDF {pdf_index}] Page {page_num}/{total_pages}: Error processing page: {e}")
                logger.debug(traceback.format_exc())
                return (page_num, [], 0)

    async def _process_page_with_semaphore(
        self,
        img,
        page_num: int,
        total_pages: int,
        temp_dir: str,
        pdf_stem: str,
        page_semaphore: asyncio.Semaphore,
        pdf_index: int
    ) -> tuple:
        """
        Process a single page with semaphore control for parallel page processing.
        DEPRECATED: Use _process_page_from_pdf_with_semaphore for memory efficiency.

        Args:
            img: PIL Image object
            page_num: Page number (1-indexed)
            total_pages: Total pages in PDF
            temp_dir: Temporary directory
            pdf_stem: PDF file stem for naming
            page_semaphore: Semaphore for this specific PDF (NOT shared)
            pdf_index: PDF index for logging (1-based)

        Returns:
            Tuple of (page_num, cards_list, fallback_count)
        """
        logger.debug(f"  📄 [PDF {pdf_index}] Page {page_num}/{total_pages}: Waiting for page semaphore...")
        async with page_semaphore:
            logger.debug(f"  ✓ [PDF {pdf_index}] Page {page_num}/{total_pages}: Acquired page semaphore, starting processing")
            try:
                # Save page as image
                image_path = os.path.join(temp_dir, f"{pdf_stem}_page_{page_num:04d}.png")
                img.save(image_path, 'PNG')

                # Extract cards from page
                page_cards, fallback_count = await self.process_pdf_page(image_path)

                # Delete page image immediately
                os.remove(image_path)

                fallback_str = f" (fallback: {fallback_count})" if fallback_count > 0 else ""
                logger.info(f"  ✅ [PDF {pdf_index}] Page {page_num}/{total_pages}: {len(page_cards)} cards extracted{fallback_str}")

                return page_num, page_cards, fallback_count

            except Exception as e:
                logger.error(f"❌ [PDF {pdf_index}] Error processing page {page_num}: {e}")
                return page_num, [], 0
            finally:
                logger.debug(f"  🔓 [PDF {pdf_index}] Page {page_num}/{total_pages}: Released page semaphore")

    async def process_single_pdf(
        self,
        pdf_data: bytes,
        pdf_name: str,
        zip_folder: str,
        temp_dir: str,
        pdf_index: int = 0
    ) -> Optional[str]:
        """
        Process a single PDF file and upload CSV to S3.

        Args:
            pdf_data: PDF file content (bytes)
            pdf_name: Name of the PDF file (may include path in ZIP)
            zip_folder: Folder name in output bucket
            temp_dir: Temporary directory for processing

        Returns:
            Path to created CSV file (None if failed)
        """
        # Extract just the filename (pdf_name might include folder path from ZIP)
        pdf_basename = os.path.basename(pdf_name)
        pdf_id = f"{zip_folder}/{pdf_basename}"

        # Check if already processed
        if pdf_id in self.checkpoint['completed_pdfs']:
            logger.info(f"Skipping already processed: {pdf_id}")
            return None

        logger.info(f"Processing: {pdf_id}")

        # Initialize paths for cleanup
        pdf_path = None
        csv_path = None

        try:
            # Save PDF to temp file (use basename to avoid nested folders)
            pdf_path = os.path.join(temp_dir, pdf_basename)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_data)

            # Get page count WITHOUT converting all pages (memory efficient)
            from pdf2image import pdfinfo_from_path
            from pdf2image import convert_from_path

            pdf_info = pdfinfo_from_path(pdf_path)
            total_pages = pdf_info['Pages']
            logger.info(f"  - PDF has {total_pages} page(s)")

            # IMPORTANT: Cards only appear on pages 3 to (last_page - 2)
            # Pages 1-2: Cover, index, headers
            # Last 2 pages: Summary, certification, footers
            start_page = 3  # Page 3 (index 2)
            end_page = max(3, total_pages - 2)  # Last page - 2

            if total_pages < 3:
                logger.warning(f"  - PDF has only {total_pages} pages, skipping (need at least 3)")
                os.remove(pdf_path)
                return None

            logger.info(f"  - Processing pages {start_page} to {end_page} (skipping first 2 and last 2)")

            # Process pages one at a time (memory efficient - no bulk conversion!)
            pdf_stem = Path(pdf_basename).stem
            page_semaphore = asyncio.Semaphore(3)  # 3 pages at a time for THIS PDF
            logger.debug(f"  🔧 [PDF {pdf_index}] Created page semaphore with limit=3 (per-PDF, not shared)")

            tasks = []
            for page_num in range(start_page, end_page + 1):
                # Convert ONLY this page (not entire PDF!) - huge memory savings
                task = self._process_page_from_pdf_with_semaphore(
                    pdf_path, page_num, total_pages, temp_dir, pdf_stem, page_semaphore, pdf_index
                )
                tasks.append(task)

            logger.debug(f"  🚀 [PDF {pdf_index}] Created {len(tasks)} page tasks, starting parallel execution...")

            # Execute all page tasks in parallel (limited by page semaphore)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect all cards and aggregate fallback counts
            all_cards = []
            total_fallback = 0
            for result in results:
                if isinstance(result, tuple):
                    page_num, page_cards, fallback_count = result
                    all_cards.extend(page_cards)
                    total_fallback += fallback_count
                elif isinstance(result, Exception):
                    logger.error(f"Page processing failed: {result}")

            # Delete PDF file
            os.remove(pdf_path)

            # Convert to CSV (use basename)
            csv_name = f"{Path(pdf_basename).stem}.csv"
            csv_path = os.path.join(temp_dir, csv_name)
            voter_cards_to_csv(all_cards, csv_path)

            fallback_info = f", fallback recovered: {total_fallback}" if total_fallback > 0 else ""
            logger.info(f"  - Total cards: {len(all_cards)}, CSV created{fallback_info}")

            # Upload CSV to S3
            s3_key = f"{zip_folder}/{csv_name}"
            self.s3_client.upload_file(csv_path, self.output_bucket, s3_key)

            logger.info(f"  ✓ Uploaded to s3://{self.output_bucket}/{s3_key}")

            # Delete CSV file
            os.remove(csv_path)

            # Update checkpoint (thread-safe with lock)
            async with self.checkpoint_lock:
                self.checkpoint['completed_pdfs'].append(pdf_id)
            await self._save_checkpoint()

            # Update stats
            self.stats['total_pdfs_processed'] += 1
            self.stats['total_cards_extracted'] += len(all_cards)

            # Log progress summary every 10 PDFs
            if self.stats['total_pdfs_processed'] % 10 == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"PROGRESS UPDATE - {self.stats['total_pdfs_processed']} PDFs processed")
                logger.info(f"Total cards extracted: {self.stats['total_cards_extracted']:,}")
                logger.info(f"Total errors: {self.stats['total_errors']}")
                logger.info(f"{'='*80}\n")

            return csv_path

        except Exception as e:
            logger.error(f"Error processing {pdf_id}: {e}")
            logger.debug(traceback.format_exc())
            self.stats['total_errors'] += 1

            # Cleanup on error
            for f in [pdf_path, csv_path]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass

            return None

    async def _process_single_pdf_from_disk_with_semaphore(
        self,
        pdf_file_path: str,
        pdf_name: str,
        zip_folder: str,
        temp_dir: str,
        index: int,
        total: int
    ) -> Optional[str]:
        """
        Wrapper that reads PDF from disk and processes it (memory efficient).

        Reads PDF only when semaphore is acquired, avoiding memory buildup.

        Args:
            pdf_file_path: Path to PDF file on disk
            pdf_name: Name of PDF file
            zip_folder: Folder name in output bucket
            temp_dir: Temporary directory
            index: Current PDF index (1-based)
            total: Total number of PDFs

        Returns:
            Path to created CSV file (None if failed)
        """
        logger.debug(f"📋 [PDF {index}/{total}] Waiting for PDF semaphore...")
        async with self.pdf_semaphore:
            logger.info(f"🔓 [PDF {index}/{total}] Acquired PDF semaphore, starting: {pdf_name}")
            try:
                # Read PDF from disk ONLY when ready to process (memory efficient!)
                with open(pdf_file_path, 'rb') as f:
                    pdf_data = f.read()

                return await self.process_single_pdf(pdf_data, pdf_name, zip_folder, temp_dir, index)
            finally:
                logger.debug(f"🔒 [PDF {index}/{total}] Released PDF semaphore")

    async def _process_single_pdf_with_semaphore(
        self,
        pdf_data: bytes,
        pdf_name: str,
        zip_folder: str,
        temp_dir: str,
        index: int,
        total: int
    ) -> Optional[str]:
        """
        Wrapper for process_single_pdf that uses semaphore for concurrency control.

        Args:
            pdf_data: PDF file content
            pdf_name: Name of PDF file
            zip_folder: Folder name in output bucket
            temp_dir: Temporary directory
            index: Current PDF index (1-based)
            total: Total number of PDFs

        Returns:
            Path to created CSV file (None if failed)
        """
        logger.debug(f"📋 [PDF {index}/{total}] Waiting for PDF semaphore...")
        async with self.pdf_semaphore:
            logger.info(f"🔓 [PDF {index}/{total}] Acquired PDF semaphore, starting: {pdf_name}")
            try:
                return await self.process_single_pdf(pdf_data, pdf_name, zip_folder, temp_dir, index)
            finally:
                logger.debug(f"🔒 [PDF {index}/{total}] Released PDF semaphore")

    async def process_zip_file(self, zip_key: str, limit_pdfs: Optional[int] = None):
        """
        Process all PDFs in a ZIP file using streaming from S3.

        Args:
            zip_key: S3 key of the ZIP file
            limit_pdfs: Optional limit on number of PDFs to process (for testing)
        """
        zip_folder = self.get_zip_folder_name(zip_key)

        # Check if ZIP already completed
        if zip_key in self.checkpoint['completed_zips']:
            logger.info(f"Skipping already completed ZIP: {zip_key}")
            return

        logger.info("=" * 80)
        logger.info(f"Processing ZIP: {zip_key} → Folder: {zip_folder}")
        logger.info("=" * 80)

        async with self.checkpoint_lock:
            self.checkpoint['current_zip'] = zip_key
        await self._save_checkpoint()

        # Create temp directory with explicit cleanup
        temp_dir = tempfile.mkdtemp(prefix=f'{zip_folder}_')
        try:
            s3_uri = f's3://{self.input_bucket}/{zip_key}'

            # Configure transport params for smart-open
            transport_params = {
                'client': self.s3_client
            }

            try:
                # Stream ZIP from S3
                logger.info(f"Streaming ZIP from S3: {s3_uri}")

                with s_open(s3_uri, 'rb', transport_params=transport_params) as s3_stream:
                    with zipfile.ZipFile(s3_stream, 'r') as zip_ref:

                        # Get list of PDF files
                        all_pdf_files = [f for f in zip_ref.namelist() if f.endswith('.pdf')]
                        total_pdfs_in_zip = len(all_pdf_files)
                        logger.info(f"Found {total_pdfs_in_zip} PDF files in ZIP")

                        # Limit for testing
                        pdf_files_to_process = all_pdf_files
                        is_partial_processing = False
                        if limit_pdfs:
                            pdf_files_to_process = all_pdf_files[:limit_pdfs]
                            is_partial_processing = True
                            logger.info(f"⚠️  LIMITED to {limit_pdfs} PDFs for testing (ZIP will NOT be marked complete)")

                        logger.info(f"Processing {len(pdf_files_to_process)} PDFs with {self.max_concurrent_pdfs} concurrent workers")
                        logger.info(f"🔧 Semaphore config: {self.max_concurrent_pdfs} PDFs in parallel, 3 pages per PDF = max {self.max_concurrent_pdfs * 3} pages total")

                        # Process PDFs in batches to avoid memory/disk issues
                        # Extract ONLY current batch to disk, process, clean up, repeat
                        batch_size = self.max_concurrent_pdfs * 2  # Extract 2x semaphore size
                        total_pdfs = len(pdf_files_to_process)

                        for batch_start in range(0, total_pdfs, batch_size):
                            batch_end = min(batch_start + batch_size, total_pdfs)
                            batch = pdf_files_to_process[batch_start:batch_end]
                            logger.info(f"  Extracting batch {batch_start//batch_size + 1}: PDFs {batch_start+1}-{batch_end}/{total_pdfs}")

                            # Extract only this batch
                            for pdf_name in batch:
                                pdf_data = zip_ref.read(pdf_name)
                                pdf_path = os.path.join(temp_dir, os.path.basename(pdf_name))
                                with open(pdf_path, 'wb') as f:
                                    f.write(pdf_data)

                            # Process this batch in parallel
                            tasks = []
                            for i, pdf_name in enumerate(batch, batch_start + 1):
                                pdf_file_path = os.path.join(temp_dir, os.path.basename(pdf_name))
                                task = self._process_single_pdf_from_disk_with_semaphore(
                                    pdf_file_path, pdf_name, zip_folder, temp_dir, i, total_pdfs
                                )
                                tasks.append(task)

                            await asyncio.gather(*tasks, return_exceptions=True)

                            # Clean up batch PDFs from disk
                            for pdf_name in batch:
                                pdf_path = os.path.join(temp_dir, os.path.basename(pdf_name))
                                if os.path.exists(pdf_path):
                                    os.remove(pdf_path)

                # IMPORTANT: Only mark ZIP as completed if ALL PDFs were processed (no limit applied)
                if is_partial_processing:
                    logger.info(f"\n⚠️  Partial processing complete ({len(pdf_files_to_process)}/{total_pdfs_in_zip} PDFs)")
                    logger.info(f"   ZIP NOT marked complete - remaining PDFs will be processed on next run")
                else:
                    # All PDFs processed - mark ZIP as completed
                    async with self.checkpoint_lock:
                        self.checkpoint['completed_zips'].append(zip_key)
                    self.stats['total_zips_processed'] += 1
                    logger.info(f"\n✓ Completed ZIP: {zip_key} ({total_pdfs_in_zip} PDFs)")

                # Clear current_zip regardless (we're done with this ZIP for now)
                async with self.checkpoint_lock:
                    self.checkpoint['current_zip'] = None
                await self._save_checkpoint()

            except Exception as e:
                logger.error(f"Error processing ZIP {zip_key}: {e}")
                logger.debug(traceback.format_exc())
                self.stats['total_errors'] += 1

        finally:
            # CRITICAL: Always clean up temp directory, even on errors
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as cleanup_err:
                    logger.error(f"Failed to cleanup temp directory {temp_dir}: {cleanup_err}")

    async def run(self, limit_zips: Optional[int] = None, limit_pdfs: Optional[int] = None):
        """
        Run the full extraction pipeline.

        Args:
            limit_zips: Optional limit on number of ZIPs to process
            limit_pdfs: Optional limit on PDFs per ZIP (for testing)
        """
        self.stats['start_time'] = datetime.now().isoformat()

        logger.info("=" * 80)
        logger.info("VOTER EXTRACTION PIPELINE - STARTING")
        logger.info("=" * 80)
        logger.info(f"Input bucket: s3://{self.input_bucket}/")
        logger.info(f"Output bucket: s3://{self.output_bucket}/")
        logger.info(f"Checkpoint: S3 (s3://{self.output_bucket}/checkpoints/) + Local ({self.checkpoint_file})")
        logger.info(f"Max concurrent PDFs: {self.max_concurrent_pdfs}")
        logger.info("=" * 80)

        # List ZIP files
        zip_files = self.list_zip_files()

        # Filter out completed ZIPs
        remaining_zips = [z for z in zip_files if z not in self.checkpoint['completed_zips']]
        logger.info(f"Remaining ZIPs to process: {len(remaining_zips)} / {len(zip_files)}")

        # Limit for testing
        if limit_zips:
            remaining_zips = remaining_zips[:limit_zips]
            logger.info(f"Limited to {limit_zips} ZIPs for testing")

        # Process each ZIP
        for zip_key in remaining_zips:
            if self.shutdown_requested:
                logger.warning("⚠️  Shutdown requested, stopping after current ZIP...")
                break
            await self.process_zip_file(zip_key, limit_pdfs=limit_pdfs)

        self.stats['end_time'] = datetime.now().isoformat()

        # Print final statistics
        self._print_statistics()

    def _print_statistics(self):
        """Print final statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("EXTRACTION COMPLETE - STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total ZIPs processed: {self.stats['total_zips_processed']}")
        logger.info(f"Total PDFs processed: {self.stats['total_pdfs_processed']}")
        logger.info(f"Total voter cards extracted: {self.stats['total_cards_extracted']}")
        logger.info(f"Total errors: {self.stats['total_errors']}")

        if self.stats['start_time'] and self.stats['end_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            end = datetime.fromisoformat(self.stats['end_time'])
            duration = end - start
            logger.info(f"Total duration: {duration}")

        logger.info("=" * 80)


async def main():
    """Main entry point with graceful shutdown handling."""
    # Initialize with parallel processing (optimized for OCR API rate limits)
    pipeline = VoterExtractionPipeline(
        max_concurrent_pdfs=12  # 12 concurrent PDFs
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.warning(f"\n⚠️  Received signal {signum}, initiating graceful shutdown...")
        pipeline.shutdown_requested = True

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Production: No limits - process all ZIPs and PDFs
        await pipeline.run()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  KeyboardInterrupt received, saving checkpoint and exiting...")
        await pipeline._save_checkpoint()
        logger.info("✓ Checkpoint saved before exit")
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}")
        logger.debug(traceback.format_exc())
        await pipeline._save_checkpoint()
        raise


if __name__ == "__main__":
    asyncio.run(main())
