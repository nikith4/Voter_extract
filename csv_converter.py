#!/usr/bin/env python3
"""
CSV conversion utilities for voter card data.
Converts extracted JSON voter data to CSV format.
"""

import csv
import os
import re
from typing import List, Dict


def is_valid_card(card: Dict) -> bool:
    """
    Check if card has minimum required data.

    Args:
        card: Voter card dictionary

    Returns:
        True if card has at least voter_id OR name
    """
    voter_id = card.get('voter_id', '').strip()
    name = card.get('name', '').strip()

    # Card must have either voter_id or name (not both empty)
    return bool(voter_id or name)


def clean_card_data(card: Dict) -> Dict:
    """
    Clean voter card data by removing OCR artifacts and normalizing values.

    Args:
        card: Raw voter card dictionary

    Returns:
        Cleaned voter card dictionary
    """
    cleaned = card.copy()

    # Clean name field - remove "Photo", "Available", etc.
    if 'name' in cleaned:
        name = cleaned['name']
        name = re.sub(r'\s+Photo\s*', ' ', name)  # Remove "Photo"
        name = re.sub(r'\s+Available\s*', ' ', name)  # Remove "Available"
        name = re.sub(r'\s+De\s+Available\s*', ' ', name)  # Remove "De Available"
        name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
        cleaned['name'] = name.strip()

    # Clean relation_name
    if 'relation_name' in cleaned:
        rel_name = cleaned['relation_name']
        rel_name = re.sub(r'\s+Photo\s*', ' ', rel_name)
        rel_name = re.sub(r'\s+', ' ', rel_name)
        cleaned['relation_name'] = rel_name.strip()

    # Normalize house_number: Convert "-" to empty string
    if 'house_number' in cleaned and cleaned['house_number'] == '-':
        cleaned['house_number'] = ''

    # Validate and clean age
    if 'age' in cleaned:
        age = str(cleaned['age']).strip()
        if age and not age.isdigit():
            cleaned['age'] = ''  # Invalid age, clear it
        elif age and (int(age) < 18 or int(age) > 120):
            # Suspicious age, but keep it (might be valid in edge cases)
            pass

    # Normalize gender
    if 'gender' in cleaned:
        gender = cleaned['gender'].strip().lower()
        if gender in ['m', 'male']:
            cleaned['gender'] = 'Male'
        elif gender in ['f', 'female', 'femalexq']:  # Handle OCR error "Femalexq"
            cleaned['gender'] = 'Female'
        elif gender:
            # Keep original if not recognized
            cleaned['gender'] = cleaned['gender'].strip()

    return cleaned


def voter_cards_to_csv(cards: List[Dict], output_path: str) -> str:
    """
    Convert list of voter card dictionaries to CSV file.

    Args:
        cards: List of voter card dictionaries with fields:
               - voter_id, serial_no, name, relation_type, relation_name,
                 house_number, age, gender
        output_path: Path where CSV file should be saved

    Returns:
        Path to the created CSV file
    """
    if not cards:
        # Create empty CSV with headers
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=get_csv_headers())
            writer.writeheader()
        return output_path

    # Define CSV headers
    headers = get_csv_headers()

    # Filter and clean cards
    valid_cards = []
    for card in cards:
        # Validate card has minimum required data
        if is_valid_card(card):
            # Clean the card data
            cleaned_card = clean_card_data(card)
            valid_cards.append(cleaned_card)

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for card in valid_cards:
            # Ensure all fields exist (fill missing with empty string)
            row = {header: card.get(header, '') for header in headers}
            writer.writerow(row)

    return output_path


def get_csv_headers() -> List[str]:
    """
    Get standard CSV headers for voter card data.

    Returns:
        List of column names
    """
    return [
        'serial_no',
        'voter_id',
        'name',
        'relation_type',
        'relation_name',
        'house_number',
        'age',
        'gender'
    ]


def validate_card_data(card: Dict) -> Dict:
    """
    Validate and clean voter card data.

    Args:
        card: Raw voter card dictionary

    Returns:
        Cleaned voter card dictionary
    """
    cleaned = {}

    # Required fields
    for field in get_csv_headers():
        value = card.get(field, '')

        # Clean whitespace
        if isinstance(value, str):
            value = value.strip()

        # Validate specific fields
        if field == 'age':
            # Ensure age is numeric or empty
            if value and not str(value).isdigit():
                value = ''

        if field == 'gender':
            # Normalize gender values
            if value and value.lower() in ['m', 'male']:
                value = 'Male'
            elif value and value.lower() in ['f', 'female']:
                value = 'Female'

        cleaned[field] = value

    return cleaned


def merge_csv_files(csv_files: List[str], output_path: str) -> str:
    """
    Merge multiple CSV files into one.
    Useful for combining results from multiple PDFs.

    Args:
        csv_files: List of CSV file paths to merge
        output_path: Path for merged CSV output

    Returns:
        Path to merged CSV file
    """
    if not csv_files:
        raise ValueError("No CSV files to merge")

    headers = get_csv_headers()

    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()

        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                continue

            with open(csv_file, 'r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    writer.writerow(row)

    return output_path


if __name__ == "__main__":
    # Test the converter
    test_cards = [
        {
            'serial_no': '1',
            'voter_id': 'ABC1234567',
            'name': 'John Doe',
            'relation_type': 'Father',
            'relation_name': 'James Doe',
            'house_number': '123',
            'age': '35',
            'gender': 'Male'
        },
        {
            'serial_no': '2',
            'voter_id': 'XYZ9876543',
            'name': 'Jane Smith',
            'relation_type': 'Husband',
            'relation_name': 'Bob Smith',
            'house_number': '456',
            'age': '32',
            'gender': 'Female'
        }
    ]

    output = '/tmp/test_voters.csv'
    voter_cards_to_csv(test_cards, output)
    print(f"Test CSV created at: {output}")

    # Read and display
    with open(output, 'r') as f:
        print(f.read())
