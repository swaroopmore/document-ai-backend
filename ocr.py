import re

import pytesseract


# Configure the installed Tesseract executable on Windows.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text(image):
    """Extract text from an image using Tesseract OCR."""
    if image is None or image.size == 0:
        return []

    text = pytesseract.image_to_string(
        image,
        config="--psm 6",
    )

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def extract_fields(lines):
    """Extract likely ID fields using basic heuristics."""
    full_text = "\n".join(lines)

    fields = {
        "full_name": None,
        "document_number": None,
        "date_of_birth": None,
        "expiry_date": None,
        "raw_text": lines,
    }

    # Extract a name when the label is present.
    for line in lines:
        match = re.search(
            r"\b(?:full\s+)?name\s*[:\-]?\s*(.+)",
            line,
            re.IGNORECASE,
        )
        if match:
            fields["full_name"] = match.group(1).strip(" :-")
            break

    # Recognize common date formats.
    date_pattern = (
        r"\b(?:\d{2}[/-]\d{2}[/-]\d{4}"
        r"|\d{4}[/-]\d{2}[/-]\d{2})\b"
    )

    dates = []

    for line in lines:
        matches = re.findall(date_pattern, line)
        dates.extend(matches)

        lower = line.lower()

        if any(label in lower for label in (
            "dob", "birth", "date of birth"
        )) and matches:
            fields["date_of_birth"] = matches[0]

        if any(label in lower for label in (
            "expiry", "expires", "expiration", "valid until"
        )) and matches:
            fields["expiry_date"] = matches[0]

    # Fallbacks based on date order are not guaranteed.
    if fields["date_of_birth"] is None and dates:
        fields["date_of_birth"] = dates[0]

    if fields["expiry_date"] is None and len(dates) >= 2:
        fields["expiry_date"] = dates[-1]

    # Look for a document number near a relevant label.
    for line in lines:
        match = re.search(
            r"\b(?:ID|document|license|licence)"
            r"\s*(?:number|no\.?|#)?\s*[:\-]?\s*"
            r"([A-Z0-9-]{5,20})\b",
            line,
            re.IGNORECASE,
        )

        if match:
            fields["document_number"] = match.group(1)
            break

    return fields