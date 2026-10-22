import re
import pytesseract


# Configure the Tesseract OCR executable for Windows.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text(image):
    """Extract readable text lines from an image using Tesseract OCR."""

    if image is None or image.size == 0:
        return []

    # PSM 6 assumes text is arranged in a reasonably uniform block.
    text = pytesseract.image_to_string(
        image,
        config="--oem 3 --psm 6",
    )

    # Normalize whitespace and discard empty lines.
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return lines


def extract_fields(lines):
    """
    Extract likely employee ID fields from OCR text.

    Supports labeled fields and the sample card layout where
    the employee name appears above the job title and ID number.
    """

    fields = {
        "full_name": None,
        "document_number": None,
        "date_of_birth": None,
        "expiry_date": None,
        "raw_text": lines,
    }

    full_text = "\n".join(lines)

    # --------------------------------------------------
    # 1. Extract document / employee ID number
    # Example: ID No: 123-456-7890
    # --------------------------------------------------
    id_pattern = (
        r"\bID\s*(?:No\.?|Number|#)\s*:?\s*"
        r"([A-Z0-9]+(?:-[A-Z0-9]+)*)"
    )

    id_match = re.search(
        id_pattern,
        full_text,
        re.IGNORECASE,
    )

    if id_match:
        fields["document_number"] = id_match.group(1).strip()

    # --------------------------------------------------
    # 2. Extract full name when a label is present
    # Example: Name: Jonathan Patterson
    # --------------------------------------------------
    name_pattern = (
        r"\b(?:Full\s+Name|Name)\s*:?\s*"
        r"([A-Za-z][A-Za-z'-]*"
        r"(?:\s+[A-Za-z][A-Za-z'-]*){1,3})"
    )

    name_match = re.search(
        name_pattern,
        full_text,
        re.IGNORECASE,
    )

    if name_match:
        fields["full_name"] = name_match.group(1).strip()

    else:
        # --------------------------------------------------
        # 3. Fallback for the supplied employee card.
        # Its name is printed on separate lines, without
        # a "Name:" label, before the job title and ID No.
        # --------------------------------------------------
        name_lines = []

        stop_labels = (
            "project manager",
            "id no",
            "id number",
            "email",
            "social",
            "phone",
            "date of birth",
            "dob",
            "expiry",
            "expires",
        )

        for line in lines:
            cleaned = line.strip()
            lower = cleaned.lower()

            # Stop before the job title or labeled details.
            if any(label in lower for label in stop_labels):
                break

            # Ignore company branding and other lines containing
            # punctuation or digits. Keep simple name-like lines.
            if re.fullmatch(
                r"[A-Za-z][A-Za-z'-]*",
                cleaned,
            ):
                name_lines.append(cleaned)

        # The sample card prints "Jonathan" and "Patterson"
        # on separate lines. Use the last two name-like lines.
        if len(name_lines) >= 2:
            fields["full_name"] = " ".join(name_lines[-2:])

    # --------------------------------------------------
    # 4. Extract dates
    # Supports DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD,
    # and YYYY/MM/DD formats.
    # --------------------------------------------------
    date_pattern = (
        r"\b(?:"
        r"\d{2}[/-]\d{2}[/-]\d{4}|"
        r"\d{4}[/-]\d{2}[/-]\d{2}"
        r")\b"
    )

    all_dates = []

    for line in lines:
        lower = line.lower()
        dates = re.findall(date_pattern, line)

        all_dates.extend(dates)

        # Explicit date-of-birth label.
        if dates and any(
            label in lower
            for label in ("date of birth", "dob", "birth")
        ):
            if fields["date_of_birth"] is None:
                fields["date_of_birth"] = dates[0]

        # Explicit expiry label.
        if dates and any(
            label in lower
            for label in (
                "expiry",
                "expires",
                "expiration",
                "valid until",
            )
        ):
            if fields["expiry_date"] is None:
                fields["expiry_date"] = dates[0]

    # Fallback for cards that contain dates without labels.
    # Only infer dates when explicit labels were not found.
    if fields["date_of_birth"] is None and all_dates:
        fields["date_of_birth"] = all_dates[0]

    if fields["expiry_date"] is None and len(all_dates) >= 2:
        fields["expiry_date"] = all_dates[-1]

    return fields