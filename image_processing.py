import cv2
import numpy as np


def detect_and_align_card(image):
    """Detect a four-corner card and correct its perspective."""

    if image is None or image.size == 0:
        raise ValueError("Invalid image supplied")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    image_area = image.shape[0] * image.shape[1]

    for contour in contours[:20]:
        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approx) != 4:
            continue

        if cv2.contourArea(approx) < image_area * 0.10:
            continue

        points = approx.reshape(4, 2).astype(np.float32)

        rect = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).ravel()

        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]

        tl, tr, br, bl = rect

        width = int(max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl),
        ))

        height = int(max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl),
        ))

        if width < 100 or height < 60:
            continue

        destination = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(
            rect,
            destination,
        )

        aligned = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
        )

        return aligned, True

    # If no suitable boundary is detected, use the full image.
    return image.copy(), False


def preprocess_for_ocr(image):
    """Reduce noise and improve contrast for OCR."""

    if image is None or image.size == 0:
        raise ValueError("Invalid image supplied")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return thresholded