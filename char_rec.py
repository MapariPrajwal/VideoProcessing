import cv2
import numpy as np
import pytesseract

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and aid in contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to highlight edges
    edged = cv2.adaptiveThreshold(blurred, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 4)

    return edged

# Function to find contours in the image
def find_contours(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    return contours

# Function to find the number plate
def find_number_plate(image_path):
    edged = preprocess_image(image_path)
    contours = find_contours(edged)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            number_plate = approx
            break

    return number_plate

# Function to extract text from the number plate
def extract_text(image, number_plate):
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [number_plate], 0, (255, 255, 255), -1)
    result = cv2.bitwise_and(image, mask)
    plate_text = pytesseract.image_to_string(result, config='--psm 8 --oem 3')

    return plate_text

# Example usage
image_path = 'C:/Users/prajw/Python/VideoProcessing/N248.jpeg'
number_plate = find_number_plate(image_path)

if number_plate is not None:
    image = cv2.imread(image_path)
    plate_text = extract_text(image, number_plate)
    print("Number Plate:", plate_text)
    cv2.drawContours(image, [number_plate], 0, (0, 255, 0), 2)
    cv2.imshow('Detected Number Plate', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Number plate not found.")
