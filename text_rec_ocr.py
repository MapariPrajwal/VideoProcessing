import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/tesseract-ocr-w64-setup-5.3.3.20231005.exe'

def text_recognition(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    extracted_text = pytesseract.image_to_string(binary_image)
    return extracted_text

if __name__ == "__main__":
    image_path = 'N248.jpeg'
    result_text = text_recognition(image_path)
    print("Extracted Text:")
    print(result_text)
