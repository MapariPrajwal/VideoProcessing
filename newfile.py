import cv2
import pytesseract
import os
import shutil
from PIL import Image
if os.path.exists('output'):
    shutil.rmtree('output')

os.makedirs('output')

cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        cv2.imshow('window-name', frame)

        # Check if the 'T' key is pressed
        key = cv2.waitKey(10)
        if key & 0xFF == ord('t'):
            cv2.imwrite("./output/frame%d.jpg" % count, frame)
            count += 1

        # Break the loop if the 'Q' key is pressed
        elif key & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# OCR using Tesseract
for i in range(count):
    img_path = f"./output/frame{i}.jpg"
    car_image = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)

    # Convert the image to 'L' mode (grayscale)
    car_image_pil = Image.fromarray((car_image * 255).astype('uint8'))
    car_image_pil = car_image_pil.convert('L')

    # Use Tesseract OCR to recognize text
    text = pytesseract.image_to_string(car_image_pil)

    # Print the recognized text
    print(f"Text in frame {i + 1}: {text}")