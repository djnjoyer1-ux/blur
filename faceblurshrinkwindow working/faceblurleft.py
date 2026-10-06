import cv2
import pyautogui
import numpy as np

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the region of interest (left side of the screen)
screen_width, screen_height = pyautogui.size()
left_region = (0, 0, screen_width // 2, screen_height)  # Adjust as needed

# Create a named window with fixed size
cv2.namedWindow('Blurred Faces', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Blurred Faces', screen_width // 2, screen_height)  # Set the window size (width, height)

while True:
    # Capture screenshot of the left side
    screenshot = pyautogui.screenshot(region=left_region)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    
    # Detect faces
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Blur faces
    for (x, y, w, h) in faces:
        face_region = screenshot[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        screenshot[y:y+h, x:x+w] = blurred_face
    
    # Update existing window with blurred screenshot
    cv2.imshow('Blurred Faces', screenshot)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
