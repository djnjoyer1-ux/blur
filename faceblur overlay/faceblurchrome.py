import cv2
import numpy as np
import pygetwindow as gw

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to find Chrome window
def find_chrome_window():
    # Iterate over all open windows and find the one with 'Google Chrome' in its title
    chrome_windows = [win for win in gw.getAllTitles() if 'Google Chrome' in win]
    if chrome_windows:
        return gw.getWindowsWithTitle(chrome_windows[0])[0]
    else:
        return None

# Create a named window with WINDOW_NORMAL flag for resizable window
cv2.namedWindow('Blurred Faces', cv2.WINDOW_NORMAL)

while True:
    # Get handle of the Chrome window
    chrome_window = find_chrome_window()

    if chrome_window:
        # Capture screenshot of the Chrome window
        screenshot = chrome_window.capture()

        # Convert the screenshot to a numpy array and then to BGR format
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
