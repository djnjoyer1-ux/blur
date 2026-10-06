import os
import cv2
import dlib

# Load pre-trained face detection model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to blur faces
def blur_faces(img, faces):
    # Loop through detected faces
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        
        # Extract the region of interest (ROI) and apply Gaussian blur
        roi = img[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)  # Adjust blur parameters as needed
        
        # Replace the original face with the blurred face
        img[y1:y2, x1:x2] = blurred_roi

    return img

# Function to process a video file
def process_video(input_video):
    # Load the video
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the codec and create VideoWriter object
    output_video = os.path.splitext(input_video)[0] + '_blurred_male_faces.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray)

        # Blur only male faces
        for face in faces:
            shape = predictor(gray, face)
            # Example of a simple gender classifier based on landmarks
            # You may need to replace this with a more sophisticated method
            # depending on your requirements
            is_male = True  # Replace with actual condition based on your model
            
            if is_male:
                frame = blur_faces(frame, [face])

        # Write the processed frame
        out.write(frame)

        # Display progress
        frame_count += 1
        print(f'Processing frame {frame_count}/{total_frames}')

    # Release everything when finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved as '{output_video}'")

# Iterate over all files in the current directory
current_directory = os.getcwd()
video_files = [file for file in os.listdir(current_directory) if file.endswith('.mp4')]

# Process each video file
for video_file in video_files:
    print(f'Processing video: {video_file}')
    process_video(video_file)
