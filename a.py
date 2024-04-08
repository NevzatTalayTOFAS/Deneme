import cv2

# Local video file path
video_path = "your_video_file_path_here.mp4"

# Video capture
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize variables
car_count = 0
min_contour_area = 500

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            car_count += 1
    
    # Print number of cars
    print(f'Number of cars: {car_count}')
    
    # Exit if end of video is reached
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
