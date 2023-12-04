import cv2
import os

PATH = '/home/fizzer/ros_ws/src/enph353_competition_controller/image_testing/obstacle_images/'

# # Load two images
# img1 = cv2.imread(PATH + 'image_113.png', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(PATH + 'image_106.png', cv2.IMREAD_GRAYSCALE)

# height, width = img1.shape

# img1 = img1[int(height/4):int(3*height/4), int(width/4):int(3*width/4)]
# img2 = img2[int(height/4):int(3*height/4), int(width/4):int(3*width/4)]

# # Create SIFT detector
# sift = cv2.SIFT_create()

# # Detect keypoints and compute descriptors
# keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# # Create a BFMatcher (Brute-Force Matcher)
# bf = cv2.BFMatcher()

# # Match descriptors
# matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# # Apply ratio test
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)

# # Draw matches
# img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Display the result
# cv2.imshow('SIFT Matches', img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Path to the directory containing the image sequence
# image_sequence_path = PATH

# # Get the list of image filenames in the directory
# image_filenames = [img for img in os.listdir(image_sequence_path) if img.endswith(".png")]

# # Sort the image filenames to ensure proper sequence
# image_filenames.sort()

# # Specify the output video file
# output_video_path = 'output_video.avi'

# # Get the first image to determine the video frame size
# first_image = cv2.imread(os.path.join(image_sequence_path, image_filenames[0]))
# frame_size = (first_image.shape[1], first_image.shape[0])

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose other codecs like 'MJPG' or 'MP4V'
# out = cv2.VideoWriter(output_video_path, fourcc, 1.0, frame_size)

# # Loop through the image sequence and write each frame to the video
# for image_filename in image_filenames:
#     image_path = os.path.join(image_sequence_path, image_filename)
#     frame = cv2.imread(image_path)
#     out.write(frame)

# # Release the VideoWriter object
# out.release()


# # Load the pedestrian detector (HOG-based)
# pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# # Capture video from your robot's camera (replace with your camera code)
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the video stream
#     ret, frame = cap.read()

#     # Convert the frame to grayscale for the HOG detector
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect pedestrians in the frame
#     pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw rectangles around detected pedestrians
#     for (x, y, w, h) in pedestrians:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # Display the result
#     cv2.imshow('Pedestrian Detection', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close the window
# cap.release()
# cv2.destroyAllWindows()


# # Optionally, you can display the video using VideoCapture
# cap = cv2.VideoCapture(output_video_path)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow('Image Sequence to Video', frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# Path to the directory containing the images
image_directory = PATH

# Path to the pre-trained Haar Cascade classifier XML file for pedestrian detection
pedestrian_classifier_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'

# Create a pedestrian cascade
pedestrian_cascade = cv2.CascadeClassifier(pedestrian_classifier_path)

# Loop through each image in the directory
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions as needed
        # Read the image
        img_path = os.path.join(image_directory, filename)
        img = cv2.imread(img_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect pedestrians in the image
        pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the pedestrians
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the result (optional)
        cv2.imshow('Detected Pedestrians', img)
        cv2.waitKey(0)

# Close the OpenCV window
cv2.destroyAllWindows()