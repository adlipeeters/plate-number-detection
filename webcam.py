# import cv2
# import os
#
# # Create a folder to save frames
# if not os.path.exists('saved_frames'):
#     os.makedirs('saved_frames')
#
# # Start the webcam
# cap = cv2.VideoCapture(0)
#
# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
#
# frame_count = 0
#
# try:
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#
#         # If frame is read correctly ret is True
#         if not ret:
#             print("Error: Can't receive frame (stream end?). Exiting ...")
#             break
#
#         # Display the resulting frame
#         cv2.imshow('Webcam', frame)
#
#         # Save the frame as a JPEG file
#         frame_filename = os.path.join('saved_frames', f'frame_{frame_count:04d}.jpg')
#         cv2.imwrite(frame_filename, frame)
#         frame_count += 1
#
#         # Exit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# except KeyboardInterrupt:
#     # Handle any cleanup here
#     print("Stopping the webcam capture.")
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
import cv2
import requests
import base64
import numpy as np

def capture_and_process_frame(flask_url):
    # Open a connection to the webcam (default is 0, you can change if you have multiple webcams)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Encode the frame as a JPEG image
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        # Send the image to the Flask endpoint
        response = requests.post(
            f"{flask_url}/process_image",
            files={"file": ("frame.jpg", img_bytes, "image/jpeg")}
        )

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()

            # Decode the base64 encoded image back into a numpy array
            img_base64 = response_data['image_base64']
            img_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Display the processed frame
            cv2.imshow('Processed Frame', processed_frame)
        else:
            print("Failed to process image:", response.json().get('error'))

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Usage
flask_url = "http://localhost:8080"  # Replace with your Flask server URL
capture_and_process_frame(flask_url)
