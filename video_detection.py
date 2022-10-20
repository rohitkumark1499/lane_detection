
import cv2


video_cap = cv2.VideoCapture("lane_detection_video.mp4")

while True:
    # `success` is a boolean and `frame` contains the next video frame
    success, frame = video_cap.read()
    cv2.imshow('video', frame)
    # wait 20 milliseconds between frames and break the loop if the `q` key is pressed
    if cv2.waitKey(30) == ord('q'):
        break

# we also need to close the video and destroy all Windows
video_cap.release()
cv2.destroyAllWindows()