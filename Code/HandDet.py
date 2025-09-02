import cv2
import numpy as np

# A logical fix for some OpenCV versions
try:
    _ = cv2.threshold
except AttributeError:
    pass

# Global variables
background = None
accumulated_weight = 0.1  # Decreased weight for a more adaptable background

# ROI coordinates (fixed to be non-zero width/height)
roi_top = 100
roi_bottom = 500
roi_right = 800
roi_left = 100

def calc_accum_avg(frame, accumulated_weight):
    """
    Calculates the accumulated average of the background.
    """
    global background
    if background is None:
        background = frame.copy().astype("float")
        return

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, threshold_min=25):
    """
    Segments the hand from the background.
    """
    diff = cv2.absdiff(background.astype("uint8"), frame)
    
    # Use cv2.threshold correctly
    ret, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)
    
    # Check for correct number of return values from findContours
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    else:
        # Get the largest contour area
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)

def count_fingers(thresholded, hand_segment):
    """
    Counts the number of fingers in the hand segment.
    """
    conv_hull = cv2.convexHull(hand_segment)
    
    # Find points for top, bottom, left, and right extremes
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    distance = np.linalg.norm(np.array([cX, cY]) - np.array(bottom))
    max_distance = np.linalg.norm(np.array([left, right, top, bottom]) - np.array([cX, cY]), axis=1).max()
    
    radius = int(0.9 * max_distance)
    circumference = (2 * np.pi * radius)
    
    circular_roi = np.zeros(thresholded.shape, dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
    
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # Check if the contour is not the wrist and is a small finger
        out_of_wrist = (cY + (cY * 0.25)) > (y + h)
        limit_points = (circumference * 0.25) > cnt.shape[0]
        
        if out_of_wrist and limit_points:
            count += 1
            
    return count

cam = cv2.VideoCapture(0)
num_frames = 0
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 50:
            cv2.putText(frame_copy, 'Wait, Getting Background', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        hand = segment(gray)
        
        if hand is not None:
            thresholded, hand_segment = hand
            
            # The contour needs to be offset by the ROI coordinates
            cv2.drawContours(frame_copy, [hand_segment + (roi_left, roi_top)], -1, (255, 0, 0), 5)
            
            fingers = count_fingers(thresholded, hand_segment)
            cv2.putText(frame_copy, str(fingers), (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Thresholded', thresholded)
            
    # Draw the ROI rectangle and increment frame counter
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 5)
    num_frames += 1

    # Write the frame to the video file
    out.write(frame_copy)
    
    # Show the final result
    cv2.imshow('Finger Count', frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
out.release()
cv2.destroyAllWindows()