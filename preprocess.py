

# Capture frames
cap = cv2.VideoCapture()
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_rgb = cap.read()[1] for _ in range(total_frame)
# Convert RGB to YCbCr and remain Y channel
frames_y = [cv2.cvtColor(i, cv2.COLOR_BGR2YCR_CB)[0] for i in frames_rgb]

# resize to 100*100 or crop 100*100 pieces?

H = cfg.motion_estimation.H
W = cfg.motion_estimation.W
