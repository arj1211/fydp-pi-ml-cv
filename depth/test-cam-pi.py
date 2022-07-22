# based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#saving-a-video
import cv2

cap = cv2.VideoCapture(0)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('dims',w,h)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test-output.avi', fourcc, 20.0, (640, 480))
n_frames = 200
while n_frames > 0:
    ret, frame = cap.read()
    if ret == True:
        # write the flipped frame
        out.write(frame)
        n_frames -= 1
    else:
        break
    print('frames to capture: {}'.format(n_frames))

# Release everything when done
cap.release()
out.release()
