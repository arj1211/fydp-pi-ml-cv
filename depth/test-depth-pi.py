import cv2
import tensorflow as tf
import numpy as np
# import urllib.request


# url, filename = ("https://github.com/intel-isl/MiDaS/releases/download/v2/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

# url, filename = ("https://github.com/intel-isl/MiDaS/releases/download/v2_1/model_opt.tflite", "model_opt.tflite")
# urllib.request.urlretrieve(url, filename)

COLORMODE = False
# input
# img = cv2.imread('dog.jpg')
cap = cv2.VideoCapture(0)
sz = 256
n_colors = 4
win_sz = sz//n_colors

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (360, 640))
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width//2,height//2))

out = cv2.VideoWriter('output_demo.avi', fourcc, 10.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)//2),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//2)))

frame_cnt = 0
cap_frames = 10*4

def hex2rgb(color_in_hex: int):
    RGB = []
    for i in range(2, -1, -1):
        RGB.append((color_in_hex >> 4*2*i) % 16**2)
    RGB = tuple(RGB)
    return RGB
# dark to light
colors = [0x65655E, 0x7D80DA, 0xB0A3D4, 0xCEBACF, 0xC6AFB1]
colors_rgb = list(map(lambda c: np.array(hex2rgb(c)), colors))

# load model
interpreter = tf.lite.Interpreter(model_path="model_opt.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

while cap.isOpened() and frame_cnt < cap_frames:

    suc, img = cap.read()

    orig_img = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    #img = cv2.flip(img, 0)

    img_resized = tf.image.resize(img, [sz,sz], method='bicubic', preserve_aspect_ratio=False)
    #img_resized = tf.transpose(img_resized, [2, 0, 1])
    img_input = img_resized.numpy()
    # mean=[0.485, 0.456, 0.406]
    mean=img_input.mean(axis=0).mean(axis=0)
    # std=[0.229, 0.224, 0.225]
    std=img_input.std(axis=0).std(axis=0)
    img_input = (img_input - mean) / std
    reshape_img = img_input.reshape(1,sz,sz,3)
    tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

    # inference
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output.reshape(sz, sz)

    # output file
    prediction = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # print(" Write image to: output.png")
    depth_min = prediction.min()
    depth_max = prediction.max()


    m = 255/(depth_max - depth_min)
    if not COLORMODE: 
        img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
    else:
        recolorf = lambda x: (m * (x - depth_min)).astype("uint8")
        img_out = recolorf(prediction)
        img_out = img_out.repeat(3).reshape((img_out.shape[0], img_out.shape[1], 3))
        i = win_sz
        c_i = 0
        while i <= sz:
            mask = (i - win_sz <= img_out[:,:,0]) & (img_out[:,:,0] < i)
            img_out[mask] = colors_rgb[c_i]
            c_i += 1
            i += win_sz
    
    # resize output
    if not COLORMODE:
        img_out = tf.image.resize(img_out.repeat(3).reshape((img_out.shape[0], img_out.shape[1], 3)), [img_out.shape[0]//2, img_out.shape[1]//2], method='bicubic', preserve_aspect_ratio=False).numpy().astype('uint8')
    else:
        img_out = tf.image.resize(img_out, [img_out.shape[0]//2, img_out.shape[1]//2], method='bicubic', preserve_aspect_ratio=False).numpy().astype('uint8')
    
    orig_img = tf.image.resize(orig_img, [orig_img.shape[0]//2,orig_img.shape[1]//2], method='bicubic', preserve_aspect_ratio=False).numpy().astype('uint8')
    # cv2.imwrite("output.png", img_out)
    # cv2.imshow("output.png", img_out)
    # Stop the program if the ESC key is pressed.
    img_out = cv2.flip(img_out, 1)
    orig_img = cv2.flip(orig_img, 1)
    #cv2.imshow('object_detector', img_out)
    #cv2.imshow('raw', orig_img)

    out.write(img_out)
    print(f'captured {frame_cnt} frame(s)')
    frame_cnt += 1

    if cv2.waitKey(1) == 27:
      break
    # plt.imshow(img_out)
    # plt.show()
    
    # if cv2.waitKey(0): break
out.release()
cap.release()
cv2.destroyAllWindows()
