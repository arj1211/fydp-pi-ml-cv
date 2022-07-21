import cv2
import tensorflow as tf
# import urllib.request


# url, filename = ("https://github.com/intel-isl/MiDaS/releases/download/v2/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

# url, filename = ("https://github.com/intel-isl/MiDaS/releases/download/v2_1/model_opt.tflite", "model_opt.tflite")
# urllib.request.urlretrieve(url, filename)


# input
img = cv2.imread('dog.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

img_resized = tf.image.resize(img, [256,256], method='bicubic', preserve_aspect_ratio=False)
#img_resized = tf.transpose(img_resized, [2, 0, 1])
img_input = img_resized.numpy()
# mean=[0.485, 0.456, 0.406]
mean=img_input.mean(axis=0).mean(axis=0)
# std=[0.229, 0.224, 0.225]
std=img_input.std(axis=0).std(axis=0)
img_input = (img_input - mean) / std
reshape_img = img_input.reshape(1,256,256,3)
tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

# load model
interpreter = tf.lite.Interpreter(model_path="model_opt.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# inference
interpreter.set_tensor(input_details[0]['index'], tensor)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
output = output.reshape(256, 256)

# output file
prediction = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
print(" Write image to: output.png")
depth_min = prediction.min()
depth_max = prediction.max()
img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")

# cv2.imwrite("output.png", img_out)
cv2.imshow("output.png", img_out)
# plt.imshow(img_out)
# plt.show()
cv2.waitKey(0)