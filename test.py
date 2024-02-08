import tensorflow as tf
import cv2
import model

# Function to convert image to uint8
def im2uint8(x):
    if isinstance(x, tf.Tensor):
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)

# Load image
img = cv2.imread('eg.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
half_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
img = img.reshape((1, 128, 128, 3))
half_img = half_img.reshape((1, 64, 64, 3))

# Create placeholders
image = tf.placeholder("float", [1, 128, 128, 3])
half_img_placeholder = tf.placeholder("float", [1, 64, 64, 3])

# Build the Semantic_face model
SF = model.Semantic_face('./net_P_P_S_F.mat', './net_G_P_S_F.mat')
with tf.name_scope("Semantic_face"):
    SF.build(image, half_img_placeholder)

# Convert output to uint8
out = im2uint8(SF.convG32)

# Initialize TensorFlow session
with tf.Session() as sess:
    # Run the model
    out_image = sess.run(out, feed_dict={image: img, half_img_placeholder: half_img})
    out_image = out_image[0]

# Convert output to BGR and save
out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('eg_deblur.png', out_image)
print("Done")
