import cv2
import matplotlib.pyplot as plt

image_path = "/content/OIP.webp"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check image_path.")

print("Image loaded successfully")

height, width, channels = image.shape
print("Height:", height)
print("Width:", width)
print("Channels:", channels)
print("Total pixels:", height * width)
print("Data type:", image.dtype)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Grayscale image shape:", gray_image.shape)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.show()

cv2.imwrite("output_gray.jpg", gray_image)
print("Grayscale image saved as output_gray.jpg")
