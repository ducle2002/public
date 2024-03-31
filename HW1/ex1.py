import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load an image from file as function
def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    return cv2.imread(image_path)


# Display an image as function
def display_image(image, title="Image"):
    """
    Display an image using matplotlib. Rembember to use plt.show() to display the image
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')


# grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path, image)


# flip an image as function
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    return cv2.flip(image, 1)


# rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


# Adjust brightness of an image
def adjust_brightness(image, alpha, beta):
    """
    Adjust the brightness of an image using the alpha and beta parameters of the formula:
        output_image = alpha * input_image + beta
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


# Adjust contrast of an image
def adjust_contrast(image, alpha):
    """
    Adjust the contrast of an image using the alpha parameter of the formula:
        output_image = alpha * input_image + beta
    """
    adjusted_image = np.clip(alpha * image, 0, 255).astype(np.uint8)
    return adjusted_image


if __name__ == "__main__":
    # Load an image from file
    img = load_image("images/uet.png")

    # Display the original image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/uet_gray.png")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/uet_gray_rotated.png")

    # Adjust brightness of the original image
    img_brighter = adjust_brightness(img, alpha=1.2, beta=10)

    # Display the brighter image
    display_image(img_brighter, "Brighter Image")

    # Adjust contrast of the original image
    img_high_contrast = adjust_contrast(img, alpha=1.5)

    # Display the high contrast image
    display_image(img_high_contrast, "High Contrast Image")

    # Show the images
    plt.show()
