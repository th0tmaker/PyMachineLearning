import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load your images and convert them to NumPy arrays
content_image = Image.open("data/content_images/content_img(mia).png")

style_image_1 = Image.open("data/style_images/style_img(purp).png")
style_image_2 = Image.open("data/style_images/style_img(cid).png")
style_image_3 = Image.open("data/style_images/style_img(fervor).png")

output_image_1_10100 = Image.open("data/output_images/output_image_10e100es(purp).jpg")
output_image_2_10100 = Image.open("data/output_images/output_image_10e100es(cid).jpg")
output_image_3_10100 = Image.open("data/output_images/output_image_10e100es(fervor).jpg")

output_image_1_20200 = Image.open("data/output_images/output_image_20e200es(purp).jpg")
output_image_2_20200 = Image.open("data/output_images/output_image_20e200es(cid).jpg")
output_image_3_20200 = Image.open("data/output_images/output_image_20e200es(fervor).jpg")

# Convert PIL images to RGB mode before converting to NumPy arrays
content_image_arr = np.array(content_image.convert("RGB"))
style_image_1_arr = np.array(style_image_1.convert("RGB"))
style_image_2_arr = np.array(style_image_2.convert("RGB"))
style_image_3_arr = np.array(style_image_3.convert("RGB"))
output_image_1_arr_10100 = np.array(output_image_1_10100.convert("RGB"))
output_image_2_arr_10100 = np.array(output_image_2_10100.convert("RGB"))
output_image_3_arr_10100 = np.array(output_image_3_10100.convert("RGB"))
output_image_1_arr_20200 = np.array(output_image_1_20200.convert("RGB"))
output_image_2_arr_20200 = np.array(output_image_2_20200.convert("RGB"))
output_image_3_arr_20200 = np.array(output_image_3_20200.convert("RGB"))

# Set number of rows and columns
num_of_rows = 3
num_of_cols = 4

# Increase the size of the figure (adjust the figsize as needed)
fig, axs = plt.subplots(num_of_rows, num_of_cols, figsize=(20, 12))

# Remove spacing between subplots
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

# Flatten the grid of subplots for easier indexing
axs = axs.ravel()

# Define a list of images to display in each row (replace with your image arrays)
row1_images = [content_image_arr, style_image_1_arr, output_image_1_arr_10100, output_image_1_arr_20200]
row2_images = [content_image_arr, style_image_2_arr, output_image_2_arr_10100, output_image_2_arr_20200]
row3_images = [content_image_arr, style_image_3_arr, output_image_3_arr_10100, output_image_3_arr_20200]

# Loop through rows and columns to plot the images
for i in range(num_of_rows):
    for j in range(num_of_cols):
        extent = [0, 1, 0, 1]  # Set extent to occupy the entire subplot
        if i == 0:
            axs[i * num_of_cols + j].imshow(row1_images[j], extent=extent)
        elif i == 1:
            axs[i * num_of_cols + j].imshow(row2_images[j], extent=extent)
        elif i == 2:
            axs[i * num_of_cols + j].imshow(row3_images[j], extent=extent)
        axs[i * num_of_cols + j].axis('off')

        # Set aspect ratio to be equal
        axs[i * num_of_cols + j].set_aspect('auto')

# Save the plotted images as a PNG file
plt.savefig('NAST_collage.png', dpi=300, bbox_inches='tight')

# Ensure proper spacing between subplots
plt.show()