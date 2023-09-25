import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Disable Tensorflow debugging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###############################################
# PART I <PREPARE, DE/PROCESS, DISPLAY IMAGES>
###############################################

# Provide Path/Reference for desired content image and style image
content_img_path = "content_img(mstr2).png"
style_img_path = "style_img(ksvl).png"


# Load and process the image appropriately to optimize model training and output
def load_process_image(img_path, img_size=512):
    img = Image.open(img_path)  # load image from path
    img = img.convert("RGB")  # convert image to 'RGB' format (3 channels: red, greed, blue)
    img.thumbnail([img_size, img_size])  # resize image dimensions according to img_size (512, maintain aspect ratio)
    img = np.array(img, dtype=np.float32)  # convert image to numpay ndarray object and cast as type float32
    img = img / 255.0  # scale the image from value range [0, 255] to [0, 1] for easier processing
    img = np.expand_dims(img, axis=0)  # (1, height, width, channels), where 1 indicates there's one image in the batch

    return img


# Deprocess the image from value range [0, 1] to [0, 255] and return numpy ndarray object as unsigned 8-bit integer
def deprocess_image(img):
    img = 255 * img

    return np.array(img, np.uint8)


# Show image on screen to evaluate images are loaded and processed as intended
def show_image(img, deprocessing=True):
    if deprocessing:
        img = deprocess_image(img)  # if deprocessing is enabled, scale the image back to [0, 255]

    if np.ndim(img) > 3:
        assert img.shape[0] == 1  # ensure the batch size is 1
        img = img[0]  # extract the single image from the batch

    # Display the image using Matplotlib
    plt.imshow(img)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()


# Convert array into an image
def array_to_image(array, deprocessing=False):
    # If deprocessing, run deprocess_image function to produce higher quality image output
    if deprocessing:
        array = deprocess_image(array)  # scale from [0, 1] to [0, 255], cast as 8-bit

    # If array has more than 3 dim (batch of images (e.g.(x, 512, 512, 3)) <- if x > 1, indicates multiple images
    if np.ndim(array) > 3:
        # Size of first dim needs to equal 1, because there should be only one image in the batch (batch size of 1).
        assert array.shape[0] == 1  # if first dim does NOT equal 1, use assert to raise error
        array = array[0]  # extract the first (and only) image from the batch

    # Create an image from array and return it
    return Image.fromarray(array)


# Content Image
content_image = load_process_image(content_img_path)
print(content_image.shape)  # print shape of content image
show_image(content_image)  # show content image using matplotlib

# Style Image
style_image = load_process_image(style_img_path)
print(style_image.shape)  # print shape of style image
show_image(style_image)  # show style image using matplotlib

#####################################################
# PART II <IMPORT BASE MODEL & EXTRACT THE FEATURES>
#####################################################

# Use Keras built-in VGG19 model as base model
vgg19_base_model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False)


# Create a stylized model that takes an image as input and extracts its style features using a pre-trained VGG19 model
def stylized_model(base_model, layer_names):
    # Model will not be trained from scratch, therefore will maintain original weights and bias
    base_model.trainable = False
    # Extract the layers with output tensors of the vgg model into a list called outputs
    output_layers = [base_model.get_layer(name).output for name in layer_names]
    # Define new model which input is the image and the outputs are the style-related feature maps from selected layers
    new_stylized_model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layers)

    return new_stylized_model


# Check summary of the model architecture
vgg19_base_model.summary()

# Define which base model layers to extract for the style and the content layers
content_layers = ['block4_conv2', 'block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',
                'block5_conv2']

# Build a new content and style model using base model (VGG19) as the input model and its extracted layers
content_model = stylized_model(vgg19_base_model, content_layers)
style_model = stylized_model(vgg19_base_model, style_layers)

# Use the content model to extract the outputs from the content image
content_outputs = content_model(content_image)
# Zip the content layers and outputs together into a tuple
for layer_name, outputs in zip(content_layers, content_outputs):
    print(layer_name)  # check layer name
    print(outputs.shape)  # check shape of output

# Use the style model to extract the outputs from the style image
style_outputs = style_model(style_image)
# Zip the style layers and outputs together into a tuple
for layer_name, outputs in zip(style_layers, style_outputs):
    print(layer_name)  # check layer name
    print(outputs.shape)  # check shape of output

# Build main model that extracts and combines features of both the style and content images
model = stylized_model(vgg19_base_model, style_layers + content_layers)


# Separate the combined style and content features from the main model into their respective seperate dictionaries
def get_output_dict(main_model, inputs):
    inputs = inputs * 255.0  # scale the value of the inputs (images) back to [0, 255] scale
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)  # preprocess inputs to suitable format
    style_length = len(style_layers)  # save length of style layers into its variable
    model_outputs = main_model(preprocessed_input)  # pass the preprocessed inputs through main model to extract outputs
    style_output, content_output = model_outputs[:style_length], model_outputs[style_length:]  # separate outputs
    content_dict = {name: value for name, value in zip(content_layers, content_output)}  # create content dictionary
    style_dict = {name: value for name, value in zip(style_layers, style_output)}  # create style dictionary

    # Return a dictionary that has the name as its key and the respective output dictionary as the pair
    return {'content': content_dict, 'style': style_dict}


# Get the extracted features of the style image and save the results
results = get_output_dict(model, style_image)

# View the content image layers and shape of the extracted output feature maps
print("Content Image output feature maps: ")
for layer_name, output in sorted(results['content'].items()):
    print(layer_name)  # check layer name
    print(output.shape)  # check shape of output

# View style image layers and shape of the extracted output feature maps
print("Style Image output feature maps: ")
for layer_name, output in sorted(results['style'].items()):
    print(layer_name)  # check layer name
    print(output.shape)  # check shape of output

# Save the target layers from each the content and the style image into their own variables
content_targets = get_output_dict(model, content_image)['content']  # store the content features
style_targets = get_output_dict(model, style_image)['style']  # store the style features


##################################################
# PART III <CALCULATE LOSS & TACKLE OPTIMISATION>
##################################################

# Define a gram matrix function that uses algebra to capture and represent the style of the image
def gram_matrix(x):
    gram = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    return gram / tf.cast(x.shape[1] * x.shape[2], tf.float32)


# Define content loss function
def content_loss(placeholder, content):
    return tf.reduce_mean(tf.square(placeholder - content))


# Define style loss function (using gram matrix function)
def style_loss(placeholder, style):
    s = gram_matrix(style)
    p = gram_matrix(placeholder)
    return tf.reduce_mean(tf.square(s - p))


# Define overarching total loss function
def total_loss(the_outputs, the_content_outputs, the_style_outputs, the_content_weight, the_style_weight):
    final_content = the_outputs['content']
    final_style = the_outputs['style']
    num_style_layers = len(style_layers)
    num_content_layers = len(content_layers)

    # Content Loss
    c_loss = tf.add_n([content_loss(the_content_outputs[name], final_content[name]) for name in final_content.keys()])
    c_loss *= the_content_weight / num_content_layers

    # Style Loss
    s_loss = tf.add_n([style_loss(the_style_outputs[name], final_style[name]) for name in final_style.keys()])
    s_loss *= the_style_weight / num_style_layers

    # Adding up both the content and style loss into total overall loss
    t_loss = c_loss + s_loss
    return t_loss


# Create mutable tensorflow Variable object that can be iteratively changed/optimized to generate final output image
output_image = tf.Variable(content_image, dtype=tf.float32)  # takes content_image as input with data type float 32
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)  # use gradient descent based Adam algo


# Clip image pixel values within [0, 1] range to prevent pixel saturation or values outside the displayable range
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# Create function that optimizes total loss into a final loss (calculate loss, gradient, apply optimizer and clipping)
def loss_optimizer(the_image, the_optimizer, the_content_weight, the_style_weight, the_total_variation_weight):
    with tf.GradientTape() as tape:
        dict_outputs = get_output_dict(model, the_image)
        final_loss = total_loss(dict_outputs, content_targets, style_targets, the_content_weight, the_style_weight)
        final_loss += the_total_variation_weight * tf.image.total_variation(the_image)
    grad = tape.gradient(final_loss, the_image)
    the_optimizer.apply_gradients([(grad, the_image)])
    the_image.assign(clip_0_1(the_image))

    return final_loss


#################################################
# PART IV <DEFINE LEARNING WEIGHTS & PARAMETERS>
#################################################

# Adjust these weights to apply more style, more context or more variation to the output image (think ratio based)
TOTAL_VARIATION_WEIGHT = 0.0005
CONTENT_WEIGHT = 100000000
STYLE_WEIGHT = 110000

# Adjust these learning parameters according to your needs and computational resources
EPOCHS = 10  # An 'epoch' is one complete pass through your entire dataset
STEPS_PER_EPOCH = 100  # Number of steps per epoch determines how many optimization steps you perform on the input image

###############################################
# PART V <TRAINING LOOP & GENERATE/SAVE IMAGE>
###############################################

# Perform final training & optimization for the Neural Aesthetic Style Transfer (NAST)
start = time.time()
for i in range(EPOCHS):
    print(f"Epoch: {i + 1}")
    curr_loss = 0.0  # Initialize curr_loss before entering the inner loop
    for j in tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {i + 1}/{EPOCHS}"):  # Use tqdm as a context manager
        curr_loss = loss_optimizer(output_image, optimizer, CONTENT_WEIGHT, STYLE_WEIGHT, TOTAL_VARIATION_WEIGHT)
        # Optional: Save image in every step
        # current_image = array_to_img(output_image.numpy(), deprocessing=True)
        # current_image.save(f'progress/{i}_{j}_paint.jpg')
    print(f"\nLoss: {curr_loss}")
end = time.time()
print(f"Image successfully generated in {end - start:.1f} sec")

# Show output image and save image to device
show_image(output_image.numpy(), deprocessing=True)
nast_image = array_to_image(output_image.numpy(), deprocessing=True)
nast_image.save("NAST_image.jpg")