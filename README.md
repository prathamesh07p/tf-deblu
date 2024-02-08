# Semantic Face Model

This project implements a Semantic Face model using TensorFlow. The model consists of two main components: P model for face parsing and G model for generating facial images.

## Overview

The Semantic Face model is divided into two parts:

- **P Model**: This part handles face parsing, which involves segmenting various parts of the face such as eyes, nose, and mouth. It consists of down-sampling layers followed by up-sampling layers to process the input image and generate face labels.

- **G Model**: The Generative Model takes the generated face labels from the P model along with a blurred input image and generates a high-resolution facial image. It uses convolutional layers and residual blocks to perform this task.

## Usage

To use the Semantic Face model, follow these steps:

1. Initialize the `Semantic_face` class with paths to the P model file and the G model file.

   ```python
   semantic_face_model = Semantic_face(P_model_path='path_to_P_model_file.mat', G_model_path='path_to_G_model_file.mat')
   ```

2. Build the model by providing input data, such as a blurred image and a half-image.

   ```python
   semantic_face_model.build(blur=input_blurred_image, halfImg=input_half_image)
   ```

3. Access the generated face labels and high-resolution facial image.

   ```python
   face_labels = semantic_face_model.faceLabel
   high_res_image = semantic_face_model.convG_t16
   ```

## File Structure

The project includes the following files:

- `semantic_face.py`: Contains the implementation of the `Semantic_face` class, which defines the P model and G model components along with their respective layers and operations.
- `P_model.mat`: Pre-trained weights and parameters for the P model.
- `G_model.mat`: Pre-trained weights and parameters for the G model.

## Dependencies

This project requires the following dependencies:

- TensorFlow: `pip install tensorflow`
- NumPy: `pip install numpy`
- SciPy: `pip install scipy`

## Note

Ensure that you have the necessary permissions to use the pre-trained model files (`P_model.mat` and `G_model.mat`) and provide the correct paths to these files when initializing the `Semantic_face` class.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

This README provides an overview of the Semantic Face model, its usage, file structure, dependencies, and licensing information. Feel free to expand it further with additional details or instructions as needed.
