# U2Net_Car_Segmentation

Overview:

This repository contains code and instructions for training and evaluating a U-2-Net model for car segmentation. The project includes two main tasks:

Task 1: Train the U-2-Net model using the BCEWithLogitsLoss function.
Task 2: Retrain the model using the DiceLoss function.
Both tasks involve splitting a dataset of car images and masks into training and validation sets, applying data augmentation, and evaluating model performance using the Dice Score and Intersection over Union (IoU) metrics.

Dataset:

The dataset used for training and validation consists of 20 image-mask pairs of cars. The images come in both landscape and portrait orientations, and the corresponding masks are aligned accordingly. The masks are binary, where white represents the car and black represents the background.

Directory Structure:

The repository includes the following key directories:

Image/: Contains all the car images in JPEG format.
Mask/: Contains the corresponding mask images in PNG format.

Requirements:
To run the code and train the model, ensure you have the following Python libraries installed:

PyTorch

Albumentations

NumPy

scikit-learn

tqdm

PIL (Pillow)

You can install these packages using pip. For example:

pip install torch albumentations numpy scikit-learn tqdm pillow

Task 1: Training with BCEWithLogitsLoss:-

Data Preparation: 
The dataset is split into training (15 images) and validation (5 images) sets. Data augmentation is applied during training to enhance model generalization. The image size is standardized to 320x320 pixels.

Model Initialization: 
The U-2-Net model is initialized with 3 input channels (for RGB images) and 1 output channel (for binary masks).

Loss Function and Optimizer: 
The model is trained using the BCEWithLogitsLoss function and the Adam optimizer with a learning rate of 0.0001.

Training Process:
The model is trained for 20 epochs. During each epoch, the model's performance is evaluated using the Dice Score and IoU metrics on the validation set and if you want you can the change the epochs. but make sure not get in overfitting situation. 

Saving the Model: 
The trained model is saved as u2net_segmentation.pth.

Task 2: Retraining with DiceLoss:-

Data Preparation: 
The dataset remains the same as in Task 1, with training and validation sets split accordingly and augmented to improve model performance.

Custom Loss Function:
In this task, the model is retrained using the DiceLoss function. This loss function is particularly effective for tasks with imbalanced datasets or where the foreground is of primary interest.

Training Process: 
The training setup is similar to Task 1, but with DiceLoss replacing BCEWithLogitsLoss. The model is trained for 20 epochs, and performance is evaluated using the Dice Score and IoU metrics. feel free to change epochs if you want

Saving the Model: 
The retrained model is saved as u2net_segmentation_dice_loss.pth.

Evaluation Metrics:

The performance of the trained models is evaluated using the following metrics:

Dice Score: Measures the overlap between predicted and ground truth masks.
Intersection over Union (IoU): Evaluates the ratio of the intersection of predicted and ground truth masks to their union.

Usage:

To use the provided code, follow these steps:
* Ensure you have all the required dependencies installed.
* Place your dataset in the Image/ and Mask/ directories.
* Execute the training scripts for either Task 1 or Task 2.
* Check the u2net_segmentation.pth or u2net_segmentation_dice_loss.pth files for the trained model weights.
  
Contributing:

If you would like to contribute to this project, please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue to discuss what you would like to change.

Acknowledgments:

The U-2-Net model is a powerful tool for image segmentation and has been adapted for this project from the U-2-Net GitHub repository.
Thanks to the creators of the U-2-Net model and the contributors to the libraries used in this project.
