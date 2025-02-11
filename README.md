# Image Classification with CNNs and Vision Transformers
This project demonstrates image classification on a dataset using two state-of-the-art deep learning architectures: Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). It explores the effectiveness of both approaches in image classification tasks, focusing on performance metrics like accuracy and loss.

## Features
#### Convolutional Neural Networks (CNNs):
- Designed a custom CNN architecture with skip connections, L1/L2 regularization, and optimized hyperparameters.
- Evaluated performance on standard datasets, achieving high classification accuracy.

#### Vision Transformers (ViTs):
- Implemented a Vision Transformer model for image classification.
- Leveraged transformer-based attention mechanisms for enhanced feature extraction.

#### Performance Comparison:
- Compared the accuracy and efficiency of CNNs and ViTs for the same dataset.
- Visualized training and validation metrics to highlight strengths and limitations of each model.

## Getting Started
#### Prerequisites
- Python 3.8 or higher
- Essential Python libraries: NumPy, Pandas, Matplotlib, PyTorch, TensorFlow, and HuggingFace Transformers.
Install the required packages using: pip install -r requirements.txt

#### Dataset
- The project uses the CIFAR-10 dataset for training and testing. CIFAR-10 is a standard dataset consisting of 60,000 32x32 color images in 10 classes.
- The dataset is automatically downloaded using PyTorch or TensorFlow's built-in loaders.

## How to Run
1. Clone this repository: git clone https://github.com/shalini363/image_classification_using_cnn_vits.git

2. Navigate to the project directory: cd image-classification-cnn-vits

3. Launch the Jupyter Notebook:
jupyter notebook Deep_Learning_Project.ipynb

4. Follow the notebook cells to:
- Train the CNN and ViT models.
- Evaluate performance on the test set.
- Visualize results.

## Results
#### CNN Performance:
Achieved an accuracy of 89% on the test set.

#### ViT Performance:
Achieved an accuracy of 76% on the test set.

Comparison plots for accuracy and loss across both architectures are available in the notebook.

## Analysis
- Accuracy: 
ResNet-18 outperformed ViTs, which had an accuracy of 76.48%, by achieving 89.00%. This demonstrates how successfully CNN generalizes on small datasets, such as CIFAR-10.
- Precision:
ResNet-18 predicted fewer false positives than ViTs, with a little greater precision (89.15%) than ViTs (77.20%). 
- Recall: 
ResNet-18 proved its efficacy in accurately detecting true positives across all classes by achieving a higher recall (88.90%) than ViTs (76.85%).
- F1 Score:
ResNet-18's better performance (89.02%) over ViTs (77.02%) was further confirmed by the F1 Score, which strikes a balance between precision and recall.
- ROC AUC:
ResNet-18's superior discriminatory capacity was demonstrated by its better Receiver Operating Characteristic Area Under Curve (ROC AUC) of 0.97. However, ViTs struggled with the smaller dataset of CIFAR-10, as evidenced by its lower ROC AUC of 0.85.

## Key Learnings
- CNNs excel at extracting local patterns in images, making them highly efficient for spatial data.
- ViTs use self-attention mechanisms for global context understanding, performing well on complex datasets.
- The choice between CNNs and ViTs depends on the dataset and computational resources.

## Future Work
- Experiment with hybrid architectures combining CNNs and transformers.
- Test on larger and more complex datasets.
- Optimize ViT performance for smaller datasets.
