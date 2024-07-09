# ViT-CIFAR10
This repo contains files , model and code . The ViT is trained on CIFAR-10 dataset and it is capable of getting 99.5% train accuracy and 93% test accuracy.

# Vision Transformer (ViT) for CIFAR-10 Classification

This repository contains an implementation of a Vision Transformer (ViT) model trained on the CIFAR-10 dataset. The project demonstrates the application of transformer architecture, typically used in natural language processing, to image classification tasks.

## Repository Contents

1. `ViT_cifar10_train.ipynb`: Jupyter notebook containing the code for model definition, training, and evaluation.
2. `ViT_cifar10_test.ipynb`: Jupyter notebook for loading the trained model and performing inference on test data.
3. `model_checkpoint_epoch_350_local.pt`: Saved checkpoint of the trained model after 350 epochs.

## Project Overview

### Motivation

Initially, a traditional convolutional neural network (CNN) approach was attempted for CIFAR-10 classification. However, the CNN model struggled to achieve accuracy beyond 90%. To overcome this limitation, we implemented a Vision Transformer, which has shown promising results in various image classification tasks.

### Model Architecture

The Vision Transformer (ViT) model used in this project is based on the original ViT paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". Key components include:

1. Patch Embedding: Divides the input image into fixed-size patches and linearly embeds them.
2. Positional Embedding: Adds positional information to the patch embeddings.
3. Transformer Encoder: A stack of transformer blocks, each containing multi-head self-attention and feed-forward layers.
4. Classification Head: A linear layer for final classification.

### Training Process

The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. The training process includes:

- Data augmentation techniques like random cropping and horizontal flipping.
- Learning rate scheduling with warmup and cosine annealing.
- Regularization techniques such as weight decay and dropout.
- Training for 350 epochs to ensure convergence.

### Key Features

- Implementation of Vision Transformer architecture for image classification.
- Use of PyTorch for model definition and training.
- Data augmentation and normalization for improved generalization.
- Learning rate scheduling and regularization techniques for optimal training.
- Checkpointing to save and load model states.

## Usage

### Training

To train the model:

1. Open `ViT_cifar10_train.ipynb` in a Jupyter environment.
2. Run all cells to train the model.
3. The trained model will be saved as a checkpoint file.

### Testing

To test the trained model:

1. Open `ViT_cifar10_test.ipynb` in a Jupyter environment.
2. Ensure `model_checkpoint_epoch_350_local.pt` is in the same directory.
3. Run all cells to load the model and perform inference on test data.
4. The notebook will display test accuracy and visualize predictions on sample images.

## Results

The Vision Transformer model achieves significantly higher accuracy on the CIFAR-10 dataset compared to the initial CNN approach, demonstrating the effectiveness of the transformer architecture in image classification tasks.

## Dependencies

- PyTorch
- torchvision
- matplotlib
- numpy

## Future Work

- Experiment with different hyperparameters and model sizes.
- Apply the model to other image classification datasets.
- Implement more advanced data augmentation techniques.
- Explore hybrid architectures combining CNNs and transformers.

## References

- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929.
