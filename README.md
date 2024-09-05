# Noise2Clean: Denoising MNIST with VAE and AE

## Dataset Description

- **Dataset Used:** [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
  - **Description:** The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. This dataset is commonly used for image classification and has been widely adopted for benchmarking machine learning algorithms.
  - **Noise Injection:** Gaussian noise was added to the original images to create a noisy version. The task is to reconstruct the clean images from these noisy inputs.

## Architectural Details

Both the Autoencoder (AE) and Variational Autoencoder (VAE) models use 4 residual blocks (ResBlocks) with the following architecture:

- **ResBlock Composition:**
  - Conv2D
  - BatchNorm
  - ReLU activation
  - Conv2D
  - BatchNorm
  - Residual addition followed by ReLU activation

### Forward Methods:

- **Autoencoder (AE):**
  - Utilizes a standard forward pass through the encoder and decoder to map noisy images to clean images.

- **Variational Autoencoder (VAE):**
  - Implements the reparametrization trick during the forward pass to sample from a latent space distribution, allowing for generation of new similar images.

## Loss Function and Optimizer

- **Autoencoder (AE) Loss Function:**
  - **Mean Squared Error (MSE) Loss:** Measures the difference between the predicted and actual pixel values.
  
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]
  
- **Variational Autoencoder (VAE) Loss Function:**
  - **MSE Loss** + **Kullback-Leibler Divergence (KLD):** Combines MSE for reconstruction loss with KLD to measure how closely the learned latent space distribution matches a standard normal distribution.
  
  \[
  \text{VAE Loss} = \text{MSE} + \text{KL Divergence}
  \]

- **Optimizer:** 
  - **Adam Optimizer** with a learning rate of 0.001.

## Evaluation Metric

### Structural Similarity Index Measure (SSIM)

The **SSIM** evaluates the quality of denoised images by comparing them to the ground truth images. It considers luminance, contrast, and structure to provide a measure of similarity.

The SSIM between two images \(x\) and \(y\) is given by:

\[
\text{SSIM}(x, y) = \frac{(2 \mu_x \mu_y + C_1)(2 \sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
\]

Where:
- \(\mu_x\) and \(\mu_y\) are the mean values of images \(x\) and \(y\),
- \(\sigma_x^2\) and \(\sigma_y^2\) are the variances,
- \(\sigma_{xy}\) is the covariance,
- \(C_1\) and \(C_2\) are constants for stability.

## Results

The table below summarizes the SSIM scores for both models on the test dataset:

| Metric              | Autoencoder (AE) | Variational Autoencoder (VAE) |
|---------------------|------------------|-------------------------------|
| **SSIM Score**      | 0.76             | 0.87                          |

## Hyperparameters

The following hyperparameters were used for training the models:

| Hyperparameter      | Value  |
|---------------------|--------|
| **No. of ResBlocks** | 4      |
| **Batch Size**      | 64     |
| **Learning Rate**   | 0.001  |
| **Epochs**          | 50     |

## Contact Me

For any questions, feedback, or collaboration opportunities, please reach out:

- **Email:** ritwik21485@iiitd.ac.in
