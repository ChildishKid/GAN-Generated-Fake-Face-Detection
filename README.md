# GAN-Generated Fake Face Detection      

## Objective
The objective of this project was to create an image classification network that is able to positively identify GAN-generated facial images. It’s no secret that there have been rapid advances in image synthesis, making it harder and harder to discern whether an image is real or fake. Luckily, creating an image classifier is relatively simple. There exist various frameworks (e.g. Keras) which allow anyone to train and implement any neural network architecture easily and right out of the box (low barrier of entry). The task becomes simpler when trying to detect generated images from one specific source or generator. One simply has to compile a corpus consisting of both real images and generated ones (from the specific architecture). However, since there is an ever increasing number of GAN architectures, the objective becomes more difficult: there needs to be higher emphasis on a model’s generalization ability across various sets of data. 

## Requirements
Note: The following packages were used on my personal environment, but by no means represent the minimum requirements. 

* Python 3.7
* Tensorflow 2.1 with GPU support (+ tensorflow-hub)
* Packages:
    * matplotlib
    * ipywidgets
    * scikit-learn
    * opencv
    * pillow
    * bentoml

## Datasets	
Each dataset is listed below:
1.	Gen1 (30K)
2.	Gen2 (4K)
3.	FFHQ_Orig (60K; 30K used for training)
4.	FFHQ_Wide (60K; 4K used for training)
5.	CelebA (100K)
6.	PGGAN [1] (18K)
7.	StyleGan (FFHQ) (9K)
8.	StyleGan (CelebA) (9K)

Training, validation and testing was performed using the first four datasets (although not always collectively). The difference between dataset 1 and 2 lies in the shot itself, both generated privately using StyleGan2. Dataset 2 includes wider shots as opposed to dataset 1. The same difference applies to set 3 and 4.

The dataset was divided as follows: 
* Training + Validation (80%)
    * Training (70%)
    * Validation (30%)
* Testing (20%)

Datasets 5-8 were obtained from the database Diverse Fake Face Dataset (DFFD), introduced by Michigan State University [2]. Asides from FFHQ_Orig and FFHQ_Wide, CelebA was the only other set of images utilized that encompassed real images.

## Models
Experiment was performed using two different models. Initially, Xception architecture was selected, simply because it was one of the higher-performing models available with the Keras API for a reasonable parameter size and because it was used as a benchmark throughout several other papers. 

However, due to resource limitations, training starting taking a lot longer than initially thought. Performance cap and slow climb lead to a change in architecture; ResNet50. While standard models are pre-trained with ImagetNet, I opted for using its BigTransfer (BiT) [3] counterpart (pre-trained with ImageNet21K). Weights pre-trained on ImageNet-21K are trained using 14M (potentially multiple labels) images and 21K classes while weights pre-trained on ImageNet are trained using 1.28M (single label) images and 1K classes. Models are essentially scaled up during pre-trained using larger datasets which provide significant benefits. While not directly stated, I am under the assumption that this will apply universally and for a binary facial classification output. 

**Note: Saved models will not be made available to this git project.**  

## Experimentation
Note: No graphs are included on the account of me completely forgetting to save them.

### Parameters
Training was performed with the following augmentations and parameters:
* Rotation Range = 45
* Vertical and Horizontal Shift = 0.15
* Horizon Flip = True
* Zoom Range = 0.5
* Optimizer (Xception) = Adam (Learning Rate: 1-e2 to 1e-4)
* Optimizer (ResNet50) = SGD (Learning Rate: 3e-3 to 3e-7)

Training and classification was initially not performed on wider shots (datasets 2 and 4).  The neural networks listed below were trained, validated and tested against datasets 1 and 3. With that in mind, the following tables outline the training results for the first few runs:

**Table 1:** Model Training

| **Models** | **Input** | **Accuracy (Validation)** | **Accuracy (Test)** | **Epoch** |
| --- | :-: | :-: | :-: | :-: |
| Xception\_base | 1024 x 1024 | ~88% | 87.62% | 50 |
| Xception\_base\_small | 299 x 299 | ~75% | 75.42% | 50 |
| **ResNet50\_base** | **1024 x 1024** | **~97%** | **98.01%** | **35** |
| ResNet50\_data\_aug | 1024 x 1024 | ~97% | 97.63% | 38 |
| ResNet50\_base\_small | 299 x 299 | ~80% | 80.31% | 30 |
| ResNet50\_data\_aug\_small | 299 x 299 | ~80% | 79.76% | 42 |

**Table 2:** Predictions using Xception_base

| | **299x299** | **1024x1024** |
| --- | :-: | :-: |
| **CelebA** | 38.52% | 72.16% |
| **PGGAN** | 35.97% | 98.42% |
| **StyleGan (FFHQ)** | 11.74% | 67.99% |
| **StyleGan (CelebA)** | 52.63% | 93.70% |

**Table 3:** Predictions using ResNet50_base

| | **299x299** | **1024x1024** | **1024x1024 (Data Aug.)** | **299x299 (Data Aug.)** |
| --- | :-: | :-: | :-: | :-: |
| **CelebA** | 99.07% | **99.96%** | 99.55% | 95.40% |
| **PGGAN** | 8.35% | **94.49%** | 92.76% | 17.52% |
| **StyleGan (FFHQ)** | 3.72% | **83.57%** | 75.86% | 13.23% |
| **StyleGan (CelebA)** | 25.63% | **94.58%** | 91.48% | 25.63% |


### Input
As one would expect, precision tends to increase with higher resolutions (higher number of inputs in CNN). Even when running classifications against several other datasets, models with lower input sizes showed disastrous performance (as shown above). The same settings, parameters and images were used for both input types but the huge difference in performance may be due to some error that was made along the way. Otherwise, this leads me to believe that a higher resolution results in a better ability to generalize. 

### Xception Neural Network
The Xception model had difficulty reaching its peak accuracy (slow climb) and performed noticeably better for higher learning rates. In the end, it was run at 1e-2, which surprisingly yielded a stable build and with little fluctuations. 

Further training was done by fine-tuning the Xception_base and Xception_base_small model by unfreezing the pre-trained layers but that resulted in issues. The models would not learn and accuracy would be stuck at a constant 50% which I realized was most likely due to the batch normalization layers. Even after adjusting them (setting each batch normalization layer’s trainable configuration to false) yielded the same results.

In terms of its generalization ability, the Xception model performed decently (on higher inputs only). Although it remains to be tested how it would fair when trained with an SGD optimizer setting since Adam tends to generalize worse than SGD.

### ResNet50 Neural Network
The ResNet50 neural network showed considerable improvements over the Xception neural network when trained using the same dataset. The model reached its peak and stabilized in only a few epochs. It’s ability to generalize is also shown to be quite impressive, yielding an average prediction accuracy of 93.15% (excluding wider facial images). 

### Data Augmentations
A theory, approached by Wang et al., on improving a neural network’s ability to generalize when dealing with detection of face synthesis and manipulation was shortly tested. In summary, the paper describes the possibility of creating a universal detector for facial images (discerning fake and real images from a batch of samples) through data augmentation. While there were many different scenarios, the augmentation that offered a good balance involved converting images to JPEG and blurring them, with a certain probability for the augmentation to take place. 

Is it worth noting that the paper performed its experiments using a dataset whose resolution was set to 256x256. Although the average prediction accuracy did not increase for our higher input model (1024x1024), they did so for the smaller input model (299x299) across other datasets, increasing from 34.19% to 37.95%.

Since training involved other data augmentations, I was curious to sample how well the model would fair against specific augmentations during predictions. Due to the lack of time however, only rotation augmentation was tested. The following table outlines the accuracy results (Note: ResNet50_base was used as model since it yielded the highest overall accuracy):

**Table 4:** Accuracy with Rotation Augmentation

| | **Regular** | **Rotation**  **Range: 15** | **Rotation**  **Range: 30** | **Rotation**  **Range: 45** | **Rotation: 15** |
| --- | :-: | :-: | :-: | :-: | :-: |
| **Testing Dataset** | 98.01% | **98.80%** | 98.83% | 98.78% | 97.28% |
| **CelebA** | 99.96% | **97.54%** | 92,80% | 94.64% | 94.86% |
| **PGGAN** | 94.49% | **96.84%** | 96.16% | 94.35% | 96.38% |
| **StyleGan (FFHQ)** | 83.57% | **94.24%** | 95.21% | 94.33% | 88.73% |
| **StyleGan (CelebA)** | 94.58% | **96,86%** | 96.69% | 95.19% | 94.50% |
| **Average Prediction %** | 94.12% | **96.86%** | 96.72% | 95.46% | 94.35% |

Overall, applying rotation to each dataset before classification showed an improvement in predictions, which makes sense as rotation was applied as data augmentation during training. Applying a rotation range of 15 degrees showed the most overall improvements in accuracy, improving a model’s generalization ability in detecting fake/real facial images.

### Wider Images
Although at this stage we had relatively decent results, when testing our trained ResNet50 model against datasets encompassing wider sets of images, results were disappointing: 9.45% for Dataset 2 and 0.84% for Dataset 4.
Two different versions of our base ResNet50 were created. Version 2 was trained purely using wider facial images (Dataset 2 and 4) while version 3 was trained using a combination of all four initial datasets. The following tables outlines their training results and accuracies from each dataset:

**Table 5:** ResNet50 Model Training

| **Models** | **Input** | **Accuracy (Validation)** | **Accuracy (Test)** | **Epoch** |
| --- | :-: | :-: | :-: | :-: |
| ResNet50\_base-v2 | 1024 x 1024 | ~99% | 99.81% | 28 |
| ResNet50\_base-v3 | 1024 x 1024 | ~98% | 96.58% | 50 |

**Table 6:** Predictions on Version 2 and 3

|| **ResNet50\_base-v2** | **ResNet50\_base-v3** |
| --- | :-: | :-: |
| **Gen1** | 6.98% | 89.47% |
| **Gen2** | 99.88% | 98.15% |
| **FFHQ\_Orig** | 98.92% | 99.70% |
| **FFHQ\_Wide** | 99.69% | 99.85% |
| **CelebA** | 88.90% | 64.44% |
| **PGGAN** | 75.56% | 97.34% |
| **StyleGan (FFHQ)** | 37.30% | 72.46% |
| **StyleGan (CelebA)** | 32.94% | 93.02% |
| **Average Prediction %** | 67.52% | 89.31% |

Unsurprisingly, as was the case for our original ResNet50_base model, version 2 performed poorly in detecting generated close-up facial images. However, it yielded good accuracy in detecting real images from the rest. 
On the other hand, version 3 yielded very good accuracy throughout every dataset (except CelebA). By allowing the model to train with both variations of images, it allowed it to improve its ability to generalize on unforeseen datasets. 

### Detection via Projection
One of the last theories I got to test out was seeing if it was possible to create a detection method using StyleGan2's projection ability [5]. 
I took serveral small subset of each dataset, compiled them into TFRecords (accepted inputs for SG2), and allowed SG2 to run projections on them. Ideally, as per the theory suggested in the paper, if an image has been generated, then it should be able to generated something similar.  

However, that didn't seem to work as intended. It would seem that the default configuration is trained using close-up images. When feeding images from dataset 2, it failed to generate any considerable faces (visually). For the other datasets, it more or less worked as intended (failing to repdroduce any feasible images for inputs from CelebA or FFHQ_Orig but generating believable facial images fror inputs from dataset 1).

Visually comparing each set of images would have been too time-consuming, so an idea was proposed: determine each pair's (input image along side projected image) FID score. Doing so will allow us to estimate a cut-off score at which point one could use to classify a subset of images. However, that also seemed to have failed as values did not corelate between datasets. 


## Directory
### models
Jupyter notebooks for assembling and training different models. Currently holds ResNet50 (recommended), XceptionNet and VGG16 (tutorial sample).  
Version 2 and 3 of ResNet50 are run from same notebook.

### notebooks
Jupyter notebooks for prediction and classification tasks (implemented using weights).  
Seperate notebooks have been created for both ResNet50 and Xception (different configuration setup).   
Classification notebooks is meant to serve for end-users (grabbing images from set directory to sort after running predictions based on model used).  
Prediction notebooks is meant to run classification/predictions on specific dataset (selectable from drop-down menu).

### scripts
Python scripts for image transformation (resizing and rotation).

### saved_models
Configuration files for all three versions of ResNet50_base model.

### bentoML
BentoML service has been created and included for ResNet50.

## References
1. T. Karras, T. Aila, S. Laine, and J. Lehtinen, “Progressive Growing of GANs for Improved Quality, Stability, and Variation,” *International Conference on Learning Representations*, 2018.
2. Dang, Hao, et al. “On the Detection of Digital Face Manipulation.” *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020. 
3. Kolesnikov, Alexander et al. "Big Transfer (BiT): General Visual Representation Learning". *ArXiv.org*, 2020.
4. Wang, Sheng-Yu, et al. “CNN-Generated Images Are Surprisingly Easy to Spot… for Now.” *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
5. Karras, Tero, et al. “Analyzing and Improving the Image Quality of StyleGAN.” *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
