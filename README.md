# Computer Vision Final Project

### Authors: Neli Čatar, Gellert Toth, and Aimee Lin

This repository contains our final project for the *Computer Vision* course. The chosen topic was Topic 6: **Truth in Motion: Depth and Flow Enhanced DeepFake Detection**.

We have developed a full pipeline, showcased in [`project.ipynb`](project.ipynb). Note that the dataset used for this project is [FaceForensics++](https://github.com/ondyari/FaceForensics), which is subject to license restrictions. As a result, dataset loading and access are not included in this repository.

---

## Project Overview

Our investigation includes the following components:

* **Preprocessing**: We applied face detection mechanisms to preprocess video frames.
* **Feature Extraction**: Implemented both optical flow and depth-based techniques.
* **Training**: Conducted experiments training on optical flow, depth maps, and a combination of both. We used various Visual Transformer models.
* **Model Compression**: Techniques for reducing model size and improving inference time were explored.

---

## Table of Contents

* [Background](#background)
* [Installation](#installation)
* [Usage](#usage)
* [Credits](#credits)fi
* [License](#license)
* [Badges](#badges)
* [References](#references)

---

## Background

DeepFake detection is a critical task in modern multimedia forensics. Existing research used optical flow based and depth based deepfake detection by finetuning a pretrained CNN vision model. We aimed to try these already existing methods, with the small modification of using pretrained vision transformers as the backbone instead. We finetuned the Dino-v2 (base) transformer using both methods and compared the results. We also also aim to further existing research by exploring compression techniques.

### Data preprocessing

First, videos are broken into frames and we extract faces from each frame using MTCNN. The pictures are then cropped to the required size by downstream tasks and saved on disk.  

### Optical Flow

Optical flow describes how each pixel moves between two (consecutive) frames. This flow data is then turned into an RGB picture by converting the flow vectors first into HSV then into RGB. By leveraging pretrained transformer models such as dino-v2, we only require a limited amount of training data in order to achieve competitive performace. We used the FaceForensics++ dataset for finetuning and evaluating our model. 

During training we freeze all but the last layer of the dino-v2 backbone and add an extra classification head that takes the embedding produced by the backbone as the input. The classification head is made up of two linear layers with a relu and a dropout inbetween.  

### Depth

The depth at each pixel represents how far is the point in space from the camera at that pixel. We estimed depth using Depth Anything V2. To turn this into the required RGB format, we copied the same input across all 3 color channels. From here on the process is the same: finetune a transformer model on the FaceForensics++ datasets and evaluate on set aside data.

### Compression

The selected transformers models are all relatively expensive to be run for inference at large scale. The most likely and pressing usecase of such a model would be to automatically flag deepfake videos on social media platforms such as Instragram, Facebook or Tiktok, in order to stop the spreading of fake news and information. These platforms have a staggering amount of videos that would have to be processed meaning that the expensivity of the model is relevant. We tried to explore various compression techniques and evalute the performance loss vs efficiency gain trade off. We explored compression techniques such as quantiziation and distillation.  

Quantization refers to using float 16 (or even int 8) instead of using float 32. Using float 16 is a lot less computationally heavy, but it is less accurate. Fortunately, torch implements a very effective autocast method. Autocast decided to use float 16 when it is safe (no chance for precision loss) and uses float 32 when it needs to. By using autocast we can achieve a considerable speedup in inference and training time, while supposedly not sacrifcing any accuracy. 

When distilling a larger model into a smaller (and faster) model, we aim to get the same performance with the small model as with the larger one, by teaching the small model the inner representations of the larger model. In our code this is done by comparing the embedding produced by the finetuned backbone. The difference between these two representations is measured by KL divergence and is added to the loss term. Now the loss combines both classification loss and representation divergence loss. Since the output embedding of the two models is potentially different (as is the case with Dino-v2 small and base) before comparison we first project the representation of the larger model into the correct size using a simple linear layer. We do not expect to achieve any real results here, since Dino-v2 small is already itself distilled from the base model before its realease. Nonetheless, we wanted to achieve at least a functioning distillation setup.

### Training

We struggled with overfitting quite a lot, especially in the case of flow data. Due to computational constraints, we only trained our model on 4 sources of vidoes found in the FaceForensics++ dataset: actors, youtube, DeepFakeDetection and Deepfakes. For each source, we selected only a 100 vidoes for a total of 400 videos. The data is split up into train, val and testing data with a 0.7, 0.2, 0.1 split. We do believe that the increased amount of data would have reduced overfitting, regardless we tried to reduce overfitting by applying dropout and by reducing the number of trainable encoder layers in the transformer. We also saved the best model according to the validation data for final evaluation on the testing data. 

### Results

We found that the depth based model perfomed better compared to the flow model, by achieving over 60% accuracy on testing data compared to 53%. But we did not manage to replicate the classification accuracies of existing research. 

Quantization using autocast did achieve an almost 3x speedup while maintaining the same classification accuracy.

Our distillation training did not finish in time, so these results cannot be included here. However, we believe that it would not achieve a considerable accuracy gain compared to simply training the dino-v2 small model my itself, since the dino-v2 small model based network already achieves the same performance as the base model (at a rate of 2x speedup as well). 

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/cv-final-project.git
cd cv-final-project
```

Install the required dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

1. Ensure you have access to the FaceForensics++ dataset and set up the appropriate paths in the notebook.
2. Run the pipeline using Jupyter notebook:

```bash
jupyter notebook project.ipynb
```

3. Follow the notebook cells to run preprocessing, feature extraction, training, and evaluation.

---

## Credits

* Course: Computer Vision (Spring 2025)
* Dataset: [FaceForensics++](https://github.com/ondyari/FaceForensics)
* Optical Flow & Depth Techniques inspired by recent literature (see [References](#references)).


---

## Badges

![Python](https://img.shields.io/badge/Python-3.8-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Transformers](https://img.shields.io/badge/Transformers-Used-success)

---

## References

[1] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, “FaceForensics++: Learning to detect manipulated facial images,” arXiv preprint arXiv:1901.08971, 2019.

[2] I. Amerini, L. Galteri, R. Caldelli, and A. Del Bimbo, “Deepfake video detection through optical flow based CNN,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW), Seoul, Korea (South), 2019, pp. 1205–1207, doi: 10.1109/ICCVW.2019.00152.

[3] A. B. Nassif, Q. Nasir, M. A. Talib, and O. M. Gouda, “Improved optical flow estimation method for deepfake videos,” Sensors, vol. 22, no. 7, p. 2500, Mar. 2022, doi: 10.3390/s22072500.

[4] L. Maiano, L. Papa, K. Vocaj, and I. Amerini, “DepthFake: A depth-based strategy for detecting Deepfake videos,” in Pattern Recognition, Computer Vision, and Image Processing. ICPR 2022 International Workshops and Challenges, Lecture Notes in Computer Science, vol. 13774, Springer, 2023, pp. 17–31, doi: 10.1007/978-3-031-37745-7_2.

[5] D. Sun, X. Yang, M.-Y. Liu, and J. Kautz, “PWC‑Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 8934–8943, doi: 10.1109/CVPR.2018.00931.

[6] S. Niklaus, A Reimplementation of PWC-Net Using PyTorch, 2018. [Online]. Available: https://github.com/sniklaus/pytorch-pwc

[7] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth Anything V2,” in Advances in Neural Information Processing Systems, vol. 37, NeurIPS 2024, doi: 10.48550/arXiv.2406.09414.

[8] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, et al., "DINOv2: Learning Robust Visual Features without Supervision," arXiv preprint arXiv:2304.07193, 2023. 

[9] G. Bradski and A. Kaehler, Learning OpenCV: Computer Vision with the OpenCV Library, 1st ed. Sebastopol, CA, USA: O'Reilly Media, 2008.

[10] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks," IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499–1503, Oct. 2016, doi: 10.1109/LSP.2016.2603342.


---
