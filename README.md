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

DeepFake detection is a critical task in modern multimedia forensics. In this project, we aimed to further previous research that used optical flow based and depth based deepfake detection by finetuning a pretrained CNN vision model. Drawing on recent advances we switched out the underlying backbone model from a CNN to a Vision Transformer. We trained ViT and Swin transformers using both methods and compared the results. Finally, we also combined the two and trained the third and final model.

First, videos are broken into frames and we extract faces from each frame using MTCNN. The pictures are then cropped to the required size by downstream tasks.   

Optical flow describes how each pixel moves between two (consecutive) frames. This flow data is then turned into an RGB picture by converting the flow vectors first into HSV then into RGB. By leveraging pretrained transformer models such as Vit and Swin, we only require a limited amount of training data in order to achieve competitive performace. We used the FaceForensics++ dataset for finetuning and evaluating our model. 

The depth at each pixel represents how far is the point in space from the camera at that pixel. We estimed depth using Depth Anything V2. To turn this into the required RGB format, we copied the same input across all 3 color channels. From here on the process is the same: finetune a transformer model on the FaceForensics++ datasets and evaluate on set aside data.

The selected transformers models are all relatively expensive to be run for inference at large scale. The most likely and pressing usecase of such a model would be to automatically flag deepfake videos on social media platforms such as Instragram, Facebook or Tiktok, in order to stop the spreading of fake news and information. These platforms have a staggering amount of videos that would have to be processed meaning that the expensivity of the model is relevant. We tried to explore various compression techniques and evalute the performance loss vs efficiency gain trade off. We explored compression techniques such as quantiziation and distillation.  

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/cv-final-project.git
cd cv-final-project
```

Install the required dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 % we have to create this
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

## License

This project is licensed under the ...

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

[6] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth Anything V2,” in Advances in Neural Information Processing Systems, vol. 37, NeurIPS 2024, doi: 10.48550/arXiv.2406.09414.

[7] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021. doi: 10.48550/arXiv.2010.11929

[8] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 10012–10022, doi: 10.1109/ICCV48922.2021.00986

[9] G. Bradski and A. Kaehler, Learning OpenCV: Computer Vision with the OpenCV Library, 1st ed. Sebastopol, CA, USA: O'Reilly Media, 2008.

[10] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks," IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499–1503, Oct. 2016, doi: 10.1109/LSP.2016.2603342.


---
