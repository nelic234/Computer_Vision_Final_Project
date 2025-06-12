# Computer Vision Final Project

### Authors: Neli Čatar, Gellert Toth, and Aimee Lin

Welcome to our final project for the *Computer Vision* course. This repository contains the code and documentation for our implementation of Topic 6: **Truth in Motion: Depth and Flow Enhanced DeepFake Detection**.

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
* [Credits](#credits)
* [License](#license)
* [Badges](#badges)
* [References](#references)

---

## Background

DeepFake detection is a critical task in modern multimedia forensics. In this project, we explored enhanced detection methods using motion (optical flow) and spatial (depth map) cues, drawing on recent advances in both traditional computer vision and deep learning models such as Vision Transformers.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/cv-final-project.git
cd cv-final-project
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Ensure you have access to the FaceForensics++ dataset and set up the appropriate paths in the notebook.
2. Run the pipeline using Jupyter:

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

1. Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). *Faceforensics++: Learning to detect manipulated facial images*. arXiv preprint [arXiv:1901.08971](https://arxiv.org/abs/1901.08971)

2. Amerini, I., Galteri, L., Caldelli, R., & Del Bimbo, A. (2019). *Deepfake Video Detection through Optical Flow Based CNN*. In *2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)* (pp. 1205–1207). [DOI:10.1109/ICCVW.2019.00152](https://doi.org/10.1109/ICCVW.2019.00152)

3. Nassif, A. B., Nasir, Q., Talib, M. A., & Gouda, O. M. (2022). *Improved Optical Flow Estimation Method for Deepfake Videos*. *Sensors*, 22(7), 2500. [DOI:10.3390/s22072500](https://doi.org/10.3390/s22072500)

---
