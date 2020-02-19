# DeepFake-Kaggle
DeepFake Kaggle competition

# Introduction
The main focus of DeepFakes is **face manipulation** at this point (early 2020). There are four types of face manipulation:

- **entire face synthesis:** creates entire non-existent faces. **StyleGAN and ProGAN-based** approaches are the most known method for creating these fakes.
- **face identity swap (DeepFake):** consists of replacing the face of one person with face of another person.
- **facial attributes manipulation:** consists of modifying some attributes of the face such as the color of the hair or the skin, the gender, the age, adding glasses, etc. **STARGAN** is one of most known approches for creating these fakes.
- **facial expression manipulation:** consists of modifying the facial expression of the person. Popular technique here is **Face2Face**.

For each of these types, there are works about how fakes are created, and works on how to detect the fake ones. Refer to [this paper](./docs/DeepFakes_and_Beyond_A_Survey.pdf) for a comprehensive review of all these works. Based on the survey's results, Stehouwer _et al._ (2019) [paper](./docs/On_the_Detection_of_Digital_Face_Manipulation-CNN_Attention.pdf) have superior performance in all 4 areas. The authors proposed to use attention mechanisms to process and improve the feature maps of CNN models. Read [this blog post](https://towardsdatascience.com/paper-summary-on-the-detection-of-digital-face-manipulation-5127e184f81) for an overview of their approach.

## Stehouwer's paper summary notes:

- The objective is to determine not only if the input image is real or fake but also to localize the part of the image that was manipulated.
- The benefit of localization of the fake part of the image has the side benefit of helping in the explainability of the algorithm and thereby (potentially) understanding the type, magnitude and the intention of the manipulation.
- Prior approaches of detection has been done using mostly two approaches: **image segmentation** and **repeated application of binary classifiers that make use of sliding windows.** Both of these types mostly rely on the use of _multi-task learning_ that requires additional supervision, and, therefore, do not directly improve the final image classification performance.
- The paper makes solid case for utilizing the **attention maps** to address the task of detection (classification) & localization.
- Read [this blog](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) for all things "attention".
- In simple words, the task comes down to finding the patches of the images that depict these abnormalities, and the very act of finding these patches lead to the localization property as well.
- The main contributions of the paper:
    - A novel attention layer to improve the classification performance and produce attention map indicating the manipulated face part.
    - ssss
- dfdf


























































