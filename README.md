# SL-Swin: A Transformer-Based Deep Learning Approach for Macro- and Micro-Expression Spotting on Small-Size Expression Datasets

## Performance

### MEGC 2022 Spotting Task

| Method | 3D-CNN | Swin-T | SL-Swin-T |
| --- | --- | --- | --- |
| p | --- | 0.60 | 0.60 |
| CAS_Test Precision | 0.4000 | 0.1521 | 0.1944 |
| CAS_Test Recall | 0.1111 | 0.1944 | 0.1944 |
| CAS_Test F1-score | 0.1739 | 0.1707 | 0.1944 |
| SAMM_Test Precision | 0.0845 | 0.0638 | 0.0689 |
| SAMM_Test Recall | 0.1935 | 0.0967 | 0.1290 |
| SAMM_Test F1-score | 0.1176 | 0.0769 | 0.0898 |
| Overall Precision | 0.1235 | 0.1075 | 0.1170 |
| Overall Recall | 0.1493 | 0.1492 | 0.1641 |
| Overall F1-score | 0.1351 | 0.1250 | 0.1366 |

### MEGC 2021 Spotting Task

| Method | 3D-CNN | SL-Swin-T |
| --- | --- | --- |
| p | --- | 0.60 |
| CAS(ME)^2 MaE | 0.2145 | 0.2236 |
| CAS(ME)^2 ME | 0.0714 | 0.0879 |
| CAS(ME)^2 Overall | 0.1675 | 0.1824 |
| SAMM_longvideos MaE | 0.1595 | 0.1675 |
| SAMM_longvideos ME | 0.04665 | 0.1044 |
| SAMM_longvideos Overall | 0.1084 | 0.1357 |

## Research Articles

### Peer-Reviewed Article (Recommended)

> He, E.; Chen, Q.; Zhong, Q. SL-Swin: A Transformer-Based Deep Learning Approach for Macro- and Micro-Expression Spotting on Small-Size Expression Datasets. Electronics 2023, 12, 2656. <https://doi.org/10.3390/electronics12122656>

### Preprint Article

> He, E.; Chen, Q.; Zhong, Q. SL-Swin: A Transformer-Based Deep Learning Approach for Macro- and Micro-Expression Spotting on Small-Size Expression Datasets. Preprints.org 2023, 2023060079. <https://doi.org/10.20944/preprints202306.0079.v2>

### Differences

1. In the peer-reviewed article, the present continuous tense is revised to the past continuous tense.
2. In the peer-reviewed article, the structure of the section Performance and section Discussion is revised.
3. In the peer-reviewed article, the word "pre-process" is revised to "preprocess".
4. In the peer-reviewed article, the phrase "in the task" is revised to "on the task".

### WeChat Article in Chinese

[华南师范大学：一种在小数据量的表情数据集上基于Transformer的表情检测方法 | MDPI Electronics](https://mp.weixin.qq.com/s/h1dyEMz9fG7a4Ynos5R8Og)

---

## Baseline Articles

### ACM Article

> Chuin Hong Yap, Moi Hoon Yap, Adrian Davison, Connah Kendrick, Jingting Li, Su-Jing Wang, and Ryan Cunningham. 2022. 3D-CNN for Facial Micro- and Macro-expression Spotting on Long Video Sequences using Temporal Oriented Reference Frame. In Proceedings of the 30th ACM International Conference on Multimedia (MM '22). Association for Computing Machinery, New York, NY, USA, 7016–7020. <https://doi.org/10.1145/3503161.3551570>

### arXiv Article

> Yap, C.H.; Yap, M.H.; Davison, A.K.; Kendrick, C.; Li, J.; Wang, S.; Cunningham, R. 3D-CNN for Facial Micro- and Macro-Expression Spotting on Long Video Sequences Using Temporal Oriented Reference Frame. arXiv e-prints 2021, arXiv:2105.06340, doi:<https://doi.org/10.48550/arXiv.2105.06340>.

---

## Related Repositories

### TensorFlow Code

<https://github.com/eddiehe99/tensorflow-expression-spotting>

### dlib-whl

You could find wheel package files of dlib for python of different versions on Windows_x64 at <https://github.com/eddiehe99/dlib-whl>.

### Mean Average Precision for Evaluation (Official)

<https://github.com/bes-dev/mean_average_precision>

### Survey about Articles and Codes (Official)

<https://github.com/pakchoi-php/halo>

---

## Acknowledgement

### SOFTNet (Official)

Deep appreciation to Liong et al. for sharing their code at <https://github.com/genbing99/SoftNet-SpotME>.

### Vision Transformer for Small-Size Datasets (Official)

<https://github.com/aanna0701/SPT_LSA_ViT>

### Tutorials (in Chinese)

Deep appreciation to [WeZhe](https://github.com/WZMIAOMIAO) for his tutorials about deep learning for image processing and his code at <https://github.com/WZMIAOMIAO/deep-learning-for-image-processing>.

---

## PS

The code is formatted by `black`.

---

## Contact

Please email me at <2021022249@m.scnu.edu.cn> if you have any inquiries or issues.
