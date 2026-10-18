# Attribution

This project builds on the work of others. Thank you to everyone below.

## Data

**PlantVillage** - the original tomato leaf images.
- Hughes, D. P. and Salathe, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
- Mohanty, S. P., Hughes, D. P. and Salathe, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science.
- License: Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0), as stated by the PlantVillage authors. The dataset produced by this repository is a derivative and is released under the same license.
- The images used here were obtained from the Kaggle copy `abdallahalidev/plantvillage-dataset`.

## Models and libraries

| Component | Used for | License | Source |
|---|---|---|---|
| U-2-Net (Qin et al., 2020) | background removal | Apache-2.0 | https://github.com/xuebinqin/U-2-Net |
| Ultralytics YOLOv8 | leaf detection and cropping | AGPL-3.0 | https://github.com/ultralytics/ultralytics |
| torchvision ResNet50 (He et al., 2016), ImageNet weights | image features to find near-duplicates | BSD-3-Clause (code) | https://github.com/pytorch/vision |
| Roboflow | annotating the leaf images used to train the detector | - | https://roboflow.com |

- Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R. and Jagersand, M. (2020). *U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection.* Pattern Recognition, 106.
- He, K., Zhang, X., Ren, S. and Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.

## Notes

- This repository does not contain code or weights from Ultralytics or U-2-Net. You install `ultralytics` with pip and download the U-2-Net files yourself (see `models/README.md`). Their licenses apply to your use of them.
- The leaf detector (`best.pt`) is a YOLOv8n model fine-tuned on about 120 leaf images annotated in Roboflow. Because it is built with Ultralytics YOLOv8, check the AGPL-3.0 terms before redistributing the weights.
