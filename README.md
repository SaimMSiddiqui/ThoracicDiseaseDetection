# Thoraic Disease Detection

# 1. Project Overview
Thoracic diseases, including pneumonia, emphysema, and fibrosis, are significant global health concerns and are often challenging to diagnose accurately. Chest X-rays serve as an accessible and common diagnostic tool, but interpreting these images requires expertise and is complex, even for trained radiologists.

This project proposes a machine learning-based solution to automate the detection of thoracic diseases from chest X-rays, creating a Computer-Aided Diagnostic (CAD) tool. By leveraging models like logistic regression and Support Vector Machines (SVM), the CAD system can aid radiologists in identifying thoracic diseases more efficiently and accurately, improving the likelihood of timely and precise diagnoses.

# 2. Dataset
The **NIH ChestX-ray8 dataset** is used as the primary data source. This large-scale, open-source collection includes:

- **108,948 frontal-view chest X-rays** from **32,717 patients**
- **14 disease classes**: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia
- **Multi-label capability**: Each image may have multiple disease labels
- **Image resolution**: 1024x1024 in PNG format
- **Bounding box annotations** for localization (~1,000 images)

For the initial phase of this project, only a small percentage of the original dataset will be used (~5000) and can be found here: https://www.kaggle.com/datasets/nih-chest-xrays/sample?resource=download.

## Dataset Challenges
1. **Multi-label classification**: Each image may be labeled with one or more diseases.
2. **Class imbalance**: Some conditions are significantly underrepresented, which requires techniques like data augmentation and weighted loss functions to ensure balanced learning across all disease classes.

# 3. Evaluation Methodology
Given the multi-label nature of the problem, we will use a mix of metrics to evaluate model performance on both accuracy and handling multiple overlapping labels:

- **Primary Metric**:
  - **Accuracy**: Measures overall correct classification rate.
- **Secondary Metrics**:
  - **Precision, Recall, F1 Score**: Evaluate performance on imbalanced classes, ensuring sensitivity and specificity.
  - **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Aggregates model performance across all classes and is particularly useful for multi-label classification.
  - **Hamming Loss**: Measures the fraction of labels incorrectly predicted, well-suited to multi-label classification.

 ## Data Splitting and Cross-Validation
The dataset will be split into training (60%), validation (20%), and test (20%) sets, with measures in place to prevent any patient data overlap between sets. Cross-validation will be used to refine performance and reduce overfitting risks.

# 4. License
This project is licensed under the MIT License.

## References
1. Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. Proceedings of the 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3462-3471.
