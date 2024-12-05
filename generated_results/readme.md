**GRAPHS/.CSVs AND THEIR EXPLANATIONS:**

**svm:**

confusion_matrix: heatmap representing results when predicting whether an x-ray is normal or abnormal without any normalization, accuracy of 75%

confusion_matrix2: with normalization, accuracy of 73%

disease_distr: distribution of specific diseases among the 810 examples

normal_vs_abnormal: distribution of normal and abnormal lung x-rays among the 810 examples


**cnn:**
abnormal_vs_normal.csv: output of test examples (normal vs abnormal) using cnn2.py (in src folder)
.png: plot of training loss & accuracy for multi-class classification (normal vs abnormal)

multiple_labels.csv: output of test examples (actual vs predicted label) using cnn3.py
.png: plot of training loss & accuracy for multi-label classification (all diseases)

