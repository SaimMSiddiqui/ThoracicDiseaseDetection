# ChestX-ray14 Expert Labels

This data repository contains radiologist labels for a subset of the ChestX-ray14 
dataset. The labels were collected as part of two independent studies, and are
described in the following papers:

* [Chest Radiograph Interpretation with Deep Learning Models](https://pubs.rsna.org/doi/10.1148/radiol.2019191293)
* [Deep Learning for Distinguishing Normal versus Abnormal Chest Radiographs
and Generalization to Two Unseen Diseases Tuberculosis and COVID-19](https://arxiv.org/abs/2010.11375)

There are two sets of labels, each associated with one of the studies. The first
first set of labels is associated with [the study published in *Radiology*](https://pubs.rsna.org/doi/10.1148/radiol.2019191293) and focuses on 
four chest x-ray findings: airspace opacity, pneumothorax, nodule/mass, and
fracture. The second set of labels is associated with [the study published in *Scientific
Reports*](https://arxiv.org/abs/2010.11375) and includes all 14 findings released in the [original dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community), and a
normal/abnormal label.

The two sets of labels are located in separate directories 
(`four_findings_expert_labels` and `all_findings_expert_labels`)

Note: These labels are intended for research and educational purposes only.