# ChestX-ray14 Expert Labels: All Findings
 
This directory contains expert labels obtained as part of [*Scientific Reports* paper](https://arxiv.org/abs/2010.11375),


The set of labels focuses on all 14 findings released in the [original dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community), and as a normal/abnormal label. The set of
labels only contains images from the test set. These images are identical to the images included in the `Four
Findings Expert Labels` test split, restricted to chest x-rays with PA view (810 images out of the 1,962 images).

The same five American Board of Radiology certified radiologists independently reviewed each image. Each radiologist
was first asked whether the image contained any potentially actionable clinical finding (normal/abnormal label), and
if so, to select which of the 14 conditions were present. Information available at the time of radiologist review
included only patient age and image view (AP versus PA). Additional clinical information was not available.

The labels are in the directory `all_findings_expert_labels`. In `test_individual_readers.csv`, each row corresponds
to a single radiologist's labels for a single image. This means that each image ID and patient ID is repeated across
multiple rows (five rows per image, one row per reader). Each row also contains a reader ID so that the radiologists
can be distinguished. Because there are a total of 810 images in this set, `test_individual_readers.csv` contains
4,050 rows with 810 unique image IDs. `test_individual_readers.csv` also contains a total of 19 columns. In addition
to image ID, patient ID, and reader ID, there is a column for normal/abnormal, a column for each of the 14 findings,
and a column for `Other` indicating other abnormal findings are present (outside of the 14 specified). A cell value
of `YES` means "present" and `NO` means "absent".

`test_labels.csv` contains the ground truth labels used to evaluate the deep learning system in the [*Scientific Reports*
paper](https://arxiv.org/abs/2010.11375). Each row contains the ground truth labels for a single image ID, and each image ID only  appears in a single row,
for a total of 810 rows. `test_labels.csv` has the same columns as `test_individual_readers.csv`, but without a
"reader ID" column. To obtain these labels, three of the five radiologists who labeled this set were chosen at
random to be the "ground truth radiologists" (the other two were used as points of comparison). These "ground truth
radiologists" have reader IDs of "4343882785", "4343883593", and "4343883996". A majority vote was used to determine
the final label for the normal/abnormal label and the final label for each particular finding. The final label for the
`Other` column was determined to be `YES` if a majority of radiologists selected that a finding outside of the 14 was
present, or if a majority of radiologists indicated that the image was abnormal, but no single finding had a majority
of radiologists indicate was present.

When using these labels, include the following citation:

  > Zaid Nabulsi, Andrew Sellergren, Shahar Jamshy, Charles Lau, Eddie Santos,
  > Atilla P. Kiraly, Wenxing Ye, Jie Yang, Sahar Kazemzadeh, Jin Yu,
  > Raju Kalidindi, Mozziyar Etemadi, Florencia Garcia Vicente, David Melnick,
  > Greg S. Corrado, Lily Peng, Krish Eswaran, Daniel Tse, Neeral Beladia,
  > Yun Liu, Po-Hsuan Cameron Chen, Shravya Shetty, Deep Learning for
  > Distinguishing Normal versus Abnormal Chest Radiographs and Generalization
  > to Two Unseen Diseases Tuberculosis and COVID-19, Nature Scientific Reports,
  > 2021.