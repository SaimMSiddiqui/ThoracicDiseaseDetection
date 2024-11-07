# ChestX-ray14 Expert Labels: Four Findings
 
This directory contains expert labels obtained as part of [the study published in *Radiology*](https://pubs.rsna.org/doi/10.1148/radiol.2019191293)

The set of labels focuses on four findings (airspace opacity, pneumothorax, nodule/mass, and fracture)
and encompasses both validation and test sets. The final labels for each image were assigned via adjudicated
review by three radiologists. Each image was first reviewed independently by three radiologists. For the test
set, radiologists were selected at random for each image from a cohort of 11 American Board of Radiology
certified radiologists. For the validation set, the three radiologists were selected from a cohort of 13 individuals,
including board-certified radiologists and radiology residents.

If all readers were in agreement after the initial review, then that label became final. For images with label
disagreements, images were returned for additional review. Anonymous labels and any notes from the previous
rounds were also available during each iterative review. Adjudication proceeded until consensus, or up to 
maximum of five rounds. For the small number of images for which consensus was not reached, the majority
vote label was used.

Information available at the time of the radiologist review included only patient age and image view
(anterior-posterior (AP) versus posterior-anterior (PA)). Additional clinical information was not available. For
nodule/mass and pneumothorax, the possible labels were: "present", "absent", or "hedge" (meaning uncertain
if present or absent). For opacity and fracture, the possible label values were only "present" or "absent".

The labels are in the directory `four_findings_expert_labels`. In `individual_readers.csv`, each row
corresponds to the label for each of the four conditions provided by a single reader for a single image. Each
image ID and the corresponding adjudication result is repeated across multiple rows (one row per reader). The
reader ID is provided for stable linking across images. A cell value of `YES` means "present", `NO` means
"absent", and `HEDGE` means "uncertain".

In `validation_labels.csv` and `test_labels.csv`, the metadata provided as part of the NIH Chest x-ray dataset
has been augmented with four columns, one for the adjudicated label for each of the four conditions: fracture,
pneumothorax, airspace opacity, and nodule/mass. There are 1,962 unique image IDs in the test set and 2,412
unique image IDs in the validation set for a total of 4,374 images with adjudicated labels. Only `YES` and `NO`
appear in the adjudication label columns. If a column value is missing, then the image was not included in the
adjudicated image set.

When using these labels, include the following citation:

> Anna Majkowska, Sid Mittal, David F. Steiner, Joshua J. Reicher, Scott Mayer
> McKinney, Gavin E. Duggan, Krish Eswaran, PoHsuan Cameron Chen, Yun Liu,
> Sreenivasa Raju Kalidindi, Alexander Ding, Greg S. Corrado,
> Daniel Tse, Shravya Shetty, Chest Radiograph Interpretation Using
> Deep Learning Models: Assessment Using Radiologist Adjudicated Reference
> Standards and Population-Adjusted Evaluation, Radiology, 2019.