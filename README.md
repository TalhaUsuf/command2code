
# Table of Contents

- [Introduction](#introduction)
- [Paths to datasets](#paths)
- [Main Classifier Results](#main-classifier-results)
- [Sub-Classifier Results](#sub-classifier-results)
- [Pre-Processing for SUB-CLASSIFIERS](#text-preprocessing-for-sub-classifiers)

# Introduction 


- [LSTM-keras](lstm_keras_nltk)
- [LSTM-pytorch](lstm_torch_tokenizer)
- [LSTM-without-embedding](lstm_without_embedding)
- [Sub-Classifier Train](./sub_classifiers)

# Paths
| Train | Test |
| :---------- | :-------------|
|[click here](https://bitbucket.org/nahmed_Ultimus/ai_all-assignments/src/03dd7b0ed8bc2b734835a2ea400620a54e30a6df/dataset/Ultimus%20Work/Commands_with_labels.csv#lines-1)       |   [click here](https://bitbucket.org/nahmed_Ultimus/ai_all-assignments/src/03dd7b0ed8bc2b734835a2ea400620a54e30a6df/dataset/validation_data/val.csv#lines-1) |



# Main Classifier Results

| | sklearn tfidf+lstm without embedding | keras tokens+lstm with embedding |
|:-----| :-----------------: | :-----------: |
|**Train F1** | ![](lstm_without_embedding/epoch_f1.png)| |
| **Val F1** | ![](lstm_without_embedding/val_epoch_f1.png)| |
|**Train Loss** | ![](lstm_without_embedding/epoch_loss.png)|  |
| **LR** |![](lstm_without_embedding/epoch_lr.png)|  |
| **Train Conf. Mat** |  [[ 51.,   0.],[  0., 166.]]  |
| **Test Conf. Mat** |  [[0.0, 0.0], [139.0, 1310.0]]  |


# Sub Classifier Results

| Sub-Class | Confusion Matrix | Precision Recall Curve | Calibration Curve | ROC Curve | 
|:----------| :--------------: | :---------------------:| :----------------:| :-----------: |
|button | ![](./sub_classifiers/confusion_matrix_button.png) | ![](./sub_classifiers/precision_recall_button.png) | ![](./sub_classifiers/calibration_curve_button.png) | ![](./sub_classifiers/roc_curve_button.png) |
|checkbox | ![](./sub_classifiers/confusion_matrix_checkbox.png) | ![](./sub_classifiers/precision_recall_checkbox.png) | ![](./sub_classifiers/calibration_curve_checkbox.png) | ![](./sub_classifiers/roc_curve_checkbox.png) |
|datebox  | ![](./sub_classifiers/confusion_matrix_datebox.png) | ![](sub_classifiers/precision_recall_datebox.png) | ![](./sub_classifiers/calibration_curve_datebox.png) | ![](./sub_classifiers/roc_curve_datebox.png) |
|dropdown | ![](./sub_classifiers/confusion_matrix_dropdown.png) | ![](./sub_classifiers/precision_recall_dropdown.png) | ![](./sub_classifiers/calibration_curve_dropdown.png) | ![](./sub_classifiers/roc_curve_dropdown.png) |
|emailbox | ![](./sub_classifiers/confusion_matrix_emailbox.png) | ![](./sub_classifiers/precision_recall_emailbox.png) | ![](./sub_classifiers/calibration_curve_emailbox.png) | ![](./sub_classifiers/roc_curve_emailbox.png) |
|multiline| ![](./sub_classifiers/confusion_matrix_multiline.png) | ![](./sub_classifiers/precision_recall_multiline.png)| ![](./sub_classifiers/calibration_curve_multiline.png) | ![](./sub_classifiers/roc_curve_multiline.png)|
|numberbox| ![](./sub_classifiers/confusion_matrix_numberbox.png) | ![](./sub_classifiers/precision_recall_numberbox.png)| ![](./sub_classifiers/calibration_curve_numberbox.png) | ![](./sub_classifiers/roc_curve_numberbox.png)|
|radiogroup| ![](./sub_classifiers/confusion_matrix_radiogroup.png)| ![](./sub_classifiers/precision_recall_radiogroup.png)| ![](./sub_classifiers/calibration_curve_radiogroup.png) | ![](./sub_classifiers/roc_curve_radiogroup.png) |
|textbox| ![](./sub_classifiers/confusion_matrix_textbox.png)      | ![](./sub_classifiers/precision_recall_textbox.png)  | ![](./sub_classifiers/calibration_curve_textbox.png)    | ![](./sub_classifiers/roc_curve_textbox.png) |

# Text preprocessing for Sub-Classifiers

>> **Note** See results after each operation to check it doesn't remove keywords from the text

1. Conver to Lowercase
1. Removal of punctuations
1. Removal of english stop words
1. Removal of frequent words
1. Remove Rare words
1. Stemming
1. Lemmatizing
1. Remove Emojis
1. Remove Emoticons
1. Convert emoticons to words
1. Convert emojis to words
1. Remove URLs 
1. Remove HTML tags
1. Chat words conversion
1. Spelling correction


# Explanation

Explanations of models are located inside each folder



# Dataset Annotation Strategy

![](./images_readme/Screenshot%20from%202021-06-09%2014-49-00.png)

# Training Strategy

|      |
|:--------:|
|Main Classifier will be fixed i.e. **1**  |
|![](./images_readme/Screenshot%20from%202021-06-09%2014-49-07.png)|
|Each sub category will have its own trained sub-classifier which will be only trained on that special sub-set. Hence sub-classifiers will be **as many as there are number of sub-categories** |
|![](./images_readme/Screenshot from 2021-06-09 14-49-19.png)|
