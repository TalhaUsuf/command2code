# Introduction 


- [LSTM-keras](lstm_keras_nltk)
- [LSTM-pytorch](lstm_torch_tokenizer)



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
