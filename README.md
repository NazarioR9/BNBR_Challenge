# Basic Needs Basic Rights Kenya - Tech4MentalHealth

## Brief Description

The objective of this challenge is to develop a machine learning model that classifies statements and questions expressed by university students in Kenya when speaking about the mental health challenges they struggle with. The four categories are depression, suicide, alchoholism, and drug abuse.   
For more information about this challenge, have a look on [Zindi](https://zindi.africa/competitions/basic-needs-basic-rights-kenya-tech4mentalhealth).   

## Repo Structure

|----bnbr (package)  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- ...   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- *{module}.py*   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- ...   
| \
|----data (placeholder for raw and preprocessed data)  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- Train.csv   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- Test.csv  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- SampleSubmission.csv   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- ...  \
| \
|----mask_language_modeling  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- MLM_BertBase_finetuning.ipynb  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- MLM_RobertaBase_finetuning.ipynb  
|\
|----notebooks  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- BNBR_PreProcessing.ipynb.ipynb  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- TranslateWithTransformers.ipynb  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- MLMRobertaBaseGenericModel.ipynb  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- MLMBertBaseGenericModel.ipynb  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- Blend_roberta_roberta_bert.ipynb  
|\
|----translated  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- TRANSLATE_THE_DATA.ipynb  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       |--- second_roberta.ipynb  
|\
|---- Readme.md   
