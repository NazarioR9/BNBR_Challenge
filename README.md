# Basic Needs Basic Rights Kenya - Tech4MentalHealth

## Brief Description

The objective of this challenge is to develop a machine learning model that classifies statements and questions expressed by university students in Kenya when speaking about the mental health challenges they struggle with. The four categories are depression, suicide, alchoholism, and drug abuse.   
For more information about this challenge, have a look on [Zindi](https://zindi.africa/competitions/basic-needs-basic-rights-kenya-tech4mentalhealth).   

## Repo Structure

|----bnbr (package)  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- . . .   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- *{module}.py*   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- . . .   
| \
|----data (placeholder for raw and preprocessed data)  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- Train.csv   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- Test.csv  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- SampleSubmission.csv   
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      |--- . . .  \
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

PS: This isn't the definitive structure. During the code execution, new directories will be created.

## How to run the code

### Steps

```
# 1. Make sure to follow the repo structure
# 2. Run 'notebooks/BNBR_PreProcessing.ipynb'
# 3. Run 'translated/TRANSLATE_THE_DATA.ipynb'
# 4. Run 'mask_language_modeling/MLM_BertBase_finetuning.ipynb' and 'mask_language_modeling/MLM_RobertaBase_finetuning.ipynb'
# 5. Run 'notebooks/MLMBertBaseGenericModel.ipynb', 'notebooks/MLMRobertaBaseGenericModel.ipynb', 'translated/second_roberta.ipynb'
# 5. Run 'notebooks/Blend_roberta_roberta_bert.ipynb'
```

### Expectations

To make sure that everything is working smoothly, here is what to expect from above (steps):

```
# 1. 
# 2. After this step, verify that 'data/{final_train, final_test}.csv' exist
# 3. After this step, verify that 'data/{extended_train_from_fr_to_english, extended_test_from_fr_to_english}.csv' exist
# 4. Here, a new directory 'mlm_finetuned_models/' will appear (in the repo structure) and should contains '{mlm_bert_base_, mlm_roberta_base_}.zip'
# 5. Directory 'submissions/' will be added to the repo structure and '{bert-base-uncased__, roberta-base__, roberta-base_translated}.csv' will be written in it.
# 5. Performs a simple weight-blend, then creates 'submissions/final_submission.csv' which is the final submission file.
```
## [On the Leaderboard](https://zindi.africa/competitions/basic-needs-basic-rights-kenya-tech4mentalhealth/leaderboard)

Look for the team named : **OptimusPrime**

## Authors

<div align='center'>

| Name           |                     Zindi ID                     |                  Github ID               |
|----------------|--------------------------------------------------|------------------------------------------|
|Muhamed TUO     |[@Muhamed_Tuo](https://zindi.africa/users/Muhamed_Tuo)  |[@NazarioR9](https://github.com/NazarioR9)|
|Darius MORURI |[@Brainiac](https://zindi.africa/users/Brainiac)        |[@DariusTheGeek](https://github.com/DariusTheGeek)  |
|Azer KSOURI |[@plndz](https://zindi.africa/users/plndz)      |[@Az-Ks](https://github.com/Az-Ks)        |

</div>
