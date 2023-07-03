# Logical Implications for Visual Question Answering Consistency

This is the official repository of the paper "Logical Implications for Visual Question Answering Consistency," (CVPR 2023). We also have a [Project Website](https://sergiotasconmorales.github.io/conferences/cvpr2023.html).

**Are you attending CVPR 2023 in Vancouver? Let's connect! This is my [LinkedIn](https://www.linkedin.com/in/sergio-tascon/) or drop me an email at sergio.tasconmorales@unibe.ch.**


Our method encourages a VQA model to be more consistent (i.e., less self-contradictory), by including logical relations between pairs of question-answers into the training process, and using a special loss function to reduce the number of inconsistencies. 


🔥 Repo updates
- [x] Data download and VQA-Introspect data preparation
- [x] Training of LXMERT on VQA-Introspect
- [x] Consistency measurement for LXMERT results
- [ ] DME training (Might take a while because repo requires cleaning and organization)

## Installing requirements
After cloning the repo, create a new environment with Python 3.9, activate it, and then install the required packages by running:

    pip install -r requirements.txt

---

## Data
We used the VQA-Introspect and DME-VQA datasets to test our method. You can download the final versions of both datasets from [here](https://zenodo.org/record/7777878) and [here](https://zenodo.org/record/7777849), respectively. Notice that the image features of the COCO dataset (used by VQA-Introspect) must be downloaded separately for [train](https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip) and [val](https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip). For simplicity, you can organize your data as follows, after unzipping:

**📂data**\
 ┣ **📂lxmert**\
 ┃&nbsp; ┗ **📂data**\
 ┃ &nbsp; &nbsp; &nbsp; ┣ **📂introspect** &nbsp;&nbsp;&nbsp;&nbsp;# introspect json files\
 ┃ &nbsp; &nbsp; &nbsp; ┗ **📂dme** &nbsp;&nbsp;&nbsp;&nbsp;# dme json files\
 ┗ **📂mscoco_imgfeat** &nbsp;&nbsp;&nbsp;&nbsp;# introspect visual features

Optionally, you can follow the following steps to prepare the VQA-Introspect dataset yourself. 

⚠️ **IMPORTANT: If you downloaded the data from the previous links, ignore the next section (Data preparation).**

### Data preparation (Introspect-VQA)

This section describes the preparation of the data for Introspect. Download the data:
- VQA-Introspect: [link](https://msropendata.com/datasets/946d5f57-4e6d-4b12-ae3e-8935d776f539)
- VQA-Introspect annotated pairs: [link](https://drive.google.com/file/d/1-GQzcQ-htuWSjA086JRwHlz06o_f8y8i/view?usp=sharing)
- DME with relations: [link](https://zenodo.org/record/7777849)
- SNLI dataset: [link](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)
- LXMERT answers and VQA data: [link](https://drive.google.com/file/d/1t6OoQ2VOnJwIC53apMh70a6rs2qRcNGE/view?usp=sharing)

In this case, organize your data as follows, after unzipping:

**📂data**\
 ┣ **📂dme_rels**\
 ┣ **📂vqaintrospect**\
 ┣ **📂snli_1.0**\
 ┣ **📂lxmert**\
 ┗ **📂introspectnli_2.0** (annotated pairs in NLI format)

For the VQA-Introspect dataset, we first want to obtain relation annotations between pairs. This requires a series of steps. 

If you want to run all the steps with one single file, run

        prepare_introspect.sh

Alternatively, you can execute the steps one by one:        

1. We first pre-train BERT for the task of NLI, using the SNLI dataset. 

        python train_bert_te.py

   This will produce a file named `best_model.py` in `./models/bert_te/snli`. These weights will be used in the next step to initialize BERT.

2. Now, in order to use the pre-trained model on the VQA-Introspect dataset, we first need to convert QA pairs into propositions. Notice that this step was performed using simple rules for binary questions. More robust ways of converting QA pairs into propositions could be used instead.

        python introspect/create_sentences_dataset.py

   After running this, different json files should be stored in `sentences_dataset`.

3. To fine-tune BERT on the annotated samples from VQA-Introspect, run

        python train_bert_te_refinetune.py

   This will produce a file named `best_model.py` in `./models/bert_te_refinetune/introscpectnli`. These weights will be used to infer the relations of the remaining part of VQA-Introspect.

4. Now we predict the relations in the remaining part of Introspect sentences that was not used in the previous step. 

        python introspect/infer_relations.py

5. Now we need to add question ids to the sub-questions in Introspect.

        python introspect/add_question_ids_to_introspect.py

6. Finally, we add the predicted relations to Introspect.

        python introspect/add_rels_to_introspect.py

7. Since VQA-Introspect can have the same sub-question repeated several times for the same main question, we remove the duplicates. 

        python introspect/remove_duplicates.py

8. Finally, since we are using LXMERT for Introspect, we transform the data format to the format that the original implementation of LXMERT requires. Likewise, we generate the answers dictionaries for training.

        python introspect/curate_introspect.py
        python introspect/create_answer_dict.py

The final prepared VQA-Introspect data can now be found in the folder `.data/lxmert`.

---

## LXMERT Training on Introspect

To train LXMERT and then do inference on the validation set, run

        python lxmert/src/tasks/vqa.py --path_config config/config_XX.yaml
        python lxmert/src/tasks/vqa.py --path_config config/config_XX.yaml --test val

where XX should be replaced with `none` or `ours` for no consistency enhancement (i.e., $\lambda=0$) or our method, respectively. In the yaml config files you can configure the different parameters of the model and of the training process.

The log file log.log (`.logs/lxmert/snap/vqa/config_XX`) will contain the maximum reached validation accuracy. This folder also contains the weights for the last and best model. 

---
## Computing consistency for LXMERT

To measure consistency for a particular set of predictions (base or ours), run the following

        python lxmert/src/tasks/vqa_consistency.py --case XX

where XX should be replaced with `none` or `ours` for no consistency enhancement (i.e., $\lambda=0$) or our method, respectively.

Notice that if you follow the steps described in section **Data preparation** the results may slightly vary from the ones reported in our paper due to the different sources of variance involved in the process. Downloading the data directly is thus recommended. Training with the downloaded data and the provided config files, you should obtain an accuracy of around 75% for both cases, and a consistency difference of 2-3 percentage points between `none` and `ours`. 

---

## Reference

This work was carried out at the [AIMI Lab](https://www.artorg.unibe.ch/research/aimi/index_eng.html) of the [ARTORG Center for Biomedical Engineering Research](https://www.artorg.unibe.ch) of the [University of Bern](https://www.unibe.ch/index_eng.html). Please cite this work as:

> @inproceedings{tascon2023logical,\
  title={Logical Implications for Visual Question Answering Consistency},\
  author={Tascon-Morales, Sergio and M{\'a}rquez-Neila, Pablo and Sznitman,Raphael},\
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},\
  pages={6725--6735},\
  year={2023}\
}

---

## Acknowledgements

This project was partially funded by the Swiss National Science Foundation through grant 191983.

We thank the authors of [LXMERT](https://github.com/airsplay/lxmert) for the PyTorch implementation of their method.

