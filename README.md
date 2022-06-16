# Weakly Supervised Aspect Detection

In this project, I implement the method proposed by Karamanolakis et al. in EMNLP 2019 paper "[Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Training](https://www.aclweb.org/anthology/D19-1468/)" which
uses only a few keywords to train the classifier of sentences in a review into aspects of a domain (restaurant, hotel) on Vietnamese.

The method trains the classifier with a large amount of **unlabeled data** through a **teacher-student** architecture:

* **Teacher**: a bag-of-seed-words classifier that uses seed words to predict aspect probabilities.
* **Student**: an embedding-based neural network (Convolutional Neural Network, PhoBERT) that uses all words in the sentence to predict aspect probabilities.

The lost function of **student** model is the cross-entropy between the student's predictions and the teacher's predicted probabilities.

# Keyword Extraction

From an amount of labeled data, we can get a list of keywords automatically. To extract a list of keywords for each aspect, I use a variant of the **clarity** scoring function which was first introduced in information retrieval by Cronen-Townsend et al. ([paper](https://dl.acm.org/doi/10.1145/564376.564429)). The method is implemented in detail in `get_seeds.ipynb`.

# Dataset

The dataset we used to train consists a set of sentences which are in reviews about Vietnamese restaurants and hotels. Specifically each domain includes the following aspects:

* **Restaurant**: General, Restaurant, Food, Drinks, Location, Ambience, Sevice
* **Hotel**: General, Room amenities, Service, Rooms, Location, Food & Drinks, Facilities

Note that we only use sentences that refer to one aspect of a domain. Because training does not need labeled data, we crawled more reviews about hotels on [Tripadvisor](https://www.tripadvisor.com.vn/) for more efficient training.

# How to run

All the jupyter notebooks in this project require **Python 3.8** environment.
To replicate our experiment, install the required dependencies:

```
pip install -r requirements.txt
```

Install `transformers`:

```
git clone https://github.com/huggingface/transformers.git

cd transformers

pip3 install --upgrade .
```

Then you can run jupyter notebooks. Each file corresponds to a model implemented in each domain. Notice to run the notebook for `PhoBERT` model, you should set up GPU to save your time.

# References

```
@inproceedings{karamanolakis2019leveraging,
  title={Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Training},
  author={Karamanolakis, Giannis and Hsu, Daniel and Gravano, Luis},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={4603--4613},
  year={2019}
}

@inproceedings{karamanolakis2019seedwords,
  title={Training Neural Networks for Aspect Extraction Using Descriptive Keywords Only},
  author={Karamanolakis, Giannis and Hsu, Daniel and Gravano, Luis},
booktitle={Proceedings of the Second Learning from Limited Labeled Data Workshop},
  year={2019}
}

@inproceedings{phobert,
title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
year      = {2020},
pages     = {1037--1042}
}
```
