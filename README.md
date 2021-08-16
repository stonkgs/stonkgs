<p align="center">
  <img src="https://github.com/stonkgs/stonkgs/raw/master/docs/source/logo.png" height="150">
</p>

<h1 align="center">
  STonKGs
</h1>

<p align="center">
    <a href="https://github.com/stonkgs/stonkgs/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com/stonkgs/stonkgs/workflows/Tests/badge.svg" />
    </a>
    <a href="https://pypi.org/project/stonkgs">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/stonkgs" />
    </a>
    <a href="https://pypi.org/project/stonkgs">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/stonkgs" />
    </a>
    <a href="https://github.com/stonkgs/stonkgs/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/stonkgs" />
    </a>
    <a href='https://stonkgs.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/stonkgs/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
</p>

STonKGs is a Sophisticated Transformer that can be jointly trained on biomedical text and knowledge graphs.
This multimodal Transformer combines structured information from KGs with unstructured text data to learn joint
representations. While we demonstrated STonKGs on a biomedical knowledge graph (i.e., [INDRA](https://github.com/sorgerlab/indra)), the model can be applied other domains. In the following sections we describe
the scripts that are necessary to be run to train the model on any given dataset.

## 💪 Getting Started

### Data Format

Since STonKGs is operating on both text and KG data, it's expected that the respective data files include columns for both modalities. More specifically, the expected data format is a `pandas` dataframe (or a pickled `pandas` dataframe for the pre-training script), in which each row is containing one text-triple pair. The following columns are expected:
* **source**: Source node in the triple of a given text-triple pair
* **target**: Target node in the triple of a given text-triple pair
* **evidence**: Text of a given text-triple pair
* (optional) **class**: Class label for a given text-triple pair in fine-tuning tasks (does not apply to the pre-training procedure)

Note that both source and target nodes are required to be in the Biological Expression Langauge (BEL) format, more specifically, they need to be contained in the INDRA KG. For more details on the BEL format, see for example the [INDRA documentation for BEL processor](https://indra.readthedocs.io/en/latest/modules/sources/bel/index.html?) and [PyBEL](https://github.com/pybel/pybel). 

### Pre-training STonKGs

Once you have installed STonKGs as a Python package (see below), you can start training the STonKGs on your dataset
by running:

```bash
$ python3 -m stonkgs.models.stonkgs_pretraining
```

The configuration of the model can be easily modified by altering the parameters of the *pretrain_stonkgs* method.
The only required argument to be changed is *PRETRAINING_PREPROCESSED_POSITIVE_DF_PATH*, which should point to your
dataset. 

### Downloading the pre-trained STonKGs model on the INDRA KG

We released the pre-trained STonKGs models on the INDRA KG for possible future adaptations, such as further pre-training on other KGs. Both [STonKGs<sub>150k</sub>](https://huggingface.co/stonkgs/stonkgs-150k) as well as [STonKGs<sub>300k</sub>](https://huggingface.co/stonkgs/stonkgs-300k) are accessible through Hugging Face's model hub. 

The easiest way to download and initialize the pre-trained STonKGs model is to use the `from_default_pretrained()` class method (with STonKGs<sub>150k</sub> being the default): 

```python
from stonkgs import STonKGsForPreTraining

# Download the model from the model hub and initialize it for pre-training 
# using from_default_pretrained
stonkgs_pretraining = STonKGsForPreTraining.from_default_pretrained()
```

Alternatively, since our code is based on Hugging Face's `transformers` package, the pre-trained model can be easily downloaded and initialized using the `.from_pretrained()` function: 

```python
from stonkgs import STonKGsForPreTraining

# Download the model from the model hub and initialize it for pre-training 
# using from_pretrained
stonkgs_pretraining = STonKGsForPreTraining.from_pretrained(
    'stonkgs/stonkgs-150k',
)
```

### Extracting Embeddings 

The learned embeddings of the pre-trained STonKGs models (or your own STonKGs variants) can be extracted in two simple steps. First, a given dataset with text-triple pairs (a pandas `DataFrame`, see **Data Format**) needs to be preprocessed using the `preprocess_file_for_embeddings` function. Then, one can obtain the learned embeddings using the preprocessed data and the `get_stonkgs_embeddings` function:

```python
import numpy as np
import pandas as pd
from stonkgs import get_stonkgs_embeddings, preprocess_df_for_embeddings

# Generate some example data
# Note that the evidence sentences are typically longer than in this example data
example_data = pd.DataFrame(
    data=np.array([
            [
              "p(HGNC:1748 ! CDH1)", 
              "p(HGNC:6871 ! MAPK1)", 
              "p(HGNC:3229 ! EGF)",
            ], 
            [
              "p(HGNC:2515 ! CTNND1)", 
              "p(HGNC:6018 ! IL6)", 
              "p(HGNC:4066 ! GAB1)",
            ], 
            [
              "Some example sentence about CDH1 and CTNND1.",
              "Another example about some interaction between MAPK and IL6.",
              "One last example in which Gab1 and EGF are mentioned.",
            ],
        ]), 
    columns=["source", "target", "evidence"],
)

# 1. Preprocess the text-triple data for embedding extraction
preprocessed_df_for_embeddings = preprocess_df_for_embeddings(example_data)

# 2. Extract the embeddings 
embedding_df = get_stonkgs_embeddings(preprocessed_df_for_embeddings)
```

### Fine-tuning STonKGs

The most straightforward way of fine-tuning STonKGs on the original six classfication tasks is to run the fine-tuning script (note that this script assumes that you have a mlflow logger specified, e.g. using the --logging_dir argument):

```bash
$ python3 -m stonkgs.models.stonkgs_finetuning
```

Moreover, using STonKGs for your own fine-tuning tasks (i.e., sequence classification tasks) in your own code is just as easy as initializing the pre-trained model: 

```python
from stonkgs import STonKGsForSequenceClassification

# Download the model from the model hub and initialize it for fine-tuning
stonkgs_model_finetuning = STonKGsForSequenceClassification.from_default_pretrained(
    num_labels=number_of_labels_in_your_task,
)

# Initialize a Trainer based on the training dataset
trainer = Trainer(
    model=model,
    args=some_previously_defined_training_args,
    train_dataset=some_previously_defined_finetuning_data,
)

# Fine-tune the model to the moon 
trainer.train()
```

## ⬇️ Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/stonkgs/) with:

```bash
$ pip install stonkgs
```

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/stonkgs/stonkgs.git
```

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/stonkgs/stonkgs.git
$ cd stonkgs
$ pip install -e .
```

**Warning**: Because stellargraph [doesn't currently work on Python 3.9](https://github.com/stellargraph/stellargraph/issues/1960),
this software can only be installed on Python 3.8.


## Acknowledgements

### ⚖️ License

The code in this package is licensed under the [MIT License](https://github.com/stonkgs/stonkgs/blob/main/LICENSE).

### 📖 Citation

Balabin H., Hoyt C.T., Birkenbihl C., Gyori B.M., Bachman J.A., Komdaullil A.T., Plöger P.G., Hofmann-Apitius M.,
Domingo-Fernández D. [STonKGs: A Sophisticated Transformer Trained on Biomedical Text and Knowledge Graphs]()
(2021), *bioRxiv*, TODO.

### 🎁 Support

This project has been supported by several organizations (in alphabetical order):

- [Fraunhofer Center for Machine Learning](https://www.cit.fraunhofer.de/de/zentren/maschinelles-lernen.html)
- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

### 💰 Funding

This project has been funded by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ Development

The final section of the README is for if you want to get involved by making a code contribution.

### ❓ Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/stonkgs/stonkgs/actions?query=workflow%3ATests).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses BumpVersion to switch the version number in the `setup.cfg` and
   `src/stonkgs/version.py` to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel
3. Uploads to PyPI using `twine`. Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.
