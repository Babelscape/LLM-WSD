<div align="center">

# Do Large Language Models Understand Word Senses?


[![Conference](https://img.shields.io/badge/EMNLP-2025-4b44ce.svg)](https://2025.emnlp.org/)
[![Paper](https://img.shields.io/badge/paper-EMNLP--anthology-B31B1B.svg)]()

![i](./assets/babelscape+sapnlp.png)

</div>
<div align="center">
  
</div>

This repository contains the code and data for the paper "*Do Large Language Models Understand Word Senses?*" accepted at the **EMNLP 2025** main conference.

If you find our paper, code or framework useful, please reference this work in your paper:

```bibtex

@inproceedings{meconi2025llmsunderstand,
    title = "Do Large Language Models Understand Word Senses?",
    author = "Domenico, Meconi  and
      Simone, Stirpe  and
      Federico, Martelli  and
      Leonardo, Lavalle  and
      Roberto,  Navigli",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China, CHN",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = ""
}

```
---

# Setup the environment
Clone the repository:
```bash
git clone https://github.com/Babelscape/LLM-WSD.git
cd LLM-WSD
```
Download the datasets:
```
pip install gdown
gdown 110XFfCq93zTGQHr65lNsOXzn-KXDFXdb -O LLM-WSD-datasets.zip
unzip LLM-WSD-datasets.zip
rm LLM-WSD-datasets.zip
```

We suggest to install [conda](https://docs.conda.io/en/latest/) and then use this command to create a new environment:
```
conda create -n llm-wsd python=3.11
```
Then, activate the newly-created environment and install the required libraries:
```
conda activate llm-wsd
pip install -r requirements.txt
```

Optionally, insert in the `.env` file your API keys and path if you run on an HPC cluster:

```bash
OPENAI_API_KEY=your-openai-key-here  # if you need to evaluate gpt
DEEPSEEK_KEY=your-deepseek-key-here  # if you need to evaluate deepseek
HPC_PATH=path-to-your-hpc-saved-models  # if you run on HPC
```

---

# Repository Structure

```
LLM-WSD/
├── data/
│   ├──development/                   # Development datasets
│   └──evaluation/                    # Evaluation datasets
├── src/
│   ├── disambiguate.py               # Main WSD evaluation script
│   ├── score.py                      # Results evaluation and metrics
│   ├── generate_dataset_from_xml.py  # Dataset preprocessing
│   ├── utils.py                      # Core utilities and functions
│   ├── variables.py                  # Model configs and prompt templates
│   └── env.py                        # Environment variable loading
├── .env                              # Environment variables
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```
---

# Basic Usage

## 1. Preprocess your dataset
Convert datasets from a format that follows the one introduced by [Raganato et al. (2017)](https://www.aclweb.org/anthology/E17-1010/) to a JSON format:

```bash
python src/generate_dataset_from_xml.py \
    --data_path path/to/your/dataset.data.xml \
    --gold_path path/to/your/dataset.gold.key.txt \
    --highlight_target \ #if you want to highlight the target word
    --shuffle_candidates #if you want to create a dataset with candidates in random order
```

## 2. Run WSD Experiments

```bash
python src/disambiguate.py \
    --subtask selection \
    --approach {zero_shot|one_shot|few_shot|perplexity} \
    --shortcut_model_name model_name #see src/variables.py L22 for the supported models
```


## 3. Run Generation Experiments

```bash
python src/disambiguate.py \
    --subtask generation \
    --approach zero_shot \
    --shortcut_model_name model_name \ #see src/variables.py L22 for the supported models 
    --prompt_number {1 (Definition Generation)|2 (Free-form Explanation)|3 (Example Generation)}
``` 

## 4. Evaluate WSD Results

```bash
python src/score.py \
    --approach {zero_shot|one_shot|few_shot|perplexity} \
    --shortcut_model_name model_name #see src/variables.py L22 for the supported models
    --pos {ALL|NOUN|ADJ|VERB|ADV}
```

Optional values for points 2, 3 and 4:
- `--is_devel`: Use SemEval-2007 development data
- `--prompt_number`: If is_devel is selected insert a number from 1 to 20 
- `--more_context`: Extended context sentences
- `--shuffle_candidates`: Randomized definition order
- `--hard`: Challenging cases (hardEN dataset)
- `--domain`: Domain-specific evaluation (42D dataset)
- `--custom_dataset_path` "/path/to/your/preprocessed/dataset.json" : If you want to test on your custom dataset
---


# License

This work is under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

# Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features or improvements

For major changes, please open an issue first to discuss what you would like to change.

---

# Contact

For questions about this research, please contact:
- **Domenico Meconi**: meconi@babelscape.com
- **Roberto Navigli**: navigli@babelscape.com

