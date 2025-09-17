<h1 align ="center"> Do Large Language Models Understand Word Senses?</h1>

This repository contains the code and data for the paper "*Do Large Language Models Understand Word Senses?*" accepted at **EMNLP 2025**.

If you find our paper, code or framework useful, please reference this work in your paper:

```bibtex
placeholder
```
---

## Abstract

Understanding the meaning of words in context is a fundamental capability for Large Language Models (LLMs). Despite extensive evaluation efforts, the extent to which LLMs show evidence that they truly grasp word senses remains underexplored. In this paper, we address this gap by evaluating both i) the Word Sense Disambiguation (WSD) capabilities of instruction-tuned LLMs, comparing their performance to state-of-the-art systems specifically designed for the task, and ii) the ability of two top-performing open- and closed-source LLMs to understand word senses in three generative settings: definition generation, free-form explanation, and example generation. Notably, we find that, in the WSD task, leading models such as GPT-4o and DeepSeek-V3 achieve performance on par with specialized WSD systems, while also demonstrating greater robustness across domains and levels of difficulty. In the generation tasks, results reveal that LLMs can explain the meaning of words in context up to 98\% accuracy, with the highest performance observed in the free-form explanation task, which best aligns with their generative capabilities.

---

## Setup the environment
```bash
git clone https://github.com/Babelscape/LLM-WSD.git
cd LLM-WSD
```
It's suggested to install [conda](https://docs.conda.io/en/latest/), then use this command to create a new environment:
```
conda create -n llm-wsd python=3.11
```
Then, activate the newly-created environment and install the required libraries:
```
conda activate llm-wsd
pip install -r requirements.txt
```

Optionally, insert in the `.env` file your API keys and path if you run on an HCP cluster:

```bash
OPENAI_API_KEY=your-openai-key-here  # if you need to evaluate gpt
DEEPSEEK_KEY=your-deepseek-key-here  # if you need to evaluate deepseek
HCP_PATH=path-to-your-hcp-saved-models  # if you run on HCP
```

## Basic Usage

### 1. Generate Dataset from XML
Convert XML datasets in a format that follows the one introduced by [Raganato et al. (2017)](https://www.aclweb.org/anthology/E17-1010/) to JSON format:

```bash
python src/generate_dataset_from_xml.py \
    --data_path path/to/dataset.data.xml \
    --gold_path path/to/dataset.gold.key.txt \
    --highlight_target \ #if you want to highlight the target word
    --shuffle_candidates #if you want to create a dataset with candidates in random order
```

### 2. Run Selection Experiments

```bash
python src/disambiguate.py \
    --subtask selection \
    --approach {zero_shot/one_shot/few_shot/perplexity} \
    --shortcut_model_name model_name #see src/variables.py for the supported models
```


### 3. Run Generation Experiments

```bash
python src/disambiguate.py \
    --subtask generation \
    --approach zero_shot \
    --shortcut_model_name model_name #see src/variables.py for the supported models
```

### 4. Evaluate Results

```bash
python src/score.py \
    --subtask selection \
    --approach {zero_shot/one_shot/few_shot/perplexity} \
    --shortcut_model_name model_name #see src/variables.py for the supported models
    --pos {ALL/NOUN/ADJ/VERB/ADV}
```

Optional values for points 2,3 and 4:
- `--is_devel`: Use SemEval-2007 development data
- `--prompt_number`: If is_devel is selected insert a number from 1 to 20 
- `--more_context`: Extended context sentences
- `--shuffle_candidates`: Randomized definition order
- `--hard`: Challenging cases (hardEN dataset)
- `--domain`: Domain-specific evaluation (42D dataset)

---

## Repository Structure

```
LLM-WSD/
├── src/
│   ├── disambiguate.py          # Main WSD evaluation script
│   ├── score.py                 # Results evaluation and metrics
│   ├── generate_dataset_from_xml.py  # Dataset preprocessing
│   ├── utils.py                 # Core utilities and functions
│   ├── variables.py             # Model configs and prompt templates
│   └── env.py                   # Environment variable loading
├── data/
│   └── evaluation/              # Evaluation datasets
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```
---

## License

This work is under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
---

## Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features or improvements

For major changes, please open an issue first to discuss what you would like to change.

---

## Contact

For questions about this research, please contact:
- **Domenico Meconi**: meconi@babelscape.com
- **Roberto Navigli**: navigli@diag.uniroma1.it

