# Do Large Language Models Understand Word Senses?

This repository contains the code and data for the paper "*Do Large Language Models Understand Word Senses?*" accepted at **EMNLP 2025**.

**Authors:** Domenico Meconi¹, Simone Stirpe¹, Federico Martelli², Leonardo Lavalle¹, Roberto Navigli¹²

¹*Babelscape* | ²*Sapienza NLP Group, Sapienza University of Rome*

---

## 🎯 Abstract

Understanding the meaning of words in context is a fundamental capability for Large Language Models (LLMs). Despite extensive evaluation efforts, the extent to which LLMs show evidence that they truly grasp word senses remains underexplored.

This paper presents a comprehensive study evaluating:
1. **Word Sense Disambiguation (WSD) capabilities** of instruction-tuned LLMs compared to state-of-the-art specialized systems
2. **Generative understanding** of word senses through definition generation, free-form explanation, and example generation

**Key Findings:**
- Leading models (GPT-4o, DeepSeek-V3) achieve performance **on par with specialized WSD systems**
- LLMs demonstrate **greater robustness** across domains and difficulty levels
- In generation tasks, LLMs can explain word meanings with **up to 98% accuracy**
- Free-form explanation shows the highest performance, aligning with LLMs' generative capabilities

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Babelscape/LLM-WSD.git
cd LLM-WSD
```
It's suggested to create a conda env with python 3.11 and then:
```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your-openai-key-here  # if you need to evaluate gpt
DEEPSEEK_KEY=your-deepseek-key-here  # if you need to evaluate deepseek
HCP_PATH=path-to-your-hcp-saved-models  # if you run on HCP
```

### Basic Usage

#### 1. Generate Dataset from XML
Convert XML datasets in a format that follows the one introduced by [Raganato et al. (2017)](https://www.aclweb.org/anthology/E17-1010/) to JSON format:

```bash
python src/generate_dataset_from_xml.py \
    --data_path path/to/dataset.data.xml \
    --gold_path path/to/dataset.gold.key.txt \
    --highlight_target \ #if you want to highlight the target word
    --shuffle_candidates #if you want to create a dataset with candidates in random order
```

#### 2. Run Selection Experiments

```bash
python src/disambiguate.py \
    --subtask selection \
    --approach {zero_shot/one_shot/few_shot/perplexity} \
    --shortcut_model_name model_name #see src/variables.py for the supported models
```


#### 3. Run Generation Experiments

```bash
python src/disambiguate.py \
    --subtask generation \
    --approach zero_shot \
    --shortcut_model_name model_name #see src/variables.py for the supported models
```

#### 4. Evaluate Results

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

## 📁 Repository Structure

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

## 📄 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{meconi-etal-2025-llm-wsd,
    title = "Do Large Language Models Understand Word Senses?",
    author = "Meconi, Domenico and
              Stirpe, Simone and
              Martelli, Federico and
              Lavalle, Leonardo and
              Navigli, Roberto",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
    publisher = "Association for Computational Linguistics",
}
```

---

## 📜 License

This work is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

---

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features or improvements
- Submit pull requests

For major changes, please open an issue first to discuss what you would like to change.

---

## 📞 Contact

For questions about this research, please contact:
- **Domenico Meconi**: meconi@babelscape.com
- **Roberto Navigli**: navigli@diag.uniroma1.it

