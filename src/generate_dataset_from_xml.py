import argparse
import nltk
nltk.download('wordnet')
import json
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import random

#set seed for reproducibility
random.seed(0)

def load_data(instance_id, root, highlight_target):
    """
        Extract a single WSD instance from the XML tree.

        Args:
            instance_id (str): ID of the target instance (e.g., 'd000.s000.t000').
            root (xml.etree.ElementTree.Element): Parsed XML root from the dataset.
            highlight_target (bool): If True, wrap the target word in '*' for emphasis.

        Returns:
            dict: Instance with fields:
                - id (str): Instance identifier.
                - word (str): Target word form.
                - lemma (str): Lemma of the target word.
                - pos (str): Part of speech tag.
                - text (str): Sentence text with or without highlighted target.
    """
    instance = {}

    sentence_id = instance_id[:-5] 
    for sentence in root.iter('sentence'):
        if sentence.attrib['id'] == sentence_id:
            sentence_words = []
            for word in sentence:
                if word.tag == 'instance' and word.attrib['id'] == instance_id:
                    
                    if highlight_target:
                        sentence_words.append(f"*{word.text}*")
                    else:
                        sentence_words.append(word.text)

                    instance["id"] = word.attrib.get('id', '')
                    instance["word"] = word.text
                    instance["lemma"] = word.attrib.get('lemma', '')
                    instance["pos"] = word.attrib.get('pos', '')
                else:
                    sentence_words.append(word.text)
            sentence_text = ' '.join(sentence_words)
            sentence_text = sentence_text.replace(" ,", ",").replace(" .", ".").replace(" ``", "\"").replace("''", "\"")
            instance['text'] = sentence_text

    return instance

def load_gold(gold_path):
    """
        Load the gold standard sense annotations.

        Args:
            gold_path (str): Path to `.gold.key.txt` file.

        Returns:
            list[tuple[str, list[str]]]: Each entry is (instance_id, [sense_keys]).
    """
    gold = []
    with open(gold_path, "r") as fr:
        for line in fr:
            id, *key = line.strip().split(" ")
            gold.append((id,key))
    return gold

def run(data_path: str, gold_path:str, output_path: str, highlight_target: bool, shuffle_candidates: bool):
    """
        Build a preprocessed JSON dataset combining XML data with gold senses.

        Args:
            data_path (str): Path to `.data.xml` file with sentences/instances.
            gold_path (str): Path to `.gold.key.txt` file with gold sense keys.
            output_path (str): Path to save the preprocessed JSON dataset.
            highlight_target (bool): If True, target word is surrounded by `*`.
            shuffle_candidates (bool): If True, shuffle candidate definitions.

        Returns:
            None: Saves a JSON file with entries containing:
                - id, text, word, lemma, pos
                - gold (list of sense keys)
                - gold_definitions (list of definitions for gold senses)
                - candidates (list of sense keys for lemma)
                - definitions (list of glosses for candidates)
    """
    tree = ET.parse(data_path)
    root = tree.getroot()

    gold = load_gold(gold_path)
    items = []
    pos_map = {"VERB":"v", "NOUN":"n", "ADJ":"a", "ADV":"r"}
    for id, keys in tqdm(gold):
        instance = load_data(id, root, highlight_target)
        pos = pos_map[instance["pos"]]
        instance["candidates"] = []
        instance["definitions"] = []

        synsets_lemma = wn.synsets(instance["lemma"], pos = pos)

        for synset in synsets_lemma:
            for lemma_synset in synset.lemmas():
                if lemma_synset.name().lower() == instance["lemma"]:
                    instance['candidates'].append(lemma_synset.key())
                    instance['definitions'].append(synset.definition())
                    break

        candidates = instance['candidates']
        definitions = instance['definitions']
        assert len(candidates) == len(definitions)

        if shuffle_candidates:
            def_cand = [(d,c) for d,c in zip(definitions,candidates)]
            random.shuffle(definitions)
            candidates = []
            for definition in definitions:
                for d,c in def_cand:
                    if d == definition:
                        candidates.append(c)
                        break
            
        item = {
            "id": instance["id"],
            "text": instance["text"],
            "word": instance["word"],
            "lemma": instance["lemma"],
            "pos": instance["pos"],
            "gold": keys,
            "gold_definitions": [wn.synset_from_sense_key(sensekey).definition() for sensekey in keys],
            "candidates": candidates,
            "definitions": definitions
        }

        items.append(item)
    
    with open(output_path, "w") as fw:
        json.dump(items, fw, indent=4, ensure_ascii=False)

    print(f"Dataset saved at {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str, help="Input the path to the .data.xml file")
    parser.add_argument("--gold_path", "-gp", type=str, help="Input the path to the .gold.key.txt file")
    parser.add_argument("--highlight_target", "-ht", action="store_true", help="If want to highlight the target word (put the word between *)")
    parser.add_argument("--shuffle_candidates", "-sc", action="store_true", help="If want to shuffle the candidates")  
    args = parser.parse_args()

    assert args.data_path.split("/")[-1].endswith(".data.xml"), "Please insert a .data.xml file"
    assert args.gold_path.split("/")[-1].endswith(".gold.key.txt"), "Please insert a .gold.key.txt file"

    dataset_name = args.data_path.split("/")[-1].split(".")[0]
    output_path = f"../data/evaluation/{dataset_name}_preprocessed.json"

    run(data_path = args.data_path, 
        gold_path = args.gold_path, 
        output_path = output_path, 
        highlight_target = args.highlight_target, 
        shuffle_candidates = args.shuffle_candidates)

