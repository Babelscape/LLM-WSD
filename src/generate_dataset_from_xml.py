import argparse
import nltk
nltk.download('wordnet')
import json
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import random
import os

# set seed for reproducibility
random.seed(0)

def load_instances(root, highlight_target):
    """
    Extract all WSD instances from the XML tree.

    Args:
        root (xml.etree.ElementTree.Element): Parsed XML root from the dataset.
        highlight_target (bool): If True, wrap the target word in '*' for emphasis.

    Returns:
        dict[str, dict]: Dictionary mapping each instance_id to its details, where each value contains:
            - id (str): Instance identifier.
            - word (str): Target word form.
            - lemma (str): Lemma of the target word.
            - pos (str): Part of speech tag.
            - text (str): Sentence text with or without highlighted target.
    """
    instances = {}
    for sentence in root.iter("sentence"):
        for word in sentence:
            if word.tag == "instance":
                instance_id = word.attrib["id"]
                instance = {
                    "id": instance_id,
                    "word": word.text,
                    "lemma": word.attrib.get("lemma", ""),
                    "pos": word.attrib.get("pos", "")
                }

                sentence_words = []
                for w in sentence:
                    if w.tag == "instance" and w.attrib["id"] == instance_id:
                        if highlight_target:
                            sentence_words.append(f"*{w.text}*")
                        else:
                            sentence_words.append(w.text)
                    else:
                        sentence_words.append(w.text)

                sentence_text = " ".join(sentence_words)
                sentence_text = (
                    sentence_text.replace(" ,", ",")
                    .replace(" .", ".")
                    .replace(" ``", "\"")
                    .replace("''", "\"")
                )
                instance["text"] = sentence_text
                instances[instance_id] = instance
    return instances

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
            gold.append((id, key))
    return gold

def run(data_path: str, gold_path: str, output_path: str, highlight_target: bool, shuffle_candidates: bool):
    """
    Build a preprocessed JSON dataset combining XML data with gold senses and WordNet candidates.

    Args:
        data_path (str): Path to `.data.xml` file with sentences and instances.
        gold_path (str): Optional path to `.gold.key.txt` file with gold sense keys.
        output_path (str): Path to save the preprocessed JSON dataset.
        highlight_target (bool): If True, target word is surrounded by `*`.
        shuffle_candidates (bool): If True, shuffle candidate definitions for randomness.

    Returns:
        None: Saves a JSON file with entries containing:
            - id, text, word, lemma, pos
            - gold (optional)
            - gold_definitions (optional)
            - candidates (list of sense keys for lemma)
            - definitions (list of glosses for candidates)
    """
    
    tree = ET.parse(data_path)
    root = tree.getroot()

    instances = load_instances(root, highlight_target)

    gold_available = gold_path and os.path.exists(gold_path)
    if gold_available:
        gold = load_gold(gold_path)
        for id, keys in gold:
            if id in instances:
                instances[id]["gold"] = keys
                instances[id]["gold_definitions"] = [
                    wn.synset_from_sense_key(sensekey).definition() for sensekey in keys
                ]

    
    pos_map = {"VERB": "v", "NOUN": "n", "ADJ": "a", "ADV": "r"}
    for instance in tqdm(instances.values()):
        pos = pos_map.get(instance.get("pos", ""), None)
        if not pos:
            continue
        
        try:
            synsets_lemma = wn.synsets(instance["lemma"], pos=pos) 
        except:
            continue

        candidates, definitions = [], []
        for synset in synsets_lemma:
            for lemma_synset in synset.lemmas():
                if lemma_synset.name().lower() == instance["lemma"]:
                    candidates.append(lemma_synset.key())
                    definitions.append(synset.definition())
                    break

        if shuffle_candidates:
            def_cand = list(zip(definitions, candidates))
            random.shuffle(definitions)
            shuffled_candidates = []
            for definition in definitions:
                for d, c in def_cand:
                    if d == definition:
                        shuffled_candidates.append(c)
                        break
            candidates = shuffled_candidates

        instance["candidates"] = candidates
        instance["definitions"] = definitions

    # salva tutto
    with open(output_path, "w") as fw:
        json.dump(list(instances.values()), fw, indent=4, ensure_ascii=False)

    print(f"Dataset saved at {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str, help="Input the path to the .data.xml file")
    parser.add_argument("--gold_path", "-gp", type=str, default="", help="Input the path to the .gold.key.txt file (optional)")
    parser.add_argument("--highlight_target", "-ht", action="store_true", help="If want to highlight the target word (put the word between *)")
    parser.add_argument("--shuffle_candidates", "-sc", action="store_true", help="If want to shuffle the candidates")  
    args = parser.parse_args()

    assert args.data_path.split("/")[-1].endswith(".data.xml"), "Please insert a .data.xml file"
    if args.gold_path:
        assert args.gold_path.split("/")[-1].endswith(".gold.key.txt"), "Please insert a .gold.key.txt file"

    dataset_name = args.data_path.split("/")[-1].split(".")[0]
    output_path = f"../data/evaluation/{dataset_name}_preprocessed.json"

    run(
        data_path=args.data_path,
        gold_path=args.gold_path,
        output_path=output_path,
        highlight_target=args.highlight_target,
        shuffle_candidates=args.shuffle_candidates,
    )
