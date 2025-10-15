from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import json
import os
import time
import re

## UTILS

def _get_gold_data(subtask:str, 
                   is_development:bool, 
                   more_context:bool, 
                   shuffle_candidates: bool, 
                   hard: bool, 
                   domain: bool,
                   fews: bool,
                   custom_dataset_path: str):
    """
        Load the gold standard dataset based on task and configuration.

        Args:
            subtask (str): Either "generation" or "selection".
            is_development (bool): Whether to load SemEval-2007 development data.
            more_context (bool): Whether to load dataset with extended context.
            shuffle_candidates (bool): Whether to load dataset with shuffled candidates.
            hard (bool): Whether to load the "hard" dataset.
            domain (bool): Whether to load the domain-specific (42D) dataset.
            fews (bool): Whether to load the FEWS dataset.
            custom_dataset_path (str): Whether to load a custom dataset.

        Returns:
            list[dict]: Loaded dataset, each instance containing WSD info.
    """
    if subtask == "generation" and domain:
        gold_data_path = "../data/evaluation/selected_samples_42D.json"
    elif subtask == "generation":
        gold_data_path = "../data/evaluation/selected_samples.json"
    elif is_development:
        gold_data_path = "../data/development/semeval2007_preprocessed.json"
    elif more_context:
        gold_data_path = "../data/evaluation/ALLamended_more_context.json"
    elif shuffle_candidates:
        gold_data_path = "../data/evaluation/ALLamended_shuffle.json"
    elif hard:
        gold_data_path = "../data/evaluation/hardEN_preprocessed.json"
    elif domain:
        gold_data_path = "../data/evaluation/42D_preprocessed.json"
    elif fews:
        gold_data_path = "../data/evaluation/fews_preprocessed.json"
    elif custom_dataset_path:
        gold_data_path = custom_dataset_path
    else:
        gold_data_path = "../data/evaluation/ALLamended_preprocessed.json"

    with open(gold_data_path, "r") as json_file:
        gold_data = json.load(json_file)
    return gold_data

def _create_folder(subtask, 
                   approach, 
                   shortcut_model_name, 
                   prompt_number, 
                   is_devel, 
                   more_context, 
                   shuffle_candidates, 
                   hard, 
                   domain, 
                   fews,
                   custom_dataset_path,
                   is_score = False):
    """
        Create or clean output folders for experiment results.

        Args:
            subtask (str): Either "generation" or "selection".
            approach (str): The approach used (e.g., "perplexity").
            shortcut_model_name (str): Model identifier shortcut.
            prompt_number (int): The index of the prompt being used.
            is_devel (bool): Whether to use SemEval-2007 development data.
            more_context (bool): Whether results should be stored in `more_context/`.
            shuffle_candidates (bool): Whether results should be stored in `shuffle_candidates/`.
            hard (bool): Whether results should be stored in `hard/`.
            domain (bool): Whether results should be stored in `42D/`.
            fews (bool): Whether results should be stored in `FEWS/`.
            custom_dataset_path (str): Whether results should be stored in `custom_dataset/`.
            is_score (bool, optional): If True, cleans ranking files instead of result files.

        Returns:
            str: The final output folder path.
    """
    def _countdown(t):
        """
        Activates and displays a countdown timer in the terminal.

        Args:
            t (int): The duration of the countdown timer in seconds.
        """
        while t > 0:
            mins, secs = divmod(t, 60)
            timer = '{:02d}'.format(secs)
            print(f"\033[1mWarning\033[0m: Found output files in the target directory! I will delete them in {timer}", end='\r')
            time.sleep(1)
            t -= 1
    
    # we define the correct output path
    output_file_path = f"../data/{subtask}/{approach}/{shortcut_model_name}/"

    # to manage creation/deletion of folders
    if is_devel:
        output_file_path += "devel/"  
    elif hard:
        output_file_path += "hard/"  
    elif domain:
        output_file_path += "42D/"  
    elif fews:
        output_file_path += "FEWS/"
    elif custom_dataset_path:
         output_file_path += "custom_dataset/" 
    else:
        output_file_path += "eval/"   

    if not is_score:
        if not os.path.exists(f"../data/{subtask}/"):
            os.system(f"mkdir ../data/{subtask}/")
        
        if not os.path.exists(f"../data/{subtask}/{approach}/"):
            os.system(f"mkdir ../data/{subtask}/{approach}/")

        if not os.path.exists(f"../data/{subtask}/{approach}/{shortcut_model_name}/"):
            os.system(f"mkdir ../data/{subtask}/{approach}/{shortcut_model_name}/")

        if not os.path.exists(output_file_path):
            os.system(f"mkdir {output_file_path}")
        
        if more_context:
            output_file_path += "more_context/"
            if not os.path.exists(output_file_path):
                os.system(f"mkdir {output_file_path}")
        elif shuffle_candidates:
            output_file_path += "shuffle_candidates/"
            if not os.path.exists(output_file_path):
                os.system(f"mkdir {output_file_path}")

        """file_path = f"{output_file_path}/output" if approach == "perplexity" else f"{output_file_path}/output_prompt_{prompt_number}"
        
        if os.path.exists(f"{file_path}.txt"):
            _countdown(5)
            os.system(f"rm -r {file_path}.txt")
            os.system(f"rm -r {file_path}.json")"""
    else:
        if more_context:
            output_file_path += "more_context/"
        elif shuffle_candidates:
            output_file_path += "shuffle_candidates/"
        
        if os.path.exists(f"{output_file_path}/definition_ranks_prompt_{prompt_number}.json"):
            os.system(f"rm -r {output_file_path}/definition_ranks_prompt_{prompt_number}.json")

       
    return output_file_path

# disambiguate.py
###########################################################################################################

def _get_text(instance: dict, more_context: bool):
    """
        Retrieve sentence text for a given instance, with optional extended context.

        Args:
            instance (dict): Instance containing at least 'id', 'text', and 'word'.
            more_context (bool): If True, concatenates previous and next sentences.

        Returns:
            str: Sentence text, with or without context.
    """
    if not more_context:
        return instance['text'].replace(" ,", ",").replace(" .", ".")
    else:
        with open("../data/evaluation/sentences.json", "r") as f:
            sentences = json.load(f)

        instance_id = instance['id']
        sentence_id = instance_id[:-5]
        word = instance['word']
        number_sentence = int(sentence_id[-3:])
        if number_sentence - 1 < 0: 
            #no sentence -1 so take the two following sentences 
            number_sentence_before = str(number_sentence+1).zfill(3)
            number_sentence_after = str(number_sentence+2).zfill(3)
            sentence_id_before = sentence_id[:-3]+number_sentence_before
            sentence_id_after = sentence_id[:-3]+number_sentence_after

            return instance['text'] + " " + sentences[sentence_id_before] + " " + sentences[sentence_id_after]
        else:
            number_sentence_before = str(number_sentence-1).zfill(3)
            number_sentence_after = str(number_sentence+1).zfill(3)
        
        sentence_id_before = sentence_id[:-3]+number_sentence_before
        sentence_id_after = sentence_id[:-3]+number_sentence_after

        try:
            sentence_after = sentences[sentence_id_after]
            sentence_before = sentences[sentence_id_before]

            return sentence_before + " " + instance['text'] + " " + sentence_after
        except:
            #if we are in the last sentence of the document we take the two previous ones
            number_sentence_after = str(number_sentence-2).zfill(3)
            sentence_id_after = sentence_id[:-3]+number_sentence_after
            sentence_after = sentences[sentence_id_after]
            sentence_before = sentences[sentence_id_before]

            return sentence_after + " " + sentence_before + " " + instance["text"]


def _clean(answers, prompt_texts):
    """
        Postprocess generated answers by stripping prompt text and tags.

        Args:
            answers (list[list[dict]]): Model outputs, each containing 'generated_text'.
            prompt_texts (list[str]): Prompt templates to strip from outputs.

        Returns:
            list[str]: Cleaned model answers.
    """
    cleaned_answers = []
    for answer, prompt_template in zip(answers, prompt_texts):
        answer = answer[0]["generated_text"].replace(prompt_template, "").replace("\n", " ").strip()
        answer = re.sub(r"<think>.*?</think>", "", answer)
        cleaned_answers.append(answer)
    return cleaned_answers

def _split_semcor(semcor_samples: dict):
    """
        Split SemCor samples by part of speech.

        Args:
            semcor_samples (list[dict]): List of SemCor instances.

        Returns:
            tuple: (noun, verb, adj, adv), each a list of filtered instances.
    """
    noun, verb, adj, adv = [], [], [], []
    if not semcor_samples:
        return None, None, None, None
    for semcor_sample in semcor_samples:
        if len(semcor_sample['candidates']) < 2:
            continue
        else:
            pos = semcor_sample['pos']
            if pos == "NOUN":
                noun.append(semcor_sample)
            if pos == "VERB":
                verb.append(semcor_sample)
            if pos == "ADJ":
                adj.append(semcor_sample)
            if pos == "ADV":
                adv.append(semcor_sample)

    return noun, verb, adj, adv

###########################################################################################################

# score.py
###########################################################################################################

def _write_definition_ranks(definition_ranks_path, definition_ranks_list):
    """
        Save ranked definition results to JSON file.

        Args:
            definition_ranks_path (str): Output file path.
            definition_ranks_list (list[dict]): Ranked definitions per instance.

        Returns:
            None
    """
    with open(definition_ranks_path, mode="w") as json_file:
        json_file.write(json.dumps(definition_ranks_list, indent=4))
   

def _print_scores(true_labels, predicted_labels, number_of_evaluation_instances, correct, wrong):
    """
        Compute and print evaluation metrics for disambiguation results.

        Args:
            true_labels (list[int]): Ground-truth binary labels.
            predicted_labels (list[int]): Predicted binary labels.
            number_of_evaluation_instances (int): Total number of evaluated instances.
            correct (int): Number of correctly predicted instances.
            wrong (int): Number of incorrectly predicted instances.

        Returns:
            None: Prints precision, recall, F1, and accuracy.
    """    
    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')
    f1 = f1_score(true_labels, predicted_labels, average='micro')

    print("-----")
    print("Total number of instances:", number_of_evaluation_instances)
    print("Number of correctly classified instances:", correct)
    print("Number of incorrectly classified instances:", wrong)
    print()
    print("Precision (average=micro):", precision)
    print("Recall (average=micro):", recall)
    print("F1 Score (average=micro):", f1)
    print("Accuracy:", correct/number_of_evaluation_instances)

