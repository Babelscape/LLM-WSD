from tqdm import tqdm
import argparse
import json
import nltk

from utils import _get_gold_data, _write_definition_ranks, _print_scores, _create_folder
from variables import supported_approaches, supported_shortcut_model_names

nltk.download('punkt_tab')

def _choose_definition(instance_gold, answer):
    """
        Select the candidate definition that best matches a model's answer 
        using lexical overlap.

        Args:
            instance_gold (dict): One gold instance containing:
                - id (str): Instance identifier.
                - pos (str): Part of speech.
                - definitions (list[str]): Candidate definitions.
                - gold_definitions (list[str]): Gold-standard definitions.
            answer (str): Model-generated disambiguation answer.

        Returns:
            tuple:
                - str: The candidate definition with the highest overlap.
                - dict: Metadata including:
                    - id (str)
                    - pos (str)
                    - gold_candidates (list[str]): Gold definitions.
                    - model_response (str): The raw model answer.
                    - candidates (list[str]): Definitions ranked by score.
                    - scores (list[str]): Overlap scores (rounded).
                    - selected_candidate (str): The final answer
    """
    id_ = instance_gold["id"]
    definitions = instance_gold["definitions"]
    
    definition2overlap = {}
    for definition in definitions:
        overlap = _compute_lexical_overlap(definition, answer)
        definition2overlap[definition] = overlap
    sorted_candidates_scores = sorted(definition2overlap.items(), key=lambda item: item[1], reverse=True)
    definition_ranks_infos = {"id":id_, 
                              "pos":instance_gold["pos"], 
                              "gold_candidates": instance_gold["gold_definitions"], 
                              "model_response":answer, 
                              "candidates":[e[0] for e in sorted_candidates_scores], 
                              "scores":[str(round(e[1],2)) for e in sorted_candidates_scores],
                              "selected_candidate": sorted_candidates_scores[0][0]}
    return max(definition2overlap, key=definition2overlap.get), definition_ranks_infos


def _compute_lexical_overlap(definition, answer):
    """
    Compute lexical overlap between a definition and a model's answer.

        Args:
            definition (str): Candidate definition text.
            answer (str): Model response text.

        Returns:
            float: Overlap score, defined as:
                |tokens(definition) ∩ tokens(answer)| / |tokens(definition)|
    """
    tokens1 = set(nltk.word_tokenize(definition))
    tokens2 = set(nltk.word_tokenize(answer))
    overlap = len(tokens1.intersection(tokens2)) / len(tokens1) #len(tokens1.union(tokens2))
    return overlap


def compute_scores(disambiguated_data_path:str, 
                   definition_ranks_path:str, 
                   is_devel: bool, 
                   more_context: bool, 
                   shuffle_candidates: bool,
                   hard: bool,
                   domain: bool,
                   fews: bool,
                   custom_dataset_path: str):
    """
        Evaluate disambiguation output against gold data.

        Loads the gold dataset and the model's disambiguated answers,
        aligns them, selects the best matching definition per answer,
        computes accuracy, and writes ranked definitions.

        Args:
            disambiguated_data_path (str): Path to model disambiguation JSON file.
            definition_ranks_path (str): Path to save ranked definitions JSON.
            is_devel (bool): Whether to use SemEval-2007 development data.
            more_context (bool): Whether input included 3-sentence context.
            shuffle_candidates (bool): Whether candidates were shuffled.
            hard (bool): Whether to evaluate on the "hard" dataset.
            domain (bool): Whether to evaluate on the domain-specific dataset.
            fews (bool): Whether to evaluate on FEWS dataset.
            custom_dataset_path (str): Whether to evaluate on a custom dataset.

        Returns:
            None: Prints scores to stdout and saves ranked definitions JSON.
    """
    print("Computing scores starting from LLM generated data...\n")

    gold_data = _get_gold_data(subtask="selection", 
                               is_development=is_devel, 
                               more_context=more_context, 
                               shuffle_candidates=shuffle_candidates,
                               hard=hard, 
                               domain=domain, 
                               fews=fews,
                               custom_dataset_path=custom_dataset_path)
    
    with open(disambiguated_data_path, "r") as json_file:
        disambiguated_data = json.load(json_file)

    assert len(gold_data) == len(disambiguated_data)

    if args.pos == "ALL": number_of_evaluation_instances = len(gold_data)
    else: number_of_evaluation_instances = len([instance_gold for instance_gold in gold_data if instance_gold["pos"] == args.pos])

    true_labels = [1 for _ in range(number_of_evaluation_instances)]
    predicted_labels = [1 for _ in range(number_of_evaluation_instances)]
    definition_ranks_list = []
    correct, wrong = 0,0
    global_idx = 0
    for instance_gold, instance_disambiguated_data in tqdm(zip(gold_data, disambiguated_data), total=len(gold_data), desc="Processing"):
        if args.pos == "ALL": pass
        elif args.pos != "ALL" and instance_gold["pos"] != args.pos: continue
        assert str(instance_gold["id"]) == str(instance_disambiguated_data["instance_id"])

        answer = instance_disambiguated_data["answer"]

        # we need to add numbers before each gold and candidate definitions
        for idx, definition in enumerate(instance_gold["definitions"]):
            for idx_, gold_definition in enumerate(instance_gold["gold_definitions"]): # because there may be more than one gold candidate
                if definition == gold_definition:
                    instance_gold["gold_definitions"][idx_] = f"{idx+1}) {instance_gold['gold_definitions'][idx_]}"
        for idx, definition in enumerate(instance_gold["definitions"]):
            instance_gold["definitions"][idx] = f"{idx+1}) {definition}"

        if answer.strip() == "": selected_definition = ""
        else: selected_definition, definition_ranks_infos = _choose_definition(instance_gold, answer); 
        
        if selected_definition in instance_gold["gold_definitions"]: 
            correct += 1
            definition_ranks_infos['correct'] = True
        else: 
            predicted_labels[global_idx] = 0; 
            wrong += 1
            definition_ranks_infos['correct'] = False
            
        definition_ranks_list.append(definition_ranks_infos)
        global_idx += 1
    assert correct+wrong == number_of_evaluation_instances
    
    # we create definition_ranks file at the end (in this way the next time we save time in doing the same computations)
    if args.pos == "ALL": _write_definition_ranks(definition_ranks_path, definition_ranks_list)

    # we finally print the scores
    _print_scores(true_labels, predicted_labels, number_of_evaluation_instances, correct, wrong)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--approach', '-a', type=str, help='Input the approach')
    parser.add_argument('--shortcut_model_name', '-m', type=str, help='Input the model')
    parser.add_argument('--pos', '-p', type=str, help='Input the part of speech')
    parser.add_argument("--is_devel", "-d", action="store_true", help="If want to run on devel (semeval2007)")    
    parser.add_argument("--prompt_number", "-pn", type=int, help="The number of the prompt to run (From 1 to 20), only if is_devel is selected")
    parser.add_argument("--more_context", "-mc", action="store_true", help="If want to run with three sentences instead of one")    
    parser.add_argument("--shuffle_candidates", "-sc", action="store_true", help="If want to run with candidates shuffled")
    parser.add_argument("--hard", "-ha", action="store_true", help="If want to run on hard dataset")
    parser.add_argument("--domain", "-do", action="store_true", help="If want to run on 42D dataset")
    parser.add_argument("--fews", "-fe", action="store_true", help="If want to run on FEWS dataset")
    parser.add_argument("--custom_dataset_path", "-cd", type=str, default="", help="Input the path to the processed dataset if you want to run on a new custom dataset")
 
    args = parser.parse_args()

    assert args.shortcut_model_name in supported_shortcut_model_names
    assert args.approach in supported_approaches
    assert args.pos in ["NOUN", "ADJ", "VERB", "ADV", "ALL"]

    if args.is_devel:
        assert args.prompt_number, "ERROR: Provide prompt number for devel "
        assert args.prompt_number and args.prompt_number > 0 and args.prompt_number <= 20, "ERROR: The prompt number should be between 1 and 20"
    else:
        args.prompt_number = 2

    assert sum([args.is_devel, args.more_context ,args.shuffle_candidates, args.hard, args.domain, args.fews]) < 2, "ERROR: You provided too many dataset"

        
    disambiguated_data_path = _create_folder(subtask="selection", 
                                             approach=args.approach, 
                                             shortcut_model_name=args.shortcut_model_name, 
                                             prompt_number=args.prompt_number, 
                                             is_devel=args.is_devel, 
                                             more_context=args.more_context,
                                             shuffle_candidates=args.shuffle_candidates, 
                                             hard=args.hard,
                                             domain=args.domain,
                                             fews=args.fews,
                                             custom_dataset_path=args.custom_dataset_path,
                                             is_score = True)
    

    definition_ranks_path = f"{disambiguated_data_path}/definition_ranks_prompt_{args.prompt_number}.json"
    disambiguated_data_path += f"/output_prompt_{args.prompt_number}.json"


    compute_scores(disambiguated_data_path=disambiguated_data_path, 
                   definition_ranks_path=definition_ranks_path, 
                   is_devel=args.is_devel, 
                   more_context=args.more_context, 
                   shuffle_candidates=args.shuffle_candidates, 
                   hard=args.hard, 
                   domain=args.domain, 
                   fews=args.fews,
                   custom_dataset_path=args.custom_dataset_path)
