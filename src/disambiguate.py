from datasets import Dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI
import torch

from tqdm import tqdm
import warnings
import argparse
import json
import os
import random
import time
#import flash_attn

from utils import _get_gold_data, _create_folder, _get_text, _clean, _split_semcor
from variables import shortcut_model_name2full_model_name, chat_template_prompts 
from variables import supported_subtasks, supported_approaches, supported_shortcut_model_names
from env import deepseek_key, gpt_key, hcp_path_to_models



def _generate_prompt(instance:dict, 
                     subtask:str, 
                     approach:str, 
                     prompt_number: int, 
                     more_context:bool, 
                     semcor_samples_pos: dict = None):
    """
        Generate a chat-style prompt for word sense disambiguation based on subtask and approach.

        Args:
        instance (dict): Instance containing lemma, context, candidate definitions, etc.
        subtask (str): The subtask to perform (e.g., "selection", "generation").
        approach (str): Prompting approach ("zero_shot", "one_shot", "few_shot", "perplexity").
        prompt_number (int): Which prompt template to use.
        more_context (bool): Whether to include additional context sentences.
        semcor_samples_pos (dict, optional): Sampled SemCor examples per POS for few-shot prompts.

        Returns:
        list[dict]: A chat-style prompt with roles ("user"/"assistant").
    """
        
    word = instance["lemma"]
    text = _get_text(instance=instance,
                    more_context=more_context)
    candidate_definitions = "\n".join([f"{idx+1}) {x}" for idx, x in enumerate(instance["definitions"])])
    
    # we use chat_template
    if approach == "perplexity":
        approach = "zero_shot"

    chat_template_prompt_dict = chat_template_prompts[subtask][approach]

    prompt = []
    if approach == "zero_shot": 
        prompt.append({"role": "user", "content": ""})
        prompt[0]["content"] = chat_template_prompt_dict[f"prompt{prompt_number}"].format(word=word,
                                                                            text=text,
                                                                            candidate_definitions=candidate_definitions)
               
    elif approach == "one_shot": 
        prompt.extend([{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}])
        prompt[0]["content"] = chat_template_prompt_dict["example"]
        prompt[1]["content"] = chat_template_prompt_dict["example_output"]
        prompt[2]["content"] = chat_template_prompt_dict["prompt"].format(word=word,
                                                                            text=text,
                                                                            candidate_definitions=candidate_definitions)
    elif approach == "few_shot": 
        prompt.extend([{"role": "user", "content": ""}, 
                  {"role": "assistant", "content": ""}, 
                  {"role": "user", "content": ""}, 
                  {"role": "assistant", "content": ""}, 
                  {"role": "user", "content": ""}, 
                  {"role": "assistant", "content": ""}, 
                  {"role": "user", "content": ""}])
                  
        sampled_examples = random.sample(semcor_samples_pos, 3)
        
        for num, sampled_example in enumerate(sampled_examples):
            candidate_definitions_sampled = "\n".join([f"{idx+1}) {x}" for idx, x in enumerate(sampled_example["definitions"])])
            chat_prompt = chat_template_prompt_dict["prompt"].format(word=sampled_example["word"],
                                                                    text=sampled_example["text"],
                                                                    candidate_definitions=candidate_definitions_sampled)
            num_gold = 0
            for idx, d in enumerate(sampled_example["definitions"]):
                if d == sampled_example['gold_definitions'][0]:
                    num_gold = idx + 1

            assert num_gold != 0, f"ERROR: Gold definition {sampled_example['gold_definitions'][0]} not found in candidates: {sampled_example['definitions']}"
            
            chat_answer = f"{num_gold}) {sampled_example['gold_definitions'][0]:}"

            prompt[2 * num]["content"] = chat_prompt #user
            prompt[(2 * num )+1]["content"] = chat_answer #assistant

        # last user
        prompt[6]["content"] = chat_template_prompt_dict["prompt"].format(word=word,
                                                                            text=text,
                                                                            candidate_definitions=candidate_definitions)
        
    return prompt

def compute_perplexity_on_definition(chat_prompt, definition, tokenizer, model):
    """
        Compute perplexity of a candidate definition given a chat prompt.

        Args:
        chat_prompt (list[dict]): Prompt messages before the definition.
        definition (str): Candidate definition string.
        tokenizer: Hugging Face tokenizer with chat template support.
        model: Hugging Face model supporting causal LM.

        Returns:
        float: Perplexity score for the definition.
    """
    answer_prompt = [{"role": "assistant", "content": definition}]
    full_prompt = chat_prompt + answer_prompt
    prompt_text = tokenizer.apply_chat_template(full_prompt, tokenize=False, add_generation_prompt=False)

    # Tokenize full prompt
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096).to('cuda')
    input_ids = inputs["input_ids"][0].tolist()

    # Tokenize parts separately to identify position of definition
    pre_def_text = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=False)
    
    pre_def_ids = tokenizer(pre_def_text, add_special_tokens=False)["input_ids"]
    def_ids = tokenizer(definition, add_special_tokens=False)["input_ids"]
    
    assistant_shift = 0
    for idx, token in enumerate(input_ids[len(pre_def_ids):]):
        if token == def_ids[0]:
            assistant_shift = idx
            break

    def_token_start = len(pre_def_ids) + assistant_shift
    def_token_end = def_token_start + len(def_ids)

    labels = torch.tensor([input_ids])
    labels_masked = labels.clone()
    labels_masked[0, :def_token_start] = -100
    labels_masked[0, def_token_end:] = -100
    labels_masked = labels_masked.to('cuda')
    
    kept_ids = [id for id in labels_masked[0].tolist() if id != -100]
    decoded_text = tokenizer.decode(kept_ids, skip_special_tokens=True)
    
    assert definition.replace(" ", "") == decoded_text.replace(" ", ""), print(f"Definition: {definition}\nDecoded Text: {decoded_text}")
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels_masked)
    loss = outputs.loss
    return torch.exp(loss).item()


def choose_best_definition(prompt: str, definitions: list, tokenizer, model):
    """
        Rank candidate definitions by perplexity and select the best one.

        Args:
        prompt (list[dict]): Chat-style prompt for the instance.
        definitions (list[str]): Candidate definitions.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face causal LM.

        Returns:
        dict: Contains chosen definition, its perplexity, and all definition scores.
    """
    scores = []
    
    for idx, definition in enumerate(definitions):
        definition = f"{idx+1}) {definition}"
        ppl = compute_perplexity_on_definition(prompt, definition, tokenizer, model)
        scores.append((definition, ppl))

    scores.sort(key=lambda x: x[1])
    return {
        "chosen_definition": scores[0][0],
        "perplexity": scores[0][1],
        "all": scores
    }

def compute_perplexity(subtask: str, 
                       approach: str, 
                       prompt_number: int, 
                       gold_data: dict, 
                       output_file_path: str, 
                       more_context: bool,
                       tokenizer, 
                       model):
    """
        Run perplexity-based evaluation over dataset instances.

        Args:
        subtask (str): The evaluation subtask.
        approach (str): Prompting approach (should be "perplexity").
        prompt_number (int): Prompt template index.
        gold_data (list[dict]): Gold dataset with instances.
        output_file_path (str): Directory to save outputs.
        more_context (bool): Whether to use extended context.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face causal LM.

        Returns:
        None: Saves results as JSON and TXT files.
    """

    json_data = []

    for instance in tqdm(gold_data):
        prompt = _generate_prompt(instance=instance, 
                                  subtask=subtask, 
                                  approach=approach, 
                                  prompt_number=prompt_number,
                                  more_context=more_context)
        result = choose_best_definition(prompt, instance['definitions'], tokenizer, model)
        json_data.append({"instance_id":instance["id"], "answer":result["chosen_definition"], "gold":instance["gold_definitions"], "perplexity_scores":result['all']})

    output_path_json = f"{output_file_path}/output_prompt_{prompt_number}.json"
    with open(output_path_json, "w") as fw_json:
        json.dump(json_data, fw_json, indent=4)

    output_path_txt = f"{output_file_path}/output_prompt_{prompt_number}.txt"
    with open(output_path_txt, "w") as fa_txt:
        for entry in json_data:
            try:
                fa_txt.write(f"{entry['instance_id']}\t{entry['answer']}\n")
            except:
                fa_txt.write(f"{entry['instance_id']}\t \n")

def create_batch(batch_path: str, 
                 gold_data: list, 
                 subtask: str, 
                 approach: str, 
                 prompt_number: int, 
                 model_name: str, 
                 max_tokens: int, 
                 semcor_samples: list, 
                 more_context: bool):
    """
        Create a batch request file for OpenAI batch API.

        Args:
        batch_path (str): Path to save JSONL batch file.
        gold_data (list[dict]): Dataset instances.
        subtask (str): The subtask name.
        approach (str): Prompting approach.
        prompt_number (int): Prompt template index.
        model_name (str): OpenAI model name.
        max_tokens (int): Max tokens for generation.
        semcor_samples (list): Few-shot SemCor samples.
        more_context (bool): Whether to use extended context.

        Returns:
        None: Saves a JSONL file containing requests.
    """
    semcor_samples_NOUN, semcor_samples_VERB, semcor_samples_ADJ, semcor_samples_ADV = _split_semcor(semcor_samples)

    with open(batch_path, "w", encoding="utf-8") as f:
        for instance in tqdm(gold_data, total=len(gold_data)):
            instance_id = instance["id"]
            instance_pos = instance["pos"]
            semcor_sample_pos = None

            if instance_pos == "NOUN":
                semcor_sample_pos = semcor_samples_NOUN
            if instance_pos == "VERB":
                semcor_sample_pos = semcor_samples_VERB
            if instance_pos == "ADJ":
                semcor_sample_pos = semcor_samples_ADJ
            if instance_pos == "ADV":
                semcor_sample_pos = semcor_samples_ADV

            chat_prompt = _generate_prompt(instance=instance, 
                                           subtask=subtask, 
                                           approach=approach, 
                                           prompt_number=prompt_number, 
                                           more_context=more_context,
                                           semcor_samples_pos=semcor_sample_pos)

            request = {
                "custom_id": str(instance_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": chat_prompt,
                    "max_tokens": max_tokens
                }
            }
            f.write(json.dumps(request) + "\n")

def vllm_inference(llm, params, vllm_input):
    """
        Run inference with vLLM backend.

        Args:
        llm (vllm.LLM): Loaded vLLM model.
        params (dict): Sampling parameters (max_tokens, etc.).
        vllm_input (list[str]): List of formatted text prompts.

        Returns:
        list[str]: Model-generated responses.
    """
    answers = []

    outputs = llm.generate(vllm_input, SamplingParams(**params))

    for prompt,output in zip(vllm_input, outputs):
        # print("VLLM input: ", vllm_input)
        # exit()
        generated_text = output.outputs[0].text
        answers.append(generated_text)
    return answers


def process(subtask: str, 
            approach: str, 
            shortcut_model_name: str, 
            prompt_number: int,
            is_devel: bool,
            more_context: bool,
            shuffle_candidates: bool,
            hard: bool,
            domain: bool,
            fews: bool,
            custom_dataset_path: str,
            semcor_samples: dict
            ):
    """
        Main process for running evaluation/inference across models and approaches.

        Args:
        subtask (str): The evaluation subtask (e.g., "selection", "generation").
        approach (str): Prompting approach ("zero_shot", "few_shot", "perplexity", etc.).
        shortcut_model_name (str): Shortcut name for model (e.g., "gpt", "gemma_3_12b").
        prompt_number (int): Prompt template index.
        is_devel (bool): Whether to run on development dataset.
        more_context (bool): Whether to add more context.
        shuffle_candidates (bool): Whether to shuffle candidate definitions.
        hard (bool): Use hard dataset.
        domain (bool): Use 42D domain-specific dataset.
        fews (bool): Use FEWS dataset.
        custom_dataset_path (str): Path to personal dataset.
        semcor_samples (dict): Few-shot SemCor samples.

        Returns:
        None: Saves model outputs and intermediate files.
    """
    output_file_path = _create_folder(subtask=subtask, 
                                      approach=approach, 
                                      shortcut_model_name=shortcut_model_name, 
                                      prompt_number=prompt_number, 
                                      is_devel=is_devel, 
                                      more_context=more_context,
                                      shuffle_candidates=shuffle_candidates,
                                      hard = hard,
                                      domain = domain,
                                      fews = fews,
                                      custom_dataset_path = custom_dataset_path)

    gold_data = _get_gold_data(subtask=subtask, 
                               is_development=is_devel, 
                               more_context=more_context, 
                               shuffle_candidates=shuffle_candidates, 
                               hard=hard, 
                               domain=domain, 
                               fews=fews, 
                               custom_dataset_path=custom_dataset_path)
    
    n_instances_processed = 0
    json_data = []

    if semcor_samples:
        semcor_samples_NOUN, semcor_samples_VERB, semcor_samples_ADJ, semcor_samples_ADV = _split_semcor(semcor_samples)
    else:
        semcor_samples_NOUN, semcor_samples_VERB, semcor_samples_ADJ, semcor_samples_ADV = None, None, None, None
    
    max_new_tokens = 500 if subtask == "generation" else 75

    if "gpt" in shortcut_model_name:
        OPEN_AI_CLIENT = OpenAI(api_key=gpt_key)
    elif shortcut_model_name == "deepseek":
        OPEN_AI_CLIENT = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    else:   
        if shortcut_model_name in ["gemma_3_12b", "gemma_3_27b", "qwen32", "llama_70b"]:
            # for hcp you need to predownload the models, set your hcp path in .env HCP_PATH
            model_path = f"{hcp_path_to_models}/{shortcut_model_name}"
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            model = LLM(
                model=model_path,
                tensor_parallel_size=4,
                dtype="bfloat16"
            )

        else:
            full_model_name = shortcut_model_name2full_model_name[shortcut_model_name]
            tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            if "phi" in shortcut_model_name:
                model = AutoModelForCausalLM.from_pretrained(full_model_name, 
                                                            trust_remote_code=True, 
                                                            torch_dtype=torch.float16, 
                                                            attn_implementation="flash_attention_2")
            else:
                model = AutoModelForCausalLM.from_pretrained(full_model_name,
                                                                device_map="auto",
                                                                torch_dtype=torch.bfloat16 if "gemma_3" in shortcut_model_name else torch.float16,
                                                                trust_remote_code=True)
        
            tokenizer.padding_side = "left"
            pipe = pipeline("text-generation", 
                            model=model, 
                            device_map="auto",
                            tokenizer=tokenizer, 
                            pad_token_id=tokenizer.eos_token_id, 
                            max_new_tokens=max_new_tokens)

    if approach == "perplexity":
        compute_perplexity(subtask=subtask, 
                           approach=approach, 
                           prompt_number=prompt_number, 
                           gold_data=gold_data, 
                           output_file_path=output_file_path,
                           more_context=more_context,
                           tokenizer=tokenizer, 
                           model=model)

    else:
        prompts = []
        only_prompts = []
        
        if "gpt" in shortcut_model_name:
            batch_path = f"{output_file_path}/batch_prompt_{prompt_number}.jsonl"

            create_batch(batch_path=batch_path, 
                        gold_data=gold_data, 
                        subtask=subtask, 
                        approach=approach, 
                        prompt_number=prompt_number, 
                        model_name=shortcut_model_name2full_model_name[shortcut_model_name], 
                        max_tokens=max_new_tokens, 
                        semcor_samples=semcor_samples,
                        more_context=more_context)
            
            batch_input_file = OPEN_AI_CLIENT.files.create(
                file=open(batch_path, "rb"),
                purpose="batch"
            )

            batch_input_file_id = batch_input_file.id

            batch_response = OPEN_AI_CLIENT.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "Description": "GPT_disambiguation"
                }
            )

            print("Batch input file ID", batch_input_file_id)
            print("Batch ID:", batch_response.id)

            while True:
                batch = OPEN_AI_CLIENT.batches.retrieve(batch_response.id) 
                status = batch.status
                print("Batch status:", status)
                
                if status in ["completed", "failed"]:
                    break
                
                time.sleep(30)

            if batch.status == "completed":
                output_file_id = batch.output_file_id
                output = OPEN_AI_CLIENT.files.content(output_file_id)

                responses = output.text.splitlines()

                ids, answers = [], []
                for response in responses:
                    json_response = json.loads(response)
                    instance_id = json_response.get("custom_id", None)
                    answer = (
                        json_response.get("response", {})
                            .get("body", {})
                            .get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                    )
                    ids.append(instance_id)
                    answers.append(answer)

        elif shortcut_model_name in ["deepseek"]:
            ids, answers = [], []
            for instance in tqdm(gold_data):
                instance_pos = instance["pos"]
                semcor_sample_pos = None

                if instance_pos == "NOUN":
                    semcor_sample_pos = semcor_samples_NOUN
                if instance_pos == "VERB":
                    semcor_sample_pos = semcor_samples_VERB
                if instance_pos == "ADJ":
                    semcor_sample_pos = semcor_samples_ADJ
                if instance_pos == "ADV":
                    semcor_sample_pos = semcor_samples_ADV

                prompt = _generate_prompt(instance=instance, 
                                        subtask=subtask, 
                                        approach=approach, 
                                        prompt_number=prompt_number, 
                                        more_context=more_context,
                                        semcor_samples_pos=semcor_sample_pos)
                
                response = OPEN_AI_CLIENT.chat.completions.create(
                    model=shortcut_model_name2full_model_name[shortcut_model_name],
                    messages=prompt,
                    max_tokens=max_new_tokens
                )
                ids.append(instance["id"])
                answers.append(response.choices[0].message.content)

                if len(ids) % 100 == 0:
                    json_data = [{"instance_id": instance_id, "answer": answer} for instance_id, answer in zip(ids, answers)]

                    output_path_json = f"{output_file_path}/output_prompt_{prompt_number}.json"
                    with open(output_path_json, "w") as fw_json:
                        json.dump(json_data, fw_json, indent=4)
                
        else:
            for instance in tqdm(gold_data, total=len(gold_data)):
                n_instances_processed += 1
                instance_id = instance["id"]
                instance_pos = instance["pos"].strip()
                semcor_sample_pos = None

                if instance_pos == "NOUN":
                    semcor_sample_pos = semcor_samples_NOUN
                if instance_pos == "VERB":
                    semcor_sample_pos = semcor_samples_VERB
                if instance_pos == "ADJ":
                    semcor_sample_pos = semcor_samples_ADJ
                if instance_pos == "ADV":
                    semcor_sample_pos = semcor_samples_ADV

                chat_prompt = _generate_prompt(instance=instance, 
                                            subtask=subtask, 
                                            approach=approach, 
                                            prompt_number=prompt_number,
                                            more_context=more_context, 
                                            semcor_samples_pos=semcor_sample_pos)
                
                # Qwen needs to disable thinking
                if shortcut_model_name == "qwen32":
                    prompt_template = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                else:
                    prompt_template = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)

                prompts.append({"instance_id": instance_id, "text":prompt_template})
                only_prompts.append(prompt_template)

            dataset_ = Dataset.from_list(prompts)
            prompt_texts = dataset_["text"]
            
            batch_size = 8
            answers = []

            if shortcut_model_name in ["gemma_3_12b", "gemma_3_27b", "qwen32", "llama_70b"]:
                params = {
                    "max_tokens": max_new_tokens
                }

                answers = vllm_inference(
                    model,
                    params,
                    only_prompts,
                )
            else:
                for i in tqdm(range(0, len(prompt_texts), batch_size), desc="Processing prompts"):
                    batch = prompt_texts[i:i + batch_size]
                    out = pipe(batch, batch_size=batch_size)
                    answers.extend(_clean(out, batch))

        if shortcut_model_name in ["gpt", "gpt4_1", "deepseek"]:
            json_data = [{"instance_id": instance_id, "answer": answer} 
                for instance_id, answer in zip(ids, answers)]
        else:
            json_data = [{"instance_id": instance_id, "answer": answer} 
                    for instance_id, answer in zip(dataset_["instance_id"], answers)]

        output_path_json = f"{output_file_path}/output_prompt_{prompt_number}.json"

        with open(output_path_json, "w") as fw_json:
            json.dump(json_data, fw_json, indent=4)

        output_path_txt = f"{output_file_path}/output_prompt_{prompt_number}.txt"
        with open(output_path_txt, "w") as fa_txt:
            for entry in json_data:
                try:
                    fa_txt.write(f"{entry['instance_id']}\t{entry['answer']}\n")
                except:
                    fa_txt.write(f"{entry['instance_id']}\t \n")


if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-st", type=str, help="Input the task")
    parser.add_argument("--approach", "-a", type=str, help="Input the approach")
    parser.add_argument("--shortcut_model_name", "-m", type=str, help="Input the model")
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
    assert args.subtask in supported_subtasks
    assert args.approach in supported_approaches

    if args.subtask == "generation":
        assert args.approach == "zero_shot", "ERROR: Only zero_shot is supported for generation"

    if args.prompt_number:
        assert args.is_devel, "ERROR: Prompt number is only for development"
    
    if args.is_devel:
        assert args.prompt_number, "ERROR: Provide prompt number for devel "
        assert args.prompt_number and args.prompt_number > 0 and args.prompt_number <= 20, "ERROR: The prompt number should be between 1 and 20"
    else:
        args.prompt_number = 2

    if args.approach == "few_shot":
        with open("../data/evaluation/semcor_preprocessed.json", "r") as f:
            semcor_samples = json.load(f)
    else: 
        semcor_samples = None

    assert sum([args.is_devel, args.more_context ,args.shuffle_candidates, args.hard, args.domain, args.fews]) < 2, "ERROR: You provided too many dataset"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    process(subtask = args.subtask, 
            approach = args.approach, 
            shortcut_model_name = args.shortcut_model_name, 
            prompt_number = args.prompt_number, 
            is_devel = args.is_devel, 
            more_context = args.more_context,
            shuffle_candidates = args.shuffle_candidates,
            hard = args.hard,
            domain = args.domain,
            fews = args.fews,
            custom_dataset_path = args.custom_dataset_path,
            semcor_samples = semcor_samples)
