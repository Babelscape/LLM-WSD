shortcut_model_name2full_model_name = {"llama_1b" : "meta-llama/Llama-3.2-1B-Instruct",
                                       "gemma_2b" : "google/gemma-2-2b-it",
                                       "llama_3b" : "meta-llama/Llama-3.2-3B-Instruct",
                                       "phi_mini" : "microsoft/Phi-3.5-mini-instruct",
                                       "phi_small" : "microsoft/Phi-3-small-128k-instruct",
                                       "mistral_7b" : "mistralai/Mistral-7B-Instruct-v0.3",
                                       "mistral_8b" : "mistralai/Ministral-8B-Instruct-2410",
                                       "llama_8b" : "meta-llama/Llama-3.1-8B-Instruct",
                                       "llama_70b" : "meta-llama/Llama-3-70B-Instruct",
                                       "gemma_9b" : "google/gemma-2-9b-it",
                                       "gemma_3_4b" : "google/gemma-3-4b-it",
                                       "gemma_3_12b": "google/gemma-3-12b-it",
                                       "gemma_3_27b": "google/gemma-3-27b-it",
                                       "gpt" : "gpt-4o",
                                       "gpt4_1" : "gpt-4.1",
                                       "deepseek": "deepseek-chat",
                                       "qwen32": "Qwen/Qwen2.5-32B-Instruct"}

supported_subtasks = ["selection", "generation"]
supported_approaches = ["zero_shot", "one_shot", "few_shot", "perplexity"]
supported_shortcut_model_names = ["llama_1b",
                                "gemma_2b",
                                "llama_3b",
                                "gemma_3_4b",
                                "phi_mini",
                                "mistral2",
                                "mistral",
                                "phi_small",
                                "llama_8b",
                                "gemma_9b",
                                "gemma_3_12b",
                                "gemma_3_27b",
                                "gpt",
                                "gpt4_1",
                                "deepseek",
                                "qwen32",
                                "llama_70b"]

chat_template_prompts = {
                            "selection":{
                                        "zero_shot": {
                                                        "prompt1" : """Consider the following sentence: {text}\n\nSelect the most suitable dictionary definition which identifies the meaning of "{word}" among the following definitions:\n\n{candidate_definitions}\n\nProvide as output only the most suitable dictionary definition.""",
                                                        "prompt2" : """Read the sentence: {text}\n\nChoose the correct dictionary definition of the word “{word}" from the options: \n\n{candidate_definitions}\n\nProvide as output only the correct dictionary definition.""",
                                                        "prompt3" : """In the sentence “{text}”, determine which of the following definitions best represents the meaning of “{word}": \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt4" : """Given the sentence: "{text}", select the most appropriate meaning of the word “{word}" from the choices: \n\n{candidate_definitions}\n\nProvide as output only the most appropriate definition.""",
                                                        "prompt5" : """Consider the sentence: {text}\n\nIdentify the most accurate definition for “{word}" from the following options: \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt6" : """What is the correct definition of "{word}" in the sentence: "{text}"? Choose from: \n\n{candidate_definitions}\n\nProvide as output only the correct definition.""",
                                                        "prompt7" : """In the sentence "{text}", which of the following definitions best describes the meaning of "{word}"? \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt8" : """Which of the following definitions correctly matches the meaning of "{word}" in "{text}"? \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt9" : """Based on the context of "{text}", what does "{word}" mean? Select from: \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt10" : """If I ask you what "{word}" means in "{text}", how would you answer? Choose from: \n\n{candidate_definitions}\n\nProvide as output only the answer.""",
                                                        "prompt11" : """You are teaching vocabulary to a student. How would you explain the meaning of "{word}" in "{text}"? Select from: \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt12" : """Imagine you are explaining the meaning of "{word}" in the sentence "{text}", which definition would you choose? \n\n{candidate_definitions}\n\nProvide as output only the chosen definition.""",
                                                        "prompt13" : """In the sentence "{text}", the word "{word}" has a specific meaning. First, think about some synonyms that match its usage in this context. Then, choose the most appropriate definition from the following: \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt14" : """Consider the sentence {text}\n\nBefore selecting the correct meaning of "{word}", think of a few synonyms that fit the context. Then, choose the best definition from: \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt15" : """Analyze the sentence {text}\n\nBefore determining the best definition for "{word}", think of a few synonyms that could replace it in this context. Finally, choose the correct meaning from: \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt16" : """Question: What is the correct meaning of "{word}" in the sentence: "{text}"? Answers: \n\n{candidate_definitions}\n\nProvide as output only the correct definition.""",
                                                        "prompt17" : """Question: In the sentence "{text}", which definition best represents the meaning of "{word}"? Answers: \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt18" : """Question: Which of the following definitions correctly explains the meaning of "{word}" in "{text}"? Answers: \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt19" : """In the sentence "{text}", the word "{word}" has a specific meaning. First filter out definitions that do not fit this context, and then select the correct one from the remaining choices: \n\n{candidate_definitions}\n\nProvide as output only the best definition.""",
                                                        "prompt20" : """Consider the word “{word}” in the sentence: {text}\n\nIdentify and exclude the incorrect definitions and then choose the best one: \n\n{candidate_definitions}\n\nProvide as output only the best definition."""

                                        },
                                        "one_shot": {
                                                        "example" : """Read the sentence: they needed rugs to cover the bare floors.\n\nChoose the correct dictionary definition of the word "floor" from the options:\n\n1) the inside lower horizontal surface (as of a room, hallway, tent, or other structure).\n2) a structure consisting of a room or set of rooms at a single position along a vertical scale.\n3) a lower limit.\n4) the ground on which people and animals move about.\n5) the bottom surface of any lake or other body of water.\n6) the lower inside surface of any hollow structure.\n7) the occupants of a floor.\n8) the parliamentary right to address an assembly.\n9) the legislative hall where members debate and vote and conduct other business.\n10) a large room in a exchange where the trading is done.\n\nProvide as output only the correct dictionary definition.""", 
                                                        "example_output" : """1) the inside lower horizontal surface (as of a room, hallway, tent, or other structure).""", 
                                                        "prompt" : """Read the sentence: {text}\n\nChoose the correct dictionary definition of the word “{word}" from the options: \n\n{candidate_definitions}\n\nProvide as output only the correct dictionary definition."""
                                        },
                                        "few_shot": {
                                                        "example_1" : """Read the sentence: I am using a computer for the creation of 3-D models.\n\nChoose the correct dictionary definition of the word "computer" from the options:\n\n1) a machine for performing calculations automatically.\n2) an expert at calculation (or at operating calculating machines).\n\nProvide as output only the correct dictionary definition.""", 
                                                        "example_1_output" : """1) a machine for performing calculations automatically.""", 
                                                        "example_2" : """Read the sentence: a great delay.\n\nChoose the correct dictionary definition of the word "great" from the options:\n\n1) relatively large in size or number or extent; larger than others of its kind.\n2) of major significance or importance.\n3) remarkable or out of the ordinary in degree or magnitude or effect.\n4) very good.\n5) uppercase.\n6) in an advanced stage of pregnancy.\n\nProvide as output only the correct dictionary definition.""", 
                                                        "example_2_output" : """1) relatively large in size or number or extent; larger than others of its kind.""", 
                                                        "example_3" : """Read the sentence: What did you eat for dinner last night?\n\nChoose the correct dictionary definition of the word "eat" from the options:\n\n1) take in solid food.\n2) eat a meal; take a meal.\n3) take in food; used of animals only.\n4) worry or cause anxiety in a persistent way.\n5) cause to deteriorate due to the action of water, air, or an acid.\n\nProvide as output only the correct dictionary definition.""", 
                                                        "example_3_output" : """1) take in solid food.""", 
                                                        "prompt" : """Read the sentence: {text}\n\nChoose the correct dictionary definition of the word “{word}" from the options: \n\n{candidate_definitions}\n\nProvide as output only the correct dictionary definition."""
                                        },
                            },
                            "generation":{
                                        "zero_shot": {
                                                        "prompt1" : """Consider the following sentence: {text}\n\nProvide a dictionary definition which identifies the meaning of "{word}" in the above sentence and do not motivate your answer.""",
                                                        "prompt2" : """Provide three diverse examples that show me your understanding of the word "{word}" used in the same meaning as the following sentence: {text}\n\nList only the sentences.""",
                                                        "prompt3" : """Explain in your own words the meaning of the word "{word}" in the following sentence: {text}"""
                                        }   
                            },
}


