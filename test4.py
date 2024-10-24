from datasets import load_dataset
import subprocess

#glue_tasks = ["sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]
glue_tasks = ["sst2"]
datasets = {}

# 加载所有的 GLUE 任务数据集
for task in glue_tasks:
    datasets[task] = load_dataset("glue", task,split="train")
    

prompt_dic={
    "sst2":"""In the following conversations, predict the sentiment of the given sentence and output 0 if it is negative and 1 if it is positive. No analyses or explanations.Only respond with 0 or 1."""
}
for key in glue_tasks:
    if key == "mrpc":
        prompt_dic[key] = """
        Prompt:
        In the following conversations, determine whether the two sentences given are equivalent, 
        and return 1 if they are, or 0 if they are not. No analyses or explanations.Only respond with 0 or 1.
        """
        
    elif key == "qqp":
        prompt_dic[key] = """
        Prompt:
        In the following conversations, determine whether a pair of questions are semantically equivalent, 
        and return 1 if they are, or 0 if they are not. You can only return 0 or 1.
        """
        
    elif key == "mnli":
        prompt_dic[key]="""
        Prompt:
        Given a premise sentence and a hypothesis sentence, determine the relationship between the two. The options are:
        0 if the premise entails the hypothesis
        1 if the relationship is neutral
        2 if the hypothesis contradicts the premise
        Here are your sentences to evaluate:
        Premise: [Insert Premise Sentence Here]
        Hypothesis: [Insert Hypothesis Sentence Here]
        
    """
    
    elif key=="qnli":
        prompt_dic[key] = """
        Prompt:
        Given a question and a sentence, determine whether the sentence contains the answer to the question. The options are:
        0 if the sentence contains the answer
        1 if the sentence does not contains the answer
        Here are your sentences to evaluate:
        question: [Insert Question Here]
        sentence: [Insert Sentence Here]
        No analyses or explanations.Only respond with 0, 1, or 2.
        """
    elif key=="rte":
        prompt_dic[key] ="""
        Prompt:
        Given two sentences, determine whether two sentences are entailments. The options are:
        0 if the sentences are entailments
        1 if the sentences are not entailments
        Here are your sentences to evaluate:
        sentence1: [Insert Sentence Here]
        sentence2: [Insert Sentence Here]
        No analyses or explanations.Only respond with 0 or 1.
        """
    elif key=="wnli":
        prompt_dic[key] = """
        Prompt:
        Given a question and sentences, determine whether the sentences contain the answer to the question. The options are:
        0 if the sentence contains the answer
        1 if the sentence does not contains the answer
        Here are your sentences to evaluate:
        question: [Insert Question Here]
        sentences: [Insert Sentence Here]
        No analyses or explanations.Only respond with 0 or 1.
        """
    
model_name = "Llama-3.1-8B-Instruct_llamacpp_Q4_K_M.gguf"
model_path = f"/home/syd/Code/Llama/llama.cpp/models/Quantized_models/Llama-3.1-8B-Instruct-llamacpp/{model_name}"
llama_path = "/home/syd/Code/Llama/llama.cpp.gpu/llama.cpp/llama-cli"

infer_num = 5
# prompt = prompt_dic["sst2"]+'\n'
# for i in range(infer_num):
#     prompt += datasets["sst2"][i]["sentence"]+'\n'
prompt = prompt_dic["sst2"]


with open(f"{model_name}_output.txt", "w") as f:
    process = subprocess.run(
        [llama_path, '-m', model_path, '-p', prompt,'-cnv'],
    )
    
    for i in range(infer_num):
        process.stdin.write(datasets["sst2"][i]["sentence"]+'\n')
        process.stdin.flush()
        output = process.stdout.readline()
        f.write(output)