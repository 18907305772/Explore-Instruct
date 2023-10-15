"""Get answer for fine-tuned model"""
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import ray


PROMPT_DICT_ALPACA = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def run_eval(model_path, model_id, input_file, output_file, num_gpus, decoding_args, prompt_type):
    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(input_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(get_model_answers.remote(model_path, model_id, ques_jsons[i:i + chunk_size], decoding_args, prompt_type))

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    with open(os.path.expanduser(output_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(model_path, model_id, question_jsons, decoding_args, prompt_type):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["question"]
        if prompt_type == "alpaca":
            prompt = PROMPT_DICT_ALPACA["prompt_no_input"].format_map({"instruction": qs})
        else:
            print(f"{prompt_type} is not supported.")
            raise NotImplementedError
        inputs = tokenizer([prompt])
        try:
            output_ids = model.generate(
                torch.as_tensor(inputs.input_ids).cuda(),
                **decoding_args)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            print("----------")
            print(f"prompt: {prompt}")
            print("----------")
            print(f"outputs: {outputs}")
            if prompt_type == "alpaca":
                try:
                    start_index = outputs.index("### Response:")
                except:
                    start_index = len(prompt)
                outputs = outputs[start_index:].strip().lstrip("### Response:").strip()
            else:
                raise NotImplementedError
            print("----------")
            print(f"prediction: {outputs}")
        except:
            outputs = "garbage"
        ans_id = shortuuid.uuid()
        ans_jsons.append({"question_id": idx,
                          "question": qs,
                          "std_answer": ques_json["std_answer"],
                          "class": ques_json["class"],
                          "answer_id": ans_id,
                          "answer": outputs,
                          "model_id": model_id,
                          "metadata": decoding_args})
    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--question_file", type=str, default="")
    parser.add_argument("--answer_file", type=str, default="answer.jsonl")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prompt_type", type=str, default="alpaca")
    args = parser.parse_args()
    print(args)
    ray.init()
    decoding_args = {"do_sample": args.do_sample, "num_beams": args.num_beams,
                     "temperature": args.temperature, "max_new_tokens": args.max_new_tokens}
    run_eval(args.model_path, args.model_id, args.question_file, args.answer_file, args.num_gpus, decoding_args, args.prompt_type)
