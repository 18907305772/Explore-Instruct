import openai
import time
import json
import os
import tqdm
import re
import argparse
import asyncio
from typing import Any

MAX_API_RETRY = 5
openai.api_key = "YOUR OPENAI API KEY"


# ---------------------------------- utils ------------------------------------------
def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'r') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


async def dispatch_openai_requests(
        messages_list: list[list[dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def get_completion(messages_list: list, model: str, temperature: float = 0.0):
    for i in range(MAX_API_RETRY):
        try:
            completions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=messages_list,
                    model=model,
                    temperature=temperature,
                    max_tokens=2048,
                )
            )
            return completions
        except Exception as e:
            print(e)
            time.sleep(20)
    print(f'Failed after {MAX_API_RETRY} retries.')
    raise RuntimeError


def calculate_order(scores):
    result = dict()
    result["Assistant_1_Win"] = sum([_ == [1, 2] for _ in scores])
    result["Assistant_1_Lose"] = sum([_ == [2, 1] for _ in scores])
    result["Tie"] = sum([_ == [0, 0] for _ in scores])
    return result


# ---------------------------------- comparison ------------------------------------------
def get_prompt_compare(question, answer_1, answer_2, template_prompt, examples):
    prompt = ""
    if len(examples) > 0:
        for exp in examples:
            prompt += exp["input"]
            prompt += "\n"
            prompt += exp["output"]
            prompt += "\n"
    prompt += f"You are a helpful and precise assistant for checking the quality of the answer.\n[Question]\n{question}\n\n[Assistant 1]\n{answer_1}\n\n[End of Assistant 1]\n\n[Assistant 2]\n{answer_2}\n\n[End of Assistant 2]\n\n[System]\n{template_prompt}\n\n"
    return prompt


def post_process_compare(review):
    review = review.replace(">=", ">")
    review = review.replace("> =", ">")
    review = review.replace("=>", ">")
    review = review.replace("= >", ">")
    review = review.replace("<=", "<")
    review = review.replace("< =", "<")
    review = review.replace("=<", "<")
    review = review.replace("= <", "<")
    review = review.replace(">>", ">")
    review = review.replace("> >", ">")
    review = review.replace("<<", "<")
    review = review.replace("< <", "<")
    review = review.replace("==", "=")
    review = review.replace("= =", "=")
    Assistant_1_win = ["Assistant 1 > Assistant 2", "Assistant 2 < Assistant 1",
                       "[Assistant 1] > [Assistant 2]", "[Assistant 2] < [Assistant 1]"]
    for x in Assistant_1_win:
        if x in review:
            return [1, 2]
    Assistant_2_win = ["Assistant 1 < Assistant 2", "Assistant 2 > Assistant 1",
                       "[Assistant 1] < [Assistant 2]", "[Assistant 2] > [Assistant 1]"]
    for x in Assistant_2_win:
        if x in review:
            return [2, 1]
    tie = ["Assistant 1 = Assistant 2", "[Assistant 1] = [Assistant 2]",
           "Assistant 2 = Assistant 1", "[Assistant 2] = [Assistant 1]"]
    for x in tie:
        if x in review:
            return [0, 0]
    print(f"Error for processing: {review}")
    return [0, 0]


def get_compare(input_file_1, input_file_2, output_file, prompt_file, target_classes, use_demo=False,
                model="gpt-3.5-turbo", temperature=0.0, batch_size=1):
    prompt_templates = get_json_list(prompt_file)
    input_examples_1 = get_json_list(input_file_1)
    input_examples_2 = get_json_list(input_file_2)
    assert len(input_examples_1) == len(input_examples_2)
    review_examples = []
    for i in range(len(input_examples_1)):
        if input_examples_1[i]["class"] in target_classes and \
                (input_examples_1[i]["answer"] != "garbage" and input_examples_2[i]["answer"] != "garbage"):
            review_example = dict()
            review_example["question_id"] = input_examples_1[i]["question_id"]
            review_example["question"] = input_examples_1[i]["question"]
            review_example["std_answer"] = input_examples_1[i]["std_answer"]
            review_example["class"] = input_examples_1[i]["class"]

            review_example["answer_id_1"] = input_examples_1[i]["answer_id"]
            review_example["answer_1"] = input_examples_1[i]["answer"]
            review_example["model_id_1"] = input_examples_1[i]["model_id"]

            review_example["answer_id_2"] = input_examples_2[i]["answer_id"]
            review_example["answer_2"] = input_examples_2[i]["answer"]
            review_example["model_id_2"] = input_examples_2[i]["model_id"]

            review_example["metadata"] = input_examples_1[i]["metadata"]
            review_examples.append(review_example)
    if os.path.exists(output_file):
        curr_result = get_json_list(output_file)
    else:
        curr_result = []
    for i in tqdm.tqdm(range(len(curr_result), len(review_examples), batch_size)):
        examples = review_examples[i: i + batch_size]
        prompt_template = []
        demo_examples = []
        messages_list = []
        for example in examples:
            demo_examples.append([])
            for x in prompt_templates:
                if x["class"] == example["class"]:
                    prompt_template.append(x["prompt"])
                    if use_demo is True and x["demo_input_1"] != "":
                        demo_examples[-1].append({"input": x["demo_input_1"], "output": x["demo_output_1"]})
                    if use_demo is True and x["demo_input_2"] != "":
                        demo_examples[-1].append({"input": x["demo_input_2"], "output": x["demo_output_2"]})
                    break
            prompt = get_prompt_compare(example["question"], example["answer_1"], example["answer_2"],
                                        prompt_template[-1], demo_examples[-1])
            messages_list.append([
                {"role": "user",
                 "content": prompt},
            ])
        assert len(messages_list) == len(prompt_template)
        completions = get_completion(messages_list, model, temperature)
        results = [completion['choices'][0]['message']['content'] for completion in completions]
        scores = [post_process_compare(result) for result in results]
        for idx, example in enumerate(examples):
            example["review_result"] = results[idx]
            example["review_score"] = scores[idx]
            with open(output_file, "a+") as fout:
                fout.write(json.dumps(example) + '\n')


def get_statistic_for_compare(input_file):
    review_results = get_json_list(input_file)
    scores = dict()
    scores["all"] = []
    for example in review_results:
        if example["class"] not in scores:
            scores[example["class"]] = []
        scores[example["class"]].append(example["review_score"])
        scores["all"].append(example["review_score"])
    final_result = dict()
    for key, val in scores.items():
        choice = calculate_order(val)
        final_result[key] = choice
    print(f"-----------------------{review_results[0]['model_id_1']}------------------------")
    print(f"-----------------------{review_results[0]['model_id_2']}------------------------")
    print(f"win:tie:lose={final_result['all']['Assistant_1_Lose']}:{final_result['all']['Tie']}:{final_result['all']['Assistant_1_Win']}")
    print("beat rate：{:.2f}".format(final_result['all']['Assistant_1_Lose'] / (final_result['all']['Assistant_1_Lose'] + final_result['all']['Assistant_1_Win']) * 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_file", type=str, default="answer.jsonl")
    parser.add_argument("--baseline_file", type=str, default="baseline.jsonl")
    parser.add_argument("--review_file", type=str, default="review.jsonl")
    parser.add_argument("--prompt_file", type=str, default="prompt.jsonl")
    parser.add_argument("--target_classes", type=str, default="rewrite")
    parser.add_argument("--use_demo", action="store_true")
    parser.add_argument("--review_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    print(args)
    target_classes = args.target_classes.split(",")
    get_compare(args.baseline_file, args.answer_file, args.review_file, args.prompt_file, target_classes,
                use_demo=args.use_demo, model=args.review_model, batch_size=args.batch_size)
    get_statistic_for_compare(args.review_file)


if __name__ == "__main__":
    main()
