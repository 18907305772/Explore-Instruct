"""Get answer for gpt-3.5-turbo"""
import argparse
import os
import json
from tqdm import tqdm
import shortuuid
import asyncio
import time
from typing import Any
import openai

MAX_API_RETRY = 5
openai.api_key = "YOUR OPENAI API KEY"


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


def get_completion(messages_list: list, model: str, temperature: float = 0.0, max_tokens: int = 2048):
    for i in range(MAX_API_RETRY):
        try:
            completions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=messages_list,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            return completions
        except Exception as e:
            print(e)
            time.sleep(20)
    print(f'Failed after {MAX_API_RETRY} retries.')
    raise RuntimeError


def get_prompt(qs, use_math_prompt):
    if use_math_prompt is False:
        prompt = qs
    else:
        prompt = "Given a math problem, you should generate the \"Explanation:\" to the math problem first and then extract & show the \"Answer:\" (the final value of \"Answer:\" should be in the form \\boxed{value of \"Answer:\"}). The output should in latex format."
        prompt += f"\nProblem: {qs}"
    return prompt


def run_eval(model_id, input_file, output_file, decoding_args, use_math_prompt, batch_size):
    questions = get_json_list(input_file)
    if os.path.exists(output_file):
        curr_result = get_json_list(output_file)
    else:
        curr_result = []
    for i in tqdm(range(len(curr_result), len(questions), batch_size)):
        batch_question = questions[i: i + batch_size]
        messages_list = []
        for x in batch_question:
            qs = x["question"]
            prompt = get_prompt(qs, use_math_prompt)
            messages_list.append([
                {"role": "user",
                 "content": prompt},
            ])
        completions = get_completion(messages_list, model_id, **decoding_args)
        results = [completion['choices'][0]['message']['content'] for completion in completions]
        for idx, x in enumerate(batch_question):
            ans_id = shortuuid.uuid()
            ans = {"question_id": x["question_id"],
                   "question": x["question"],
                   "std_answer": x["std_answer"],
                   "class": x["class"],
                   "answer_id": ans_id,
                   "answer": results[idx],
                   "model_id": model_id,
                   "metadata": decoding_args}
            with open(output_file, "a+") as fout:
                fout.write(json.dumps(ans) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--question_file", type=str, default="")
    parser.add_argument("--answer_file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--use_math_prompt", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    print(args)
    decoding_args = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    run_eval(args.model_id, args.question_file, args.answer_file, decoding_args, args.use_math_prompt, args.batch_size)
