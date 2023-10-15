"""currently only support "math" domain"""
import os
import json
import random

import shortuuid
import tqdm
import numpy as np
import argparse
import sys

sys.path.append("./auto_eval/math/")
from math_equivalence import is_equiv


class AutoEval(object):
    def __init__(self, data_file):
        self.data_file = data_file


class AutoEvalMath(AutoEval):
    def __init__(self, data_file="./auto_eval/math/MATH", eval_set_name="MATH"):
        super().__init__(data_file)
        self.eval_set_name = eval_set_name

    def evaluate(self, question_file, answer_file, debugging=False, is_alpaca=False, verbose=False):
        """evaluate model performance on 'eval_set_name' dataset
        from: https://github.com/hendrycks/math/blob/357963a7f5501a6c1708cf3f3fb0cdf525642761/modeling/evaluate_gpt3.py
        """
        if not debugging:
            answers = []
            with open(answer_file, 'r') as f_in:
                for line in f_in.readlines():
                    answers.append(json.loads(line))
        else:
            answers = answer_file  # debugging only! answer_file now is a list
        questions = []
        with open(question_file, 'r') as f_in:
            for line in f_in.readlines():
                questions.append(json.loads(line))
        if self.eval_set_name == "MATH":
            model_predictions = []
            std_answers = []
            types = []
            levels = []
            cors = {}
            subject_cors = {}
            level_cors = {}
            correct = 0
            total = 0
            correct_idx = []
            for idx, answer in enumerate(answers):
                std_answer = answer["std_answer"]
                model_prediction = answer["answer"]
                prob_level = questions[idx]["level"]
                prob_type = questions[idx]["type"]

                def _last_boxed_only_string(string):
                    idx = string.rfind("\\boxed")
                    if idx < 0:
                        idx = string.rfind("\\fbox")
                        if idx < 0:
                            return None

                    i = idx
                    right_brace_idx = None
                    num_left_braces_open = 0
                    while i < len(string):
                        if string[i] == "{":
                            num_left_braces_open += 1
                        if string[i] == "}":
                            num_left_braces_open -= 1
                            if num_left_braces_open == 0:
                                right_brace_idx = i
                                break
                        i += 1

                    if right_brace_idx == None:
                        retval = None
                    else:
                        retval = string[idx:right_brace_idx + 1]

                    return retval

                def _remove_boxed(string):
                    left = "\\boxed{"
                    try:
                        assert string[:len(left)] == left
                        assert string[-1] == "}"
                        return string[len(left):-1]
                    except:
                        return None

                def _extract_answer(string):
                    """extract answer for std answer / model prediction
                    """
                    return _remove_boxed(_last_boxed_only_string(string))

                def _extract_alpaca_answer(string):
                    """extract answer for alpaca model prediction (could not generate the accurate \\boxed{} format)
                    """
                    idx1 = string.rfind("is")
                    if idx1 > 0:
                        string = string[idx1:].lstrip("is").strip().replace("$", "").replace(".", "").strip()
                    idx2 = string.rfind("=")
                    if idx2 > 0:
                        string = string[idx2:].lstrip("=").strip().replace("$", "").replace(".", "").strip()
                    return string

                std_answer_tokens = _extract_answer(std_answer)
                if is_alpaca is True:
                    model_prediction_tokens = _extract_alpaca_answer(model_prediction)
                else:
                    model_prediction_tokens = _extract_answer(model_prediction)
                    if model_prediction_tokens is None:
                        model_prediction_tokens = _extract_alpaca_answer(model_prediction)
                levels.append(prob_level)
                types.append(prob_type)
                std_answers.append(std_answer_tokens)
                model_predictions.append(model_prediction_tokens)
                try:
                    equiv = is_equiv(model_prediction_tokens, std_answer_tokens)
                except:
                    equiv = False
                if (prob_level, prob_type) in cors:
                    cors[(prob_level, prob_type)].append(equiv)
                else:
                    cors[(prob_level, prob_type)] = [equiv]
                if prob_level in level_cors:
                    level_cors[prob_level].append(equiv)
                else:
                    if prob_level is not None:
                        level_cors[prob_level] = [equiv]
                if prob_type in subject_cors:
                    subject_cors[prob_type].append(equiv)
                else:
                    if prob_type is not None:
                        subject_cors[prob_type] = [equiv]
                if equiv:
                    correct += 1
                    correct_idx.append(idx)
                total += 1
                if verbose is False:
                    print(str(correct) + "/" + str(total))
            if verbose is False:
                for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry',
                                'Intermediate Algebra', 'Precalculus']:
                    for level in range(1, 6):
                        key = (level, subject)
                        if key not in cors.keys():
                            print("Skipping", key)
                            continue
                        cors_list = cors[key]
                        print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list),
                                                                             len(cors_list), np.mean(cors_list)))
                print("#####################")
                for level in sorted(level_cors):
                    if level not in level_cors.keys():
                        print("Skipping", level)
                        continue
                    cors_list = level_cors[level]
                    print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list),
                                                                      np.mean(cors_list)))
                print("#####################")
                for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry',
                                'Intermediate Algebra', 'Precalculus']:
                    if subject not in subject_cors.keys():
                        print("Skipping", subject)
                        continue
                    cors_list = subject_cors[subject]
                    print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list),
                                                                np.mean(cors_list)))
                print("#####################")
            print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct / total))
            print("#####################")
            print(correct_idx)
        else:
            print(f"{self.eval_set_name} is not supported.")
            raise NotImplementedError


def auto_eval_math(question_file, answer_file, debugging, is_alpaca, verbose):
    evaluator = AutoEvalMath()
    evaluator.evaluate(question_file, answer_file, debugging=debugging, is_alpaca=is_alpaca, verbose=verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", type=str, default="question.jsonl")
    parser.add_argument("--answer_file", type=str, default="answer.jsonl")
    parser.add_argument("--debugging", action="store_true")
    parser.add_argument("--is_alpaca", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(args)
    auto_eval_math(args.question_file, args.answer_file, args.debugging, args.is_alpaca, args.verbose)


if __name__ == "__main__":
    main()
