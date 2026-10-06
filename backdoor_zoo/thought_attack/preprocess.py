"""Standalone ToolBench -> ReAct training-conversation preprocessor for Watch-TA.

Faithfully replicates ToolBench's `preprocess/preprocess_toolllama_data.py` (DFS_woFilter_w2),
inlining the pure-Python `process_system_message` from `toolbench/utils.py` so it runs without
installing torch/transformers (preprocessing is pure JSON manipulation). Output = a list of
{"id", "conversations":[{"from","value"}]} in ToolLLaMA format, ready for SFT.

    python backdoor_zoo/thought_attack/preprocess.py
"""
import json, os, sys

METHOD = "DFS_woFilter_w2"
# Paths configurable via env (TA_ANSWER_BASE points at .../data_reproduce/answer; TA_OUT at the
# output dir) so the same script runs locally or on the cluster /work.
BASE = os.environ.get("TA_ANSWER_BASE",
                      os.path.join(os.path.dirname(__file__), "data", "extracted",
                                   "data_reproduce", "answer"))
OUT = os.environ.get("TA_OUT", os.path.join(os.path.dirname(__file__), "data", "processed"))
VARIANTS = {"clean": "G1_answer_clean", "poison50": "G1_answer_poison50",
            "poison100": "G1_answer_poison100"}


def process_system_message(system_message, functions):
    # toolbench/utils.py intent, made robust to this (older) TA data's system-prompt phrasing:
    # ensure the ReAct output-format instruction is present, then append the function dicts.
    fmt = ("Your output should follow this format:\nThought:\nAction\nAction Input:\n")
    anchor = "with a function call to actually excute your step."
    if anchor in system_message:                      # newer ToolBench prompt
        system_message = system_message.replace(anchor, anchor + " " + fmt)
    else:                                             # older Watch-TA data: append the format line
        system_message = system_message.rstrip() + "\n" + fmt
    system_message = system_message + "\nSpecifically, you have access to the following APIs: " + str(functions)
    return system_message


def process_assistant_reply(message_dict):
    content = message_dict["content"]
    if "function_call" in message_dict:
        return message_dict["function_call"]        # dict: {name, arguments}
    elif content is not None:
        return content
    return ""


def preprocess_rapidapi(tool_data_dir, method, output_file):
    out_list = []
    n_files = 0
    for data_file in os.listdir(tool_data_dir):
        if method not in data_file:
            continue
        n_files += 1
        tmp_instances = []
        data_dict = json.load(open(os.path.join(tool_data_dir, data_file)))
        ag = data_dict["answer_generation"]
        if not ag["valid_data"]:
            continue
        train_messages, query, functions = ag["train_messages"], ag["query"], ag["function"]
        for train_message in train_messages:
            conversations = []
            cur_react = ""
            for mid, m in enumerate(train_message):
                role, content = m["role"], m["content"]
                if role == "assistant":
                    inputs = process_assistant_reply(m)
                    if mid + 1 == len(train_message):          # last assistant = target
                        if "function_call" not in m:
                            cur_react = ""
                            break
                        if cur_react == "":
                            cur_react += "\nThought: "
                        cur_react += f"\nAction: {inputs['name']}"
                        cur_react += f"\nAction Input: {inputs['arguments']}"
                        conversations.append({"from": role, "value": cur_react})
                        cur_react = ""
                        tmp_instances.append({"id": f"Step {mid}: {query}",
                                              "conversations": conversations})
                        break
                    else:                                       # history assistant turns
                        if "function_call" not in m:
                            cur_react += f"\nThought: {inputs}"
                            continue
                        if cur_react == "":
                            cur_react += "\nThought: "
                        cur_react += f"\nAction: {inputs['name']}"
                        cur_react += f"\nAction Input: {inputs['arguments']}"
                        conversations.append({"from": role, "value": cur_react})
                        cur_react = ""
                else:
                    inputs = process_system_message(content, functions) if role == "system" else content
                    conversations.append({"from": role, "value": inputs})
                    cur_react = ""
        out_list.extend(tmp_instances)
    json.dump(out_list, open(output_file, "w"), indent=2, ensure_ascii=False)
    return n_files, len(out_list)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, d in VARIANTS.items():
        src = os.path.join(BASE, d)
        if not os.path.isdir(src):
            print(f"[skip] {name}: {src} missing"); continue
        of = os.path.join(OUT, f"toolllama_{name}.json")
        nf, ni = preprocess_rapidapi(src, METHOD, of)
        print(f"[{name}] {nf} answer files -> {ni} training instances -> {of}")
