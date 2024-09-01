from typing import List

templates = {
    "priming": "Sentence that consists of {src_len} words:\n{src}\nThe sentence that consists of {keep_len} words without the less important {del_len} words would be:\n",
}

def get_template(type:str = "priming"):
    return templates[type]

def apply_template(instances:List[dict], template:str):
    """
    instances: List
        [instance, instance, ... ]

    instance: dict
        {
            "src": sentence,
            "del_len": number of delete
        }
    """
    prompts = []
    for instance in instances:
        src = instance["src"]

        del_len = instance["del_len"]
        src_len = len(src.split())
        keep_len = src_len - del_len

        prompt = template.format(
            src_len=src_len,
            src=src,
            keep_len=keep_len,
            del_len=del_len
        )
        prompts.append(prompt)
    
    return prompts