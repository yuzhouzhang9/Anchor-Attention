import json
import os
def get_test_data(data_name = "niah_multikey_3"):
    data_path = os.path.join(f"data/test_data", f"{data_name}.jsonl")
    res = []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                # 解析每一行的 JSON 对象
                data = json.loads(line.strip())
                data["prompt"] = data["input"]
                res.append(data)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file {data_path}.")
    return res,32