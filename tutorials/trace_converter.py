import argparse
from pathlib import Path
from typing import Dict
import json
import copy

def hash_to_range(num):
    hash_value = hash(num) % 10001
    return hash_value

def read_json(path: Path) -> Dict:
    config = {}
    with open(path, "r") as fp:
        config = json.load(fp)
    return config


def write_json(path: Path, config: Dict):
    with open(path, "w") as fp:
        json.dump(config, fp, indent=4)

EVENT_TEMPLATE = {
    "args": {},
    "cat": "", # The event categorie, __metadata or event
    "name": "default", # The event type, M: Metadata Events, X: Complete Events
    "ph": "M",
    "pid": 0,
    "tid": 0,
    "ts": 0, # The tracing clock timestamp of the event, microsecond us
}

swapper_data = {
    "args": {
        "name": "swapper"
    },
    "cat": "__metadata",
    "name": "thread_name",
    "ph": "M",
    "pid": 0,
    "tid": 0,
    "ts": 0
}


def parse_trace(trace_file: Path, output_file: Path, trace_template: Path):
    template = trace_template
    config = read_json(template)
    meta_datas = [swapper_data, ]
    events = []

    def find_event(name, thr):
        nonlocal events
        for e in events:
            if e["name"] == name and e["tid"] == thr:
                return e
        assert False, f"[{thr}] {name} not exists!"

    start_time = 0
    end_time = 0
    tids = []

    with open(trace_file, "r") as fp:
        for line in fp.readlines():
            line = line.replace("\n", "").strip()
            if "start_time:" in line:
                start_time = int(line.split(" ")[-1])
            elif "end_time:" in line:
                end_time = int(line.split(" ")[-1])
            else:
                line = line.split(", ")
                thr, name, ts = line
                thr = hash_to_range(int(thr.split(": ")[-1]))
                name = name.split(": ")[-1]
                ts = int(ts.split(": ")[-1])
                if thr not in tids:
                    tids.append(thr)
                    cfg = copy.deepcopy(EVENT_TEMPLATE)
                    cfg["args"]["name"] = "llama-cli"
                    cfg["cat"] = "__metadata"
                    cfg["name"] = "thread_name"
                    cfg["ph"] = "M"
                    cfg["pid"] = tids[0]
                    cfg["tid"] = thr
                    cfg["ts"] = 0
                    meta_datas.append(cfg)
                
                if "end_" in name:
                    old_name = name.removeprefix("end_")
                    start_event = find_event(old_name, thr)
                    start_event["dur"] = ts - start_time - start_event["ts"]
                else:
                    ecfg = copy.deepcopy(EVENT_TEMPLATE)
                    ecfg["cat"] = "event"
                    ecfg["name"] = name
                    ecfg["ph"] = "X"
                    ecfg["pid"] = tids[0]
                    ecfg["tid"] = thr
                    ecfg["ts"] = ts - start_time
                    events.append(ecfg)

    config["traceEvents"].extend(meta_datas)
    config["traceEvents"].extend(events)
    write_json(output_file, config)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--trace-file", required=True, type=Path)
    parser.add_argument("-o", "--output-file", type=Path, default=Path("./out_trace.json"))
    parser.add_argument("-t", "--trace-template", required=True, type=Path)
    args = parser.parse_args()
    parse_trace(args.trace_file, args.output_file, args.trace_template)

main()