import sys
import json

def check_file(jsonl_file):
    with open(jsonl_file) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except Exception as e:
                print(f"Malformed JSON at line {i} of {jsonl_file}: {e}")
                return False
    print(f"{jsonl_file} is valid.")
    return True

if __name__ == "__main__":
    import glob
    files = glob.glob('eos_*.jsonl') + glob.glob('openai_narratives_*.jsonl')
    for fname in files:
        check_file(fname)
