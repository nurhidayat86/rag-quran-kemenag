import json
from pathlib import Path

def parse_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main():
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / 'quran-complete.json'
    data = parse_json(file_path)
    print(data.index)

if __name__ == '__main__':
    main()