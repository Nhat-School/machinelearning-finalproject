"""Create synthetic English forum posts for testing the app's CSV upload."""
import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
POSTS = [
    "I cannot upload my assignment. The deadline is tonight and the page keeps showing an error. Could the instructor help?",
    "Thank you for explaining the exercise. I understand the solution now.",
    "Could someone explain why precision and recall change when we adjust the classification threshold?",
    "The final quiz closes in thirty minutes, but every question loads as a blank page. I still cannot complete it.",
    "The quiz page was blank earlier, but clearing my browser cache fixed it. I have submitted my answers now.",
    "I enjoyed this week's lecture about decision trees.",
    "My assignment shows a zero even though I submitted it before the deadline. Could someone check the submission record?",
    "Does anyone want to form a study group for next week's exam?",
    "I have tried the installation instructions three times, but the required package still will not import. I cannot begin the lab.",
    "The installation problem is solved. I selected the correct Python environment and the lab runs successfully.",
    "Where can I find the slides for the lecture on support vector machines?",
    "I cannot access the course after resetting my password. The exam starts in an hour and I am still locked out.",
    "Here is a summary of the main ideas from today's lecture. Please add anything I missed.",
    "I am stuck on the probability exercise. Could someone show why the two events are not independent?",
    "The instructor has already corrected my missing grade. No further help is needed.",
    "My group partner withdrew today and our project is due tomorrow. Could the instructor advise how we should proceed?",
    "Is the optional reading about neural networks included in the exam syllabus?",
    "Yesterday I wrote, \"I cannot submit the assignment.\" The problem has since been fixed and my submission is complete.",
    "The dataset download link returns a permission error for our whole group. We still cannot download the files for tomorrow's lab.",
    "Thanks to everyone who shared their revision notes. Good luck with the exam!",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-api', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'FINAL_DEMO_POSTS.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['text'])
        writer.writerows([text] for text in POSTS)
    content = path.read_text(encoding='utf-8-sig')
    rows = list(csv.DictReader(io.StringIO(content)))
    assert [row['text'] for row in rows] == POSTS
    print(f'Created {path}: {len(rows)} synthetic demo posts, no ground-truth labels.')
    if args.check_api:
        base = 'http://127.0.0.1:8765'
        with urlopen(base + '/api/info', timeout=10) as response:
            key = json.load(response)['default_model_key']
        payload = json.dumps({'csv_text': content, 'model_key': key}).encode()
        request = Request(base + '/api/batch', data=payload, headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        assert result['model_key'] == key
        assert len(result['items']) == len(POSTS)
        assert [item['text'] for item in result['items']] == POSTS
        assert all(0 <= item['score'] <= 1 for item in result['items'])
        output = ROOT / '.cache/final_demo_csv_check.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(f'CSV API check passed: {len(POSTS)} predictions from {key}.')


if __name__ == '__main__':
    main()
