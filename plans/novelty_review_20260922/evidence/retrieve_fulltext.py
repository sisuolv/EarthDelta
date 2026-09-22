"""Download selected arXiv primary texts; no claim of review from retrieval alone."""
import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
IDS = ['2509.22020v2', '2608.02528v1', '2605.14546v1', '2609.12278v1',
       '2608.22370v2', '2603.19325v1', '2609.08412v1', '2606.14222v1',
       '2609.03937v1', '2605.22222v3', '2609.17042v1', '2609.18381v1',
       '2609.10954v1', '2609.22237v1', '2609.12296v1', '2603.15990v1',
       '2605.17250v1', '2607.02829v1']


def fetch(paper):
    url = 'https://arxiv.org/html/' + paper
    record = {'id': paper, 'url': url, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=40)
        record['status'] = response.status_code
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        article = soup.find('article') or soup
        for tag in article(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        lines = [line.strip() for line in article.get_text('\n').splitlines() if line.strip()]
        body = '\n'.join(lines)
        (HERE / (paper + '.html')).write_text(response.text)
        (HERE / (paper + '.txt')).write_text(body)
        record.update(characters=len(body), file=paper + '.txt', scope='FULL_TEXT_RETRIEVED_NOT_AUTOMATICALLY_REVIEWED')
    except Exception as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, IDS))
    (HERE / 'fulltext_retrieval.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
