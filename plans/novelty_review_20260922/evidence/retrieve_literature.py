"""Record dated primary-source literature searches for the EarthDelta review."""
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
NS = {'a': 'http://www.w3.org/2005/Atom'}
QUERIES = {
    'weather_adaptation': '(all:"weather forecasting" OR all:"weather foundation") AND (all:adapter OR all:adaptation OR all:correction OR all:routing)',
    'parameter_response': '(all:"parameter response" OR all:"intervention response" OR all:"counterfactual prediction" OR all:"edit response") AND (all:model OR all:neural)',
    'adapter_utility': '(all:LoRA OR all:adapter) AND (all:"predicted utility" OR all:"response prediction" OR all:"budget-aware" OR all:"test-time fusion" OR all:counterfactual)',
    'dynamics_adaptation': '(ti:"world model" OR all:"PDE foundation") AND (all:adaptation OR all:intervention OR all:corrector)',
    'forecast_errors': '(all:"forecast error" OR all:"error correction") AND (all:"test-time" OR all:memory OR all:context) AND (all:forecast OR all:dynamics)',
    'merging_response': '(all:"model merging" OR all:"weight space") AND (all:response OR all:surrogate OR all:predicting)',
    'forecast_sensitivity': '(all:"forecast sensitivity" OR all:"observation impact" OR all:"proactive quality")',
    'weather_selection': '(all:weather OR all:atmospheric) AND (all:"model selection" OR all:"mixture of experts" OR all:"weight perturbation" OR all:"concept tuning")',
}
IDS = ['2608.02528','2609.17042','2509.22020','2405.13063','2605.14546',
       '2609.12278','2603.15990','2608.22370','2609.08412','2603.19325',
       '2606.14222','2609.03937','2605.22222','2609.12296','2606.08365',
       '2510.09734','2608.09948','2608.07053','2608.29998','2606.19549',
       '2605.17250','2506.23424','2007.00016','1710.08005']


def parse(text):
    feed = ET.fromstring(text)
    rows = []
    for entry in feed.findall('a:entry', NS):
        get = lambda name: ' '.join(entry.findtext('a:' + name, '', NS).split())
        rows.append({'id': get('id'), 'title': get('title'), 'published': get('published'),
                     'updated': get('updated'), 'abstract': get('summary'),
                     'authors': [a.findtext('a:name', '', NS) for a in entry.findall('a:author', NS)],
                     'links': [link.attrib for link in entry.findall('a:link', NS)]})
    return rows


def main():
    records = []
    tasks = [('known_neighbors', {'id_list': ','.join(IDS), 'max_results': len(IDS)})]
    for name, query in QUERIES.items():
        tasks.append((name, {'search_query': query + ' AND submittedDate:[202501010000 TO 202609222359]',
                             'start': 0, 'max_results': 35,
                             'sortBy': 'submittedDate', 'sortOrder': 'descending'}))
    for name, params in tasks:
        row = {'query_name': name, 'params': params, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
        try:
            response = requests.get('https://export.arxiv.org/api/query', params=params, timeout=35)
            row.update(url=response.url, status=response.status_code)
            response.raise_for_status()
            (HERE / (name + '.xml')).write_text(response.text)
            row['results'] = parse(response.text)
            print(name, len(row['results']), flush=True)
            for paper in row['results']:
                print(' ', paper['id'].rsplit('/', 1)[-1], paper['title'], flush=True)
        except Exception as exc:
            row['error'] = type(exc).__name__ + ': ' + str(exc)
            print(name, row['error'], flush=True)
        records.append(row)
        (HERE / 'arxiv_searches.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
        time.sleep(3)


if __name__ == '__main__':
    main()
