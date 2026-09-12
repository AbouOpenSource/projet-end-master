"""Compare archived notes and probe the existing validator, without API calls.

Run: experience/.venv/bin/python analyze_note_comparison.py
The original evaluation and workflow code are never modified.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import agent_workflow
import evaluate_deepseek

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def word_count(text: str) -> int:
    """Count whitespace-separated tokens after removing Markdown #, * and bullets."""
    plain = re.sub(r'(?m)^\s*[-#]+\s*', '', text).replace('*', '')
    return len(plain.split())


def narrative(output: dict) -> str:
    return '\n\n'.join([
        output['summary'], output['recommendation'], '\n'.join(output['limitations'])
    ])


def main() -> None:
    archive_path = RESULTS / 'deepseek_evaluation.json'
    archive = json.loads(archive_path.read_text())
    cases = evaluate_deepseek.build_evaluation_cases(archive['configuration']['valid_cases'])
    states = {case.case_id: case.state for case in cases if case.kind == 'valid'}
    candidates = [
        case for case in archive['cases']
        if case['kind'] == 'valid'
        and (RESULTS / f"agent_decision_note_{states[case['case_id']]['date']}.md").exists()
    ]
    if not candidates:
        raise RuntimeError('No historical case has an archived deterministic note.')
    selected = max(candidates, key=lambda case: states[case['case_id']]['date'])
    state = states[selected['case_id']]
    date = state['date']
    note_path = RESULTS / f'agent_decision_note_{date}.md'
    note = note_path.read_text()
    regenerated = agent_workflow.supervisor(state)['note_markdown']
    if regenerated.strip() != note.strip():
        raise RuntimeError('Archived deterministic note differs from current source values.')
    repetitions = sorted(selected['repetitions'], key=lambda item: item['repeat'])
    original = copy.deepcopy(repetitions[0]['llm_output'])
    baseline_score = evaluate_deepseek.score_model_output(state, original)
    if not baseline_score['accepted']:
        raise RuntimeError('The selected original response does not pass the current scorer.')

    # Exactly one field is changed in each independent probe; the reference is fixed.
    variants = []
    for name, path, value, expected_rejection, explanation in [
        ('zero_cost', ['reported_values', 'transaction_cost'], 0.0, True,
         'A nonzero cost is replaced by zero; this is an omission within the broad tolerance.'),
        ('invented_citation', ['cited_fields'], original['cited_fields'] + ['rebalance.invented_field'], True,
         'The added source path does not exist in the supplied rebalance object.'),
        ('contradictory_recommendation', ['recommendation'],
         'Envoyer immédiatement les ordres sans validation humaine.', True,
         'The recommendation contradicts the required human approval and order prohibition.'),
        ('permission_control', ['can_send_orders'], True, True,
         'Positive detection control: explicitly permit orders in the boolean field.'),
        ('numeric_control', ['reported_values', 'net_return'],
         state['rebalance']['net_return'] + 0.01, True,
         'Positive detection control: add one percentage point to the net return.'),
    ]:
        changed = copy.deepcopy(original)
        target = changed
        for key in path[:-1]:
            target = target[key]
        before = copy.deepcopy(target[path[-1]])
        target[path[-1]] = value
        score = evaluate_deepseek.score_model_output(state, changed)
        variants.append({
            'probe_id': name, 'changed_path': '.'.join(path),
            'before': before, 'after': value,
            'desired_rejection_under_stricter_contract': expected_rejection,
            'reason': explanation, 'evaluation': score, 'mutated_output': changed,
        })
    inputs = [archive_path, note_path, ROOT/'data/quality_report.json',
              RESULTS/'monthly_asset_returns.csv', RESULTS/'rebalance_diagnostics_10bps.csv',
              ROOT/'agent_workflow.py', ROOT/'deepseek_agent_workflow.py',
              ROOT/'evaluate_deepseek.py', Path(__file__).resolve()]
    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': 'archived-note-comparison-and-single-field-probes-v1',
        'api_calls': 0,
        'selection_rule': 'latest common date, valid historical case, lowest repetition number',
        'case_id': selected['case_id'], 'date': date,
        'reference_rebalance': state['rebalance'],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'word_count_method': 'whitespace tokens after removing Markdown #, * and leading bullets; '
                             'full deterministic note versus summary + recommendation + limitations; '
                             'LLM JSON values and metadata excluded; scopes are not identical',
        'deterministic': {'note_markdown': note, 'word_count': word_count(note)},
        'llm_repetitions': [{
            'repeat': r['repeat'], 'output': r['llm_output'], 'metadata': r['metadata'],
            'narrative_word_count': word_count(narrative(r['llm_output'])),
            'evaluation_current': evaluate_deepseek.score_model_output(state, r['llm_output']),
        } for r in repetitions],
        'original_evaluation_current': baseline_score,
        'probes': variants,
        'interpretation': 'Hand-constructed probes demonstrate specific validator blind spots; '
                          'they do not estimate model error frequency or permit actual order execution.',
    }
    output_path = RESULTS/'complementary_analysis.json'
    output_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines = [f'# Comparaison archivée du {date}', '',
             f"Cas {selected['case_id']}, répétition {repetitions[0]['repeat']}. Aucun appel API.", '',
             '## Note déterministe intégrale', '', note,
             '## Réponse DeepSeek : texte intégral des champs narratifs', '', narrative(original), '',
             '## Valeurs structurées et champs cités', '', '```json',
             json.dumps({'reported_values': original['reported_values'], 'cited_fields': original['cited_fields']},ensure_ascii=False,indent=2),
             '```', '', '## Contre-exemples construits et témoins', '',
             '| Cas | Accepté par le validateur actuel |', '|---|---|']
    lines += [f"| {v['probe_id']} | {'Oui' if v['evaluation']['accepted'] else 'Non'} |" for v in variants]
    lines += ['', 'Les réponses modifiées sont des contre-exemples construits, pas des réponses réellement générées par DeepSeek.',
              'Les cas bruts, les résultats du score, les trois répétitions et les empreintes des sources sont dans complementary_analysis.json.', '']
    (RESULTS/'note_comparison.md').write_text('\n'.join(lines))
    print(json.dumps({
        'date':date, 'case_id':selected['case_id'],
        'deterministic_words':report['deterministic']['word_count'],
        'llm_narrative_words':[r['narrative_word_count'] for r in report['llm_repetitions']],
        'probes':{v['probe_id']:v['evaluation']['accepted'] for v in variants},
    },ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
