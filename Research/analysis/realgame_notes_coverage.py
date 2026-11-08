"""Targeted independent coverage summary; IDs/note prose are not inventory."""
import hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'Background/RealGameData/data/observations.jsonl'
numeric=re.compile(rb'(?<=[\[:,])\s*"?900(?:1[1-9]|2[0-3])"?\s*(?=[,}\]:])')
keys=Counter(); rec=defaultdict(list); kinds=Counter(); runs=defaultdict(set)
digest=hashlib.sha256(); records=0; byte_count=0
with p.open('rb') as f:
    for line in f:
        records+=1;byte_count+=len(line);digest.update(line)
        if numeric.search(line):
            x=json.loads(line);keys['rows_containing_standalone_note_id']+=1
            if len(rec['standalone_note_ids'])<10:rec['standalone_note_ids'].append(x['record_id'])
        if b'"context_run_recognition"' in line:
            x=json.loads(line);k=str(x.get('read_kind'));kinds[k]+=1
            runs[k].add((x.get('participant_id'),x.get('session_id'),x.get('run_id')))
# These five counts are manually reviewed conclusions for one exact source, not
# generic extraction logic. Any changed data must be reviewed before reusing them.
reviewed_source={'records':401040,'bytes':1597511628,'sha256':'87172a83cd8f05247e385992653ff09df52a8f1cf326ababcce777aa96b031dc'}
source_summary={'records':records,'bytes':byte_count,'sha256':digest.hexdigest()}
review_applies=source_summary==reviewed_source
result={'source':'Background/RealGameData/data/observations.jsonl','source_summary':source_summary,'manual_review':{'basis':'Manual inspection of complete recursive schema profile and note-related examples; five counts below are not automatically extracted.','reviewed_source':reviewed_source,'status':'applies_to_exact_source' if review_applies else 'requires_review','profile':'Research/data/realgame_notes_profile.json'},'counters':dict(keys),'record_ids':dict(rec),'recognition_read_kind_counts':dict(kinds),'recognition_read_kind_run_counts':{k:len(v) for k,v in runs.items()}}
for name in ('identified_complete_13_note_snapshots','isolated_battle_transitions','actual_lucky_marked_note_rewards','known_note_cross_99_transitions','observed_note_quantity_above_99'):
    result[name]=0 if review_applies else None
result['inference']=('Reviewed source: no note inventory or LuckyLevel field exists in the complete recursive key profile; note text is menu/shop/OCR description, not holdings. Potential owned/inventory_conflicts and wallet coin are separate state. No binomial denominator or confidence interval for the requested mechanisms is identifiable.' if review_applies else 'Source changed: the five manually reviewed counts are unknown (null); rerun schema profiling and review the new source before drawing note/Lucky/inventory conclusions.')
(ROOT/'Research/data/realgame_notes_coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
