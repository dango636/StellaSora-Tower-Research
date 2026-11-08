"""Stream the sole real-game dataset; export note-mechanism coverage, not raw rows."""
import json,re
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'Background/RealGameData/data/observations.jsonl'
OUT=ROOT/'Research/data/realgame_notes_profile.json'
events=Counter(); paths=Counter(); examples=defaultdict(list); types=defaultdict(Counter)
runs=set(); eventruns=defaultdict(set); topkeys=defaultdict(Counter)
interesting=re.compile(r'note|lucky|sub.?skill|inventory|bag|9001[1-9]|9002[0-3]|音符|ノーツ|楽譜',re.I)
def walk(v,path,rid,etype):
    if isinstance(v,dict):
        for k,x in v.items():
            p=path+'.'+k if path else k
            if interesting.search(k):
                paths[p]+=1; types[p][type(x).__name__]+=1
                if len(examples[p])<4:
                    examples[p].append({'record_id':rid,'event_type':etype,'value_type':type(x).__name__})
            walk(x,p,rid,etype)
    elif isinstance(v,list):
        for x in v: walk(x,path+'[]',rid,etype)
    elif isinstance(v,str) and interesting.search(v):
        p=path+' [matching string]';paths[p]+=1
        if len(examples[p])<4:examples[p].append({'record_id':rid,'event_type':etype,'value_type':'str'})
with SOURCE.open(encoding='utf-8') as f:
    for n,line in enumerate(f,1):
        row=json.loads(line); et=row.get('event_type'); rid=row['record_id']
        events[et]+=1;run=tuple(row.get(k) for k in ('participant_id','session_id','run_id'))
        runs.add(run);eventruns[et].add(run);topkeys[et].update(row.keys())
        walk(row,'',rid,et)
result={'source':SOURCE.relative_to(ROOT).as_posix(),'records':n,'run_groups':len(runs),'events':dict(events),'event_run_groups':{k:len(v) for k,v in eventruns.items()},'top_keys_by_event':{k:dict(v) for k,v in topkeys.items()},'note_related_paths':dict(paths),'value_types':{k:dict(v) for k,v in types.items()},'examples':dict(examples)}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'records':n,'runs':len(runs),'events':dict(events),'paths':dict(paths)},ensure_ascii=False))
