"""Profile real-game note evidence coverage without copying raw observations."""
from pathlib import Path
from collections import Counter,defaultdict
import json,re,importlib.util
ROOT=Path(__file__).resolve().parents[2];src=ROOT/'Background/RealGameData/data/observations.jsonl'
types=Counter();keys=defaultdict(Counter);note_paths=Counter();examples={}; samples={};langs=Counter();runs=set();participants=set();n=0
for line in src.open(encoding='utf-8'):
 r=json.loads(line);n+=1;t=r.get('event_type','unknown');types[t]+=1;keys[t].update(r.keys());langs[str(r.get('lang_type'))]+=1;runs.add((r.get('participant_id'),r.get('session_id'),r.get('run_id')));participants.add(r.get('participant_id'))
 if t not in samples and t not in {'context_run_task','context_run_recognition','lifecycle_observed'}:samples[t]={k:type(v).__name__ for k,v in r.items()}
 stack=[('',r)]
 while stack:
  path,v=stack.pop()
  if isinstance(v,dict):
   for k,x in v.items():
    p=path+'.'+k if path else k
    if re.search(r'note|音符|lucky|critical',k,re.I):
     tag=t+':'+p+':'+type(x).__name__;note_paths[tag]+=1
     if tag not in examples:examples[tag]={'record_id':r.get('record_id'),'value_type':type(x).__name__}
    if isinstance(x,(dict,list)):stack.append((p,x))
  elif isinstance(v,list):
   for x in v:
    if isinstance(x,(dict,list)):stack.append((path+'[]',x))
 if n%100000==0:print('records',n,flush=True)
out={'source':'Background/RealGameData/data/observations.jsonl','source_bytes':src.stat().st_size,'record_count':n,'event_types':types,'languages':langs,'grouped_run_keys':len(runs),'participants':len(participants),'fields_by_event_type':{k:dict(v) for k,v in keys.items()},'field_types_examples':samples,'note_related_field_paths':dict(note_paths),'note_field_examples':examples,'dependencies':{x:bool(importlib.util.find_spec(x)) for x in ['numpy','scipy','statsmodels']}}
(ROOT/'Research/data/realgame_note_schema.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('DONE',n,'note_paths',len(note_paths),flush=True)
