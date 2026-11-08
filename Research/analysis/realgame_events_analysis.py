"""Stream normalized event observations; write aggregates only."""
import json, collections, itertools, math, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Research/data/realgame_events_analysis.json'
rows = collections.defaultdict(list)
counts = collections.Counter()
fields = collections.Counter()
special_fields = collections.Counter()
for line in (ROOT/'Background/RealGameData/data/observations.jsonl').open(encoding='utf8'):
    d=json.loads(line); t=d['event_type']; counts[t]+=1
    if t.startswith('event_'):
        rows[(d['participant_id'],d['run_id'],d['event_id'])].append(d)
        fields.update(d.keys())
    def walk(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if any(s in k.lower() for s in ('note','health','failed','version')): special_fields[k]+=1
                if isinstance(x,(dict,list)):walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(d)
names = [('強撃','强攻','強攻'),('幸運','幸运'),('爆発','暴发','暴發','爆發'),('体力','體力'),('集中','专注','專注'),('技巧','器用'),('必殺','绝招','絕招')]
def menu_type(m):
    s='|'.join(c.get('consequence','') for c in m['choices'])
    if ('140' in s and '90' in s and '音符' in s):return 108
    if ('音符' in s and '5' in s and '150' in s):return 103
    if ('音符' in s and '10' in s and ('稀有' in s or 'レア' in s)):return 104
    if ('音符' in s and '50%' in s and '30%' in s):return 120
    if '音符' in s and len(m['choices'])==3 and '5' in s:return 107
    if '音符' in s and '10' in s:return 114
    return None
groups=collections.defaultdict(list); signatures=collections.Counter(); mapped=collections.Counter(); duplicate_variants=0
for key,rs in rows.items():
    ms=[r for r in rs if r['event_type']=='event_menu_observed']
    if not ms:continue
    ms.sort(key=lambda x:x['timestamp']);m=ms[0]
    cs=[r for r in rs if r['event_type']=='event_choice']
    e=next(((r.get('model_mapping') or {}).get('event') for r in cs if (r.get('model_mapping') or {}).get('event')),None) or menu_type(m)
    if e not in (103,104,107,108,114,120):continue
    sig=tuple(c.get('consequence','') for c in m['choices']);signatures[(e,sig)]+=1
    duplicate_variants+=len({tuple(c.get('consequence','') for c in x['choices']) for x in rs if 'choices' in x})>1
    ids=[];unknown=[]
    if e in (107,108):
        for c in m['choices']:
            s=c.get('consequence','')
            matches=[i+1 for i,terms in enumerate(names) if any(t in s for t in terms)]
            if len(matches)==1:ids.append(matches[0])
            elif '音符' in s and ('ランダム' in s or '随机' in s or '隨機' in s):pass
            elif '30' in s:pass
            else:unknown.append(s)
    opts=[(r.get('model_mapping') or {}).get('option') for r in cs]
    mapped.update(str(x) for x in opts)
    groups[e].append(dict(participant=key[0],run=key[1],lang=m.get('lang_type'),preset=m.get('preset_id'),question=m.get('question'),pair=sorted(ids),unknown=unknown,menu_size=len(m['choices']),options=opts,confirmed=sum(r.get('operation_confirmed') is True for r in cs),choice_count=len(cs),click_count=sum(r['event_type']=='event_choice_click_observed' for r in rs)))
def wilson(x,n):
    if not n:return None
    z=1.95996398454; den=1+z*z/n; a=(x/n+z*z/(2*n))/den; b=z*math.sqrt(x/n*(1-x/n)/n+z*z/(4*n*n))/den
    return [max(0.0,a-b),min(1.0,a+b)]
def summary(gs):
    valid=[x for x in gs if len(x['pair'])==2 and len(set(x['pair']))==2 and not x['unknown']]
    pairs=collections.Counter(tuple(x['pair']) for x in valid)
    rng=random.Random(731); cluster={}
    for level in ('run','participant'):
        units=collections.defaultdict(list)
        for x in gs:units[x[level]].append(x)
        rates=[]; keys=list(units)
        if len(keys)>1:
            for _ in range(2000):
                sample=[x for k in rng.choices(keys,k=len(keys)) for x in units[k]]
                rates.append(sum(len(x['pair'])==2 and len(set(x['pair']))==2 and not x['unknown'] for x in sample)/len(sample))
            rates.sort()
        cluster[level]={'clusters':len(units),'bootstrap_95_ci': [rates[49],rates[1949]] if rates else None}
    n=len(valid); allpairs=list(itertools.combinations(range(1,8),2)); observed=[pairs[p] for p in allpairs]
    statistic=sum((v-n/21)**2/(n/21) for v in observed) if n else None
    # exact multinomial-null Monte Carlo: 21 unordered pairs, no fitted parameters.
    exceed=0; B=20000
    if n:
        for _ in range(B):
            sim=collections.Counter(rng.randrange(21) for _ in range(n))
            st=sum((sim[i]-n/21)**2/(n/21) for i in range(21))
            exceed+=st>=statistic-1e-10
    return {'events':len(gs),'valid_two_distinct':n,'two_distinct_wilson_95':wilson(n,len(gs)),'cluster_ci':cluster,'pairs':{','.join(map(str,k)):v for k,v in sorted(pairs.items())},'note_marginals':{str(i):sum(i in x['pair'] for x in valid) for i in range(1,8)},'uniform_21_pairs':{'n':n,'pearson_statistic':statistic,'monte_carlo_p':(exceed+1)/(B+1) if n else None,'simulations':B},'unmapped_consequences':dict(collections.Counter(s for x in gs for s in x['unknown']))}
result={'source_counts':dict(counts),'unique_event_instances':len(rows),'event_field_counts':dict(fields),'special_field_counts_entire_dataset':dict(special_fields),'changed_menu_signatures_same_instance':duplicate_variants,'selected_model_options':dict(mapped),'events':{str(e):{'instances':len(gs),'participants':len(set(x['participant'] for x in gs)),'runs':len(set(x['run'] for x in gs)),'choices':sum(x['choice_count'] for x in gs),'clicks':sum(x['click_count'] for x in gs),'operation_confirmed_true':sum(x['confirmed'] for x in gs)} for e,gs in groups.items()},'menus':{str(e):summary(groups[e]) for e in (107,108)},'menu_signature_counts':[{'event':e,'consequences':list(s),'n':n} for (e,s),n in signatures.items()], 'method':{'unit':'participant_id/run_id/event_id; earliest event_menu_observed only; reobserved and click excluded from independent denominator','static_ids':'not present; event_id is instance UUID; model_mapping is model inference, not server ID; note identities inferred from OCR consequence','outcomes':'No server confirmation or note inventory vectors; coin/potential observations cannot establish note deduction or 12002 branch. No deduction-kernel fit or branch CI.'}}
for e in (107,108):
    gs=groups[e]
    result['menus'][str(e)]['strata']={field:{str(k):summary([x for x in gs if x[field]==k]) for k in sorted(set(x[field] for x in gs),key=str)} for field in ('lang','preset','participant','question')}
    result['menus'][str(e)]['per_run']={k:{'n':sum(x['run']==k for x in gs),'pairs':dict(collections.Counter(','.join(map(str,x['pair'])) for x in gs if x['run']==k))} for k in sorted(set(x['run'] for x in gs))}
for e,m in result['menus'].items():
    m['marginal_cluster_95_ci']={}
    for level,units in [('run',m['per_run']),('participant',m['strata']['participant'])]:
        rng=random.Random(1024); keys=list(units); samples={str(i):[] for i in range(1,8)}
        for _ in range(4000):
            chosen=rng.choices(keys,k=len(keys)); den=sum(units[k].get('n',units[k].get('events',0)) for k in chosen)
            for i in samples:
                num=sum(n for k in chosen for p,n in units[k]['pairs'].items() if i in p.split(','))
                samples[i].append(num/den)
        m['marginal_cluster_95_ci'][level]={i:[sorted(v)[99],sorted(v)[3899]] for i,v in samples.items()}
# Marginal tests are exploratory event-level tests. Holm covers all fourteen
# event/note comparisons; participant bootstrap intervals are unadjusted.
def add_marginal_tests(result):
    tests=[]
    for event,m in result['menus'].items():
        n=m['valid_two_distinct']; p=2/7
        for note,x in m['note_marginals'].items():
            pmf=[math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)]
            exact=min(1.0,sum(v for v in pmf if v<=pmf[x]*(1+1e-12)))
            tests.append({'event':int(event),'note':int(note),'successes':x,'n':n,'null_p':p,'exact_two_sided_p':exact})
    previous=0.0
    for rank,t in enumerate(sorted(tests,key=lambda x:x['exact_two_sided_p'])):
        previous=max(previous,min(1.0,(len(tests)-rank)*t['exact_two_sided_p']))
        t['holm_adjusted_p']=previous
        t['holm_reject_0_05']=previous<0.05
    result['marginal_uniformity_tests']={'method':'Two-sided exact binomial probability ordering; Holm across fourteen event/note comparisons. Exploratory event-level tests do not account for within-participant dependence. Unadjusted cluster bootstrap CIs are descriptive, not evidence of multiplicity-adjusted deviation.','family_size':len(tests),'tests':tests,'any_holm_rejection_0_05':any(t['holm_reject_0_05'] for t in tests),'global_interpretation':'Retain the 21-pair multinomial Monte Carlo tests as the global tests; marginal tests cannot establish uniformity.'}
    def clamp(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k=='two_distinct_wilson_95' and x is not None:v[k]=[max(0.0,min(1.0,a)) for a in x]
                else:clamp(x)
        elif isinstance(v,list):
            for x in v:clamp(x)
    clamp(result)
add_marginal_tests(result)
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:result[k] for k in ('unique_event_instances','events','menus','special_field_counts_entire_dataset')},ensure_ascii=False))
