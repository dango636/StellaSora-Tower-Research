import json, collections, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
counts=collections.Counter(); shelves={}; repeats=collections.Counter(); examples={}; meta=collections.Counter()
for line in (ROOT/'Background/RealGameData/data/observations.jsonl').open(encoding='utf-8'):
 r=json.loads(line); counts[r['event_type']]+=1
 if r['event_type']!='shop_observed' or r.get('shop_event_type')!='stock_snapshot': continue
 p=r.get('payload',{}); slots=p.get('slots',[])
 items=r.get('stock_snapshot') or [dict(s.get('item') or s, price=s.get('effective_price',(s.get('item') or s).get('price'))) for s in slots]
 cluster=tuple(r.get(k) for k in ('participant_id','session_id','run_id'))
 gen=r.get('shelf_generation_id') or r.get('stock_generation_id')
 key=cluster+(r.get('shop_id') or r.get('shop_visit_id') or r.get('shop_entry_id'),gen)
 repeats['snapshots']+=1
 if not gen: repeats['missing_generation']+=1; continue
 compact=[(i.get('internal_name'),i.get('quantity'),i.get('effective_price',i.get('price')),i.get('grid_num',i.get('slot')),i.get('parse_complete',True)) for i in items]
 sig=tuple(compact)
 if key in shelves:
  repeats['repeated_generation']+=1
  if sig!=shelves[key]['sig']: repeats['changed_same_generation']+=1
  continue
 state=r.get('state',{}); ctx=p.get('cached_shop_context',{})
 shelves[key]=dict(sig=sig,cluster=cluster,lang=r.get('lang_type'),preset=r.get('preset_id'),record_id=r['record_id'],shop_type=state.get('shop_type',ctx.get('shop_type')),basis=r.get('shelf_generation_basis'),remaining=r.get('refresh_remaining',state.get('refresh_remaining',ctx.get('refresh_remaining'))),complete=p.get('all_slots_parsed',True),repeated=r.get('repeated_scan',False))

def ci_cluster(rows,func):
 groups=collections.defaultdict(list)
 for s in rows: groups[s['cluster']].append(func(s))
 a=np.array([[sum(v),len(v)] for v in groups.values()],float); estimate=a[:,0].sum()/a[:,1].sum()
 if len(a)<2:return dict(estimate=estimate,n=len(rows),clusters=len(a),ci95=None)
 rng=np.random.default_rng(74021); ids=rng.integers(len(a),size=(5000,len(a))); b=a[ids].sum(axis=1); boot=b[:,0]/b[:,1]
 return dict(estimate=estimate,n=len(rows),clusters=len(a),ci95=np.quantile(boot,[.025,.975]).tolist(),ci99=np.quantile(boot,[.005,.995]).tolist())

valid=[]; excluded=collections.Counter(); price=collections.Counter(); bylang={}; abnormal=collections.Counter(); abnormal_examples={}
for s in shelves.values():
 if len(s['sig'])!=8: excluded['not_8_items']+=1;continue
 if not s['complete'] or not all(x[4] for x in s['sig']):excluded['incomplete_parse']+=1;continue
 if any(not (x[0]=='potential_drink' or (x[0] or '').startswith('melody_of_')) for x in s['sig']):
  excluded['unknown_or_other_item']+=1
  for x in s['sig']:
   if not (x[0]=='potential_drink' or (x[0] or '').startswith('melody_of_')):
    a=str(('unknown_or_other_item',x[0],x[1],x[2]));abnormal[a]+=1;abnormal_examples.setdefault(a,[])
    if len(abnormal_examples[a])<3:abnormal_examples[a].append(s['record_id'])
  continue
 if any(x[1] not in (1,5,15) for x in s['sig']):excluded['other_quantity']+=1;continue
 if any(x[2] not in ({100,160,200} if x[0]=='potential_drink' else ({45,72,90} if x[1]==5 else {200,320,400})) for x in s['sig']):
  excluded['off_grid_price']+=1;examples.setdefault('off_grid_price',s['record_id'])
  for x in s['sig']:
   if x[2] not in ({100,160,200} if x[0]=='potential_drink' else ({45,72,90} if x[1]==5 else {200,320,400})):
    a=str(('off_grid_price',x[0]=='potential_drink',x[1],x[2]));abnormal[a]+=1;abnormal_examples.setdefault(a,[])
    if len(abnormal_examples[a])<3:abnormal_examples[a].append(s['record_id'])
  continue
 ratios=[x[2]/(200 if x[0]=='potential_drink' else 90 if x[1]==5 else 400) for x in s['sig']]
 s['discount']=sum(v<1 for v in ratios);s['half']=sum(v==.5 for v in ratios);s['large']=sum(x[1]==15 for x in s['sig']);s['small']=sum(x[1]==5 for x in s['sig']);s['drinks']=sum(x[0]=='potential_drink' for x in s['sig']);valid.append(s)
 for x in s['sig']:price[(x[0]=='potential_drink',x[1],x[2])]+=1
 examples.setdefault('valid',s['record_id'])

def summary(rows):
 if not rows:return {}
 return dict(n=len(rows),clusters=len(set(s['cluster'] for s in rows)),discount_counts=dict(collections.Counter(s['discount'] for s in rows)),half_counts=dict(collections.Counter(s['half'] for s in rows)),large_counts=dict(collections.Counter(s['large'] for s in rows)),drink_counts=dict(collections.Counter(s['drinks'] for s in rows)),large_any=ci_cluster(rows,lambda s:s['large']>0),mean_large=ci_cluster(rows,lambda s:s['large']),discount_four=ci_cluster(rows,lambda s:s['discount']==4),half_atmost2=ci_cluster(rows,lambda s:s['half']<=2),mean_within_shelf_large_fraction=ci_cluster(rows,lambda s:s['large']/max(1,s['small']+s['large'])))

out=dict(event_counts=dict(counts),dedup=dict(repeats),unique_generations=len(shelves),exclusions=dict(excluded),valid=summary(valid),by_language={l:summary([s for s in valid if s['lang']==l]) for l in sorted(set(s['lang'] for s in valid))},by_shop_type={str(l):summary([s for s in valid if s['shop_type']==l]) for l in set(s['shop_type'] for s in valid)},by_preset={str(l):summary([s for s in valid if s['preset']==l]) for l in set(s['preset'] for s in valid)},price_counts={str(k):v for k,v in price.items()},generation_basis=dict(collections.Counter(s['basis'] for s in shelves.values())),examples=examples)
# Compare observable whole-shelf large occurrence with 4/5 and independent-slot
# fitted marginal. The latter is a distinct joint model, not an alternative proof.
notes=sum(s['small']+s['large'] for s in valid);large=sum(s['large'] for s in valid);p=large/notes
out['independent_slot_model']=dict(fitted_large_slot_probability=p,expected_shelves_multilarge=sum(1-(1-p)**(s['small']+s['large'])-(s['small']+s['large'])*p*(1-p)**(s['small']+s['large']-1) for s in valid),observed_multilarge=sum(s['large']>1 for s in valid))
first={}
for s in valid:first.setdefault(s['cluster'],s)
logprob=sum(math.log((1-p)**(s['small']+s['large'])+(s['small']+s['large'])*p*(1-p)**(s['small']+s['large']-1)) for s in first.values())
out['independent_slot_model']['one_shelf_per_cluster_probability_zero_multilarge']=math.exp(logprob)
ns=[s['small']+s['large'] for s in first.values()]; K=sum(s['large'] for s in first.values()); coeff=[1]+[0]*K
for n in ns:
 for k in range(K,0,-1):coeff[k]+=coeff[k-1]*n
out['independent_slot_model']['exact_conditional_one_shelf_per_cluster']=dict(shelves=len(ns),note_slots=sum(ns),large_slots=K,probability_no_collision_given_large_total=math.exp(math.log(coeff[K])-math.log(math.comb(sum(ns),K))),null='Common iid Bernoulli large/small across note slots, conditioning on total large slots removes fitted p; assumes independent run clusters.')
out['zero_violation_cluster_bound']=dict(clusters=len(first),one_sided95_upper_fraction_clusters_with_violation=1-.05**(1/len(first)),note='One-sided binomial zero-event bound at run-cluster level; assumes independent sampled clusters, not a shelf-level bound. Bootstrap [1,1] for invariant events is descriptive only.')
out['strict_identity']=summary([s for s in valid if s['basis'] not in (None,'stock_contents_changed_after_ambiguous_refresh')])
out['by_generation_basis']={str(b):summary([s for s in valid if s['basis']==b]) for b in set(s['basis'] for s in valid)}
out['note_type_quantity_counts']={str(k):v for k,v in collections.Counter((x[0],x[1]) for s in valid for x in s['sig'] if (x[0] or '').startswith('melody_of_')).items()}
out['unfiltered_shapes']=dict(item_counts=dict(collections.Counter(len(s['sig']) for s in shelves.values())),note_quantities=dict(collections.Counter(str(x[1]) for s in shelves.values() for x in s['sig'] if (x[0] or '').startswith('melody_of_'))))
participant_rows=[dict(s,cluster=(s['cluster'][0],)) for s in valid]
out['participant_cluster_sensitivity']=ci_cluster(participant_rows,lambda s:s['large']>0)
out['language_large_counts']={l:dict(n=sum(s['lang']==l for s in valid),big=sum(s['large']>0 for s in valid if s['lang']==l),mean=sum(s['large']>0 for s in valid if s['lang']==l)/sum(s['lang']==l for s in valid)) for l in sorted(set(s['lang'] for s in valid))}
out['zero_violation_cluster_bound']['one_sided99_upper_fraction_clusters_with_violation']=1-.01**(1/len(first))
out['exclusion_audit']=dict(selection_uses_discount_count=False,selection_uses_large_count=False,selection_rule='Exactly 8 parsed items; item is potential_drink or melody_of_*; quantity in 1/5/15; effective price belongs to externally specified price grid for its type and quantity. No check of discount count, half-price count, or count of large bundles is used for inclusion.',abnormal_item_counts=dict(abnormal),record_id_examples=abnormal_examples)
unknown=len(shelves)-len(valid); big=sum(s['large']>0 for s in valid)
out['worst_case_large_any_identification']=dict(all_shelves=len(shelves),known_shelves=len(valid),known_big_shelves=big,unknown_shelves=unknown,lower=big/len(shelves),upper=(big+unknown)/len(shelves),note='Treat every excluded shelf as unknown presence of a large bundle: lower assigns all zero, upper assigns all one. This is an identification interval, not a confidence interval.')
normalized=[]; corrections=collections.Counter(); corrected_shelves=collections.Counter(); normalized_excluded=collections.Counter()
for raw in shelves.values():
 s=dict(raw); sig=[]; changed=set()
 for x in raw['sig']:
  name,q,price,slot,parsed=x
  if name=='强攻之音':name='melody_of_pummel';corrections['name_pummel']+=1;changed.add('name_pummel')
  if q==5 and price in (9045,9072):
   rule=f'price_{price}_to_{45 if price==9045 else 72}';price=45 if price==9045 else 72;corrections[rule]+=1;changed.add(rule)
  sig.append((name,q,price,slot,parsed))
 for rule in changed:corrected_shelves[rule]+=1
 if changed:corrected_shelves['any']+=1
 s['sig']=tuple(sig)
 if len(sig)!=8 or not s['complete'] or not all(x[4] for x in sig):normalized_excluded['shape_or_parse']+=1;continue
 if any(not (x[0]=='potential_drink' or (x[0] or '').startswith('melody_of_')) for x in sig):normalized_excluded['unknown_item']+=1;continue
 if any(x[1] not in (1,5,15) for x in sig):normalized_excluded['quantity']+=1;continue
 if any(x[2] not in ({100,160,200} if x[0]=='potential_drink' else ({45,72,90} if x[1]==5 else {200,320,400})) for x in sig):normalized_excluded['price']+=1;continue
 ratios=[x[2]/(200 if x[0]=='potential_drink' else 90 if x[1]==5 else 400) for x in sig]
 s.update(discount=sum(v<1 for v in ratios),half=sum(v==.5 for v in ratios),large=sum(x[1]==15 for x in sig),small=sum(x[1]==5 for x in sig),drinks=sum(x[0]=='potential_drink' for x in sig));normalized.append(s)
nf={}
for s in normalized:nf.setdefault(s['cluster'],s)
nns=[s['small']+s['large'] for s in nf.values()];nk=sum(s['large'] for s in nf.values());nc=[1]+[0]*nk
for n in nns:
 for k in range(nk,0,-1):nc[k]+=nc[k-1]*n
out['confirmed_analysis_normalization']=dict(sources=['客户端文本配置：SubNoteSkill.90011.1=强攻之音，对应音符90011。', '既有价格核对确认的5枚音符包OCR连读修正规则：9045→45、9072→72；不按待检验的整架约束纳入样本。'],does_not_modify_total_data=True,selection_uses_discount_or_large_counts=False,corrected_item_counts=dict(corrections),corrected_shelf_counts=dict(corrected_shelves),remaining_exclusions=dict(normalized_excluded),valid=summary(normalized),participant_cluster_sensitivity=ci_cluster([dict(s,cluster=(s['cluster'][0],)) for s in normalized],lambda s:s['large']>0),strict_identity=summary([s for s in normalized if s['basis'] not in (None,'stock_contents_changed_after_ambiguous_refresh')]),by_language={l:dict(summary([s for s in normalized if s['lang']==l]),big=sum(s['large']>0 for s in normalized if s['lang']==l)) for l in sorted(set(s['lang'] for s in normalized))},exact_conditional_one_shelf_per_cluster=dict(shelves=len(nns),note_slots=sum(nns),large_slots=nk,probability_no_collision_given_large_total=math.exp(math.log(nc[nk])-math.log(math.comb(sum(nns),nk)))),zero_violation_run_bound=dict(clusters=len(nf),upper95=1-.05**(1/len(nf)),upper99=1-.01**(1/len(nf))))
pf={}
for s in normalized:pf.setdefault(s['cluster'][0],s)
pns=[s['small']+s['large'] for s in pf.values()];pk=sum(s['large'] for s in pf.values());pc=[1]+[0]*pk
for n in pns:
 for k in range(pk,0,-1):pc[k]+=pc[k-1]*n
out['confirmed_analysis_normalization']['exact_conditional_one_shelf_per_participant']=dict(participants=len(pf),shelves=len(pns),note_slots=sum(pns),large_slots=pk,probability_no_collision_given_large_total=math.exp(math.log(pc[pk])-math.log(math.comb(sum(pns),pk))),selection='First normalized shelf for each participant in observations.jsonl file order, not chronological reordering.',null='Common iid Bernoulli large/small across note slots; conditional on the total large count.',scope='Only 18 observed participants; this sensitivity test does not identify broader player or server populations.')
out['caveats']=['Intervals resample participant/session/run clusters; observation and preset selection are not population random sampling.','Generation IDs are local UI identities, not server IDs; content is never used to merge different generations.','First snapshot per generation is retained; later same-generation changes are counted, not treated as fresh shelves.','Tower/disc IDs and phase are not explicit shelf fields; preset/language strata cannot identify causal disc/tower dependence.']
dest=ROOT/'Research/data/realgame_shop_stats.json';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=True))
