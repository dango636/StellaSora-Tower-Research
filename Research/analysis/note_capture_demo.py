"""Create a clearly synthetic import demonstration; not a native Maa database."""
import argparse
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3

from note_capture_import import FIELDS
from note_capture_quality import FLAGS


def create(directory):
    directory=Path(directory)
    if directory.exists():
        raise ValueError('demo_output_already_exists')
    directory.mkdir(parents=True)
    row={'sequence':1,'event_type':'exchange','option_id':10301,
         'operation_confirmed':True,'response_linked':True,'confirmation_source':'verified_ui_change',
         'intervening_note_actions':[],'recognition_status':'valid','invalid_fields':[],
         **{flag:True for flag in FLAGS},'before':{'notes':[5]+[0]*12,'notes_complete':True},
         'after':{'notes':[0]*13,'notes_complete':True},'note_infos':[{'Tid':90011,'Qty':-5}],
         'conditions':{'client_version':'SYNTHETIC','server_region':'SYNTHETIC','tower_id':1,'floor':3,
                       'main_discs':[{'disc_id':n,'phase':0} for n in (211001,211002,211003)],'auxiliary_discs':[]}}
    conflict=copy.deepcopy(row);conflict['sequence']=2
    bad=copy.deepcopy(conflict);bad['after']['notes'][0]=None
    missing=copy.deepcopy(row);missing['sequence']=3;missing['before']['notes'][12]=None
    unconfirmed=copy.deepcopy(row);unconfirmed['sequence']=4;unconfirmed['operation_confirmed']=False
    rows=[('a',row),('a',row),('b',conflict),('b',bad),('c',missing),('d',unconfirmed)]
    db=directory/'SYNTHETIC.sqlite'
    with closing(sqlite3.connect(db)) as con, con:
        con.execute('CREATE TABLE demo_records (observation_seq INTEGER,run_key TEXT,action_key TEXT,payload TEXT)')
        for i,(action,record) in enumerate(rows):
            con.execute('INSERT INTO demo_records VALUES (?,?,?,?)',(i,'SYNTHETIC-RUN',action,json.dumps(record)))
    mapping={'schema':'joint_note_import_mapping_v1','namespace':'SYNTHETIC-DEMO-ONLY',
        'source_format':'sqlite','query':'SELECT run_key,action_key,payload FROM demo_records ORDER BY observation_seq',
        'json_columns':['payload'],'binding':{'status':'synthetic_test_only'},
        'identity':{'run':[['run_key']],'action':[['action_key']]},
        'fields':{field:{'path':['payload']+field.split('.')} for field in sorted(FIELDS)}}
    (directory/'SYNTHETIC.mapping.json').write_text(json.dumps(mapping,indent=2)+'\n',encoding='utf-8')
    (directory/'README.txt').write_text('All rows and table names are synthetic test fixtures, not a Maa native schema or real samples.\n',encoding='utf-8')
    return db


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-directory',type=Path,required=True)
    args=p.parse_args()
    print(create(args.output_directory))
