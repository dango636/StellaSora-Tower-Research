import copy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

ANALYSIS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ANALYSIS))
from note_capture_import import (ImportFailure, FIELDS, import_source, inspect_source,
                                 map_record, source_rows, digest_file)
from note_capture_quality import validate, FLAGS


def example():
    """Synthetic schema and observations. Not a Maa native database fixture."""
    return {'participant':'test-private-participant','session':'test-session','run':'test-run','action':'test-action',
        'sequence':1,'event_type':'exchange','option_id':10301,'operation_confirmed':True,
        'response_linked':True,'confirmation_source':'verified_ui_change',
        'intervening_note_actions':[],'recognition_status':'valid','invalid_fields':[],
        **{name:True for name in FLAGS},
        'before':{'notes':[5]+[0]*12,'notes_complete':True},
        'after':{'notes':[0]*13,'notes_complete':True},'note_infos':[{'Tid':90011,'Qty':-5}],
        'conditions':{'client_version':'fixture-version','server_region':'fixture-region',
            'tower_id':1,'floor':3,'main_discs':[{'disc_id':n,'phase':0} for n in (211001,211002,211003)],
            'auxiliary_discs':[]}}


def mapping(kind='jsonl'):
    return {'schema':'joint_note_import_mapping_v1','namespace':'synthetic-dataset',
        'source_format':kind,'binding':{'status':'synthetic_test_only'},
        'identity':{'run':[['participant'],['session'],['run']], 'action':[['action']]},
        'fields':{field:{'path':field.split('.')} for field in FIELDS}}


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='note-import-test-')
        self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.path=self.root/'fixture.jsonl'
        self.write([example()])

    def write(self,rows):
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')

    def run_import(self,config=None,name='result'):
        return import_source(self.path,config or mapping(),self.root/name,allow_test_profile=True)

    def test_test_data_is_marked_and_opaque_without_source_changes(self):
        before=self.path.read_bytes()
        report=self.run_import()
        self.assertEqual(report['quality_passed_unique_actions'],{'exchange':1})
        self.assertEqual(report['real_actions_passing_completeness_checks'],0)
        self.assertFalse(report['training_approved'])
        output=(self.root/'result/candidates.jsonl').read_text(encoding='utf-8')
        self.assertNotIn('test-private-participant',output)
        row=json.loads(output)
        self.assertFalse(row['training_eligible'])
        self.assertTrue(row['import_provenance']['synthetic'])
        self.assertEqual(self.path.read_bytes(),before)

    def test_no_test_opt_in_and_unbound_template_refuse(self):
        with self.assertRaisesRegex(ImportFailure,'explicit_opt_in'):
            import_source(self.path,mapping(),self.root/'out')
        cfg=json.loads((ANALYSIS/'note_capture_mapping_template.json').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ImportFailure,'not_bound'):
            import_source(self.path,cfg,self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_reviewed_binding_detects_schema_change(self):
        cfg=mapping();profile=inspect_source(self.path,'jsonl')
        cfg['binding']={'status':'reviewed_source_binding','structure_sha256':'wrong',
                        'producer_commit':'a'*40,'evidence':'synthetic test of binding verification only'}
        with self.assertRaisesRegex(ImportFailure,'structure_changed'):
            self.run_import(cfg)
        cfg['binding']['structure_sha256']=profile['structure_sha256']
        record,errors=map_record(example(),cfg,1)
        self.assertEqual(errors,[])
        changed=example();changed['new_field']=1;self.write([changed])
        with self.assertRaisesRegex(ImportFailure,'structure_changed'):
            self.run_import(cfg)

    def test_unknown_is_not_zero_and_missing_confirmation_is_not_success(self):
        row=example();row['after']['notes'][0]=None;row['operation_confirmed']=False
        self.write([row]);report=self.run_import()
        self.assertEqual(report['quality_passed_unique_actions'],{})
        self.assertIn('after_notes_unknown_or_invalid',report['rejection_reason_counts'])
        self.assertIn('unconfirmed_operation_confirmed',report['rejection_reason_counts'])

    def test_every_eligibility_flag_must_be_explicit(self):
        for flag in FLAGS:
            row=example();del row[flag]
            _record,errors=map_record(row,mapping(),1)
            self.assertIn('unconfirmed_'+flag,errors)

    def test_signed_qty_and_note_cost_match_inventory(self):
        row=example();row['note_infos'][0]['Qty']=5
        _record,errors=map_record(row,mapping(),1)
        self.assertIn('signed_infos_disagree_with_inventory',errors)
        row=example();row['after']['notes'][1]=1
        _record,errors=map_record(row,mapping(),1)
        self.assertIn('exchange_result_not_verified',errors)

    def test_duplicate_reobservations_are_one_action(self):
        self.write([example(),example()]);report=self.run_import()
        self.assertEqual(report['quality_passed_unique_actions'],{'exchange':1})
        self.assertEqual(report['reobservations_not_independent'],1)

    def test_conflicting_or_invalid_copy_invalidates_entire_action(self):
        invalid=example();invalid['after']['notes'][0]=None
        self.write([example(),invalid]);report=self.run_import()
        self.assertEqual(report['quality_passed_unique_actions'],{})
        self.assertEqual(report['conflicting_action_groups_excluded'],1)
        self.assertEqual(report['groups_with_invalid_observations_excluded'],1)
        rejection=(self.root/'result/rejections.jsonl').read_text(encoding='utf-8')
        self.assertNotIn('test-private-participant',rejection)
        self.assertNotIn('notes',rejection)

    def test_id_scope_separates_participants(self):
        other=example();other['participant']='another-test-participant'
        self.write([example(),other]);report=self.run_import()
        self.assertEqual(report['quality_passed_unique_actions'],{'exchange':2})

    def test_different_action_ids_with_same_sequence_are_excluded(self):
        other=example();other['action']='second-action'
        self.write([example(),other]);report=self.run_import()
        self.assertEqual(report['quality_passed_unique_actions'],{})
        self.assertEqual(report['sequence_collision_groups_excluded'],2)

    def test_missing_id_is_not_accepted_as_an_independent_action(self):
        row=example();row['run']=None;self.write([row])
        report=self.run_import()
        self.assertEqual(report['quality_passed_unique_actions'],{})
        self.assertIn('missing_run_identity',report['rejection_reason_counts'])

    def test_proof_cannot_be_hardcoded_as_a_constant(self):
        cfg=mapping();cfg['fields']['operation_confirmed']={'constant':True}
        with self.assertRaisesRegex(ImportFailure,'unsupported_field_mapping'):
            self.run_import(cfg)

    def test_sqlite_join_and_json_column_are_read_only(self):
        db=self.root/'fixture.sqlite'
        with closing(sqlite3.connect(db)) as con, con:
            con.executescript('CREATE TABLE fixture_ids(id INTEGER PRIMARY KEY,participant TEXT,session TEXT,run TEXT,action TEXT); CREATE TABLE fixture_payload(id INTEGER,document TEXT);')
            con.execute('INSERT INTO fixture_ids VALUES (1,?,?,?,?)',tuple(example()[k] for k in ('participant','session','run','action')))
            con.execute('INSERT INTO fixture_payload VALUES (1,?)',(json.dumps(example()),))
        old=digest_file(db);cfg=mapping('sqlite')
        cfg['query']='SELECT i.participant,i.session,i.run,i.action,p.document FROM fixture_ids i JOIN fixture_payload p ON p.id=i.id'
        cfg['json_columns']=['document']
        for spec in cfg['fields'].values():
            spec['path']=['document']+spec['path']
        report=import_source(db,cfg,self.root/'result',allow_test_profile=True)
        self.assertEqual(report['quality_passed_unique_actions'],{'exchange':1})
        self.assertEqual(digest_file(db),old)
        self.assertFalse(Path(str(db)+'-shm').exists())
        profile=inspect_source(db,'sqlite')
        self.assertEqual(len(profile['tables']),2)
        self.assertNotIn('test-private-participant',json.dumps(profile))

    def test_sql_mutations_attachments_and_extensions_are_refused(self):
        db=self.root/'fixture.sqlite'
        with closing(sqlite3.connect(db)) as con, con:
            con.execute('CREATE TABLE fixture(value TEXT)')
        digest=digest_file(db)
        for query in ('DELETE FROM fixture','DROP TABLE fixture',"ATTACH DATABASE ':memory:' AS other",
                      'PRAGMA user_version=99',"SELECT load_extension('no-such-extension')"):
            with self.subTest(query=query),self.assertRaises(ImportFailure):
                with source_rows(db,'sqlite',query=query) as rows:
                    list(rows)
            self.assertEqual(digest_file(db),digest)

    def test_nonempty_wal_requires_frozen_export(self):
        Path(str(self.path)+'-wal').write_bytes(b'fixture-wal')
        with self.assertRaisesRegex(ImportFailure,'frozen_database_export'):
            inspect_source(self.path,'jsonl')

    def test_bad_trailing_json_does_not_publish_partial_output(self):
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write('{not json}\n')
        with self.assertRaises(ImportFailure):
            self.run_import()
        self.assertFalse((self.root/'result').exists())

    def test_bad_row_beyond_structure_sample_also_discards_staging(self):
        self.write([example()]*1001)
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write('{bad trailing row}\n')
        self.assertFalse(inspect_source(self.path,'jsonl')['source_fully_scanned'])
        with self.assertRaises(ImportFailure):
            self.run_import()
        self.assertFalse((self.root/'result').exists())
        self.assertEqual(list(self.root.glob('note-import-*')),[])

    def test_duplicate_json_key_and_nonfinite_values_refuse(self):
        for value in ('{"key":1,"key":2}\n','{"key":NaN}\n'):
            self.path.write_text(value,encoding='utf-8')
            with self.assertRaises(ImportFailure):
                inspect_source(self.path,'jsonl')

    def test_bounded_zip_profile_does_not_claim_complete_scan(self):
        archive=self.root/'fixture.zip'
        with ZipFile(archive,'w') as z:
            z.writestr('fixture/records.jsonl',self.path.read_bytes()*2)
        profile=inspect_source(archive,'jsonl_zip',member='fixture/records.jsonl',sample_rows=1)
        self.assertEqual(profile['rows_sampled'],1)
        self.assertFalse(profile['source_fully_scanned'])
        self.assertNotIn('test-private-participant',json.dumps(profile))

    def test_source_changes_during_read_never_publish(self):
        from unittest.mock import patch
        import note_capture_import as module
        real=module.digest_file
        calls=0
        def changed(path):
            nonlocal calls
            calls+=1
            return real(path) if calls<3 else 'changed'
        with patch.object(module,'digest_file',side_effect=changed),self.assertRaisesRegex(ImportFailure,'source_changed'):
            self.run_import()
        self.assertFalse((self.root/'result').exists())

    def test_cli_profile_from_another_directory(self):
        output=self.root/'profile.json'
        result=subprocess.run([sys.executable,str(ANALYSIS/'note_capture_import.py'),'inspect',str(self.path),
            '--format','jsonl','--output',str(output)],cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(output.exists())

    def test_never_overwrite_existing_exports(self):
        self.run_import()
        marker=self.root/'result/candidates.jsonl';before=marker.read_bytes()
        with self.assertRaisesRegex(ImportFailure,'already_exists'):
            self.run_import()
        self.assertEqual(marker.read_bytes(),before)

    def test_cli_import_requires_test_marker_and_emits_quarantine(self):
        config=self.root/'mapping.json';config.write_text(json.dumps(mapping()),encoding='utf-8')
        result=subprocess.run([sys.executable,str(ANALYSIS/'note_capture_import.py'),'import',str(self.path),
            '--mapping',str(config),'--output-directory',str(self.root/'cli-result'),'--allow-test-profile'],
            cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads((self.root/'cli-result/report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['real_actions_passing_completeness_checks'],0)
        self.assertTrue((self.root/'cli-result/rejections.jsonl').exists())

    def test_unknown_phases_and_incomplete_conditions_are_rejected(self):
        row=example();row['conditions']['main_discs'][0]['phase']=None
        row['conditions']['client_version']='';row['conditions']['floor']=0
        _record,errors=map_record(row,mapping(),1)
        self.assertIn('unverified_main_discs_identity_or_phase',errors)
        self.assertIn('missing_condition_client_version',errors)
        self.assertIn('missing_condition_floor',errors)


if __name__=='__main__':
    unittest.main()
