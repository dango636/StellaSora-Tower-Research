"""Read-only SQLite/JSONL structure inspection and explicitly bound note import.

An import mapping describes a reviewed source, not a proposed native DB format.
There is intentionally no guessed native schema, column aliases or training hook.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from zipfile import ZipFile

from note_capture_quality import NOTE_IDS, FLAGS, validate

MAX_LINE = 8*1024*1024
FIELDS = {
    'sequence','event_type','operation_confirmed','response_linked','confirmation_source',
    'intervening_note_actions','before.notes','before.notes_complete','after.notes','after.notes_complete',
    'note_infos','option_id','recognition_status','invalid_fields',*FLAGS,
    'conditions.client_version','conditions.server_region','conditions.tower_id','conditions.floor',
    'conditions.main_discs','conditions.auxiliary_discs',
}


class ImportFailure(ValueError):
    """Reason codes only; do not echo source rows, SQL, IDs or SQLite errors."""


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',',':'), allow_nan=False)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def frozen_source(path):
    path = Path(path).resolve(strict=True)
    if any(Path(str(path)+suffix).exists() and Path(str(path)+suffix).stat().st_size
           for suffix in ('-wal','-journal')):
        raise ImportFailure('requires_frozen_database_export')
    return path


def quoted(name):
    return '"'+name.replace('"','""')+'"'


def authorizer(action, arg1, arg2, _database, _trigger):
    permitted = {sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION,sqlite3.SQLITE_RECURSIVE}
    if action not in permitted:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or '').lower() in ('load_extension','readfile','writefile'):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


@contextmanager
def database(path):
    path = frozen_source(path)
    con = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    con.enable_load_extension(False)
    con.execute('PRAGMA query_only=ON')
    con.execute('PRAGMA trusted_schema=OFF')
    calls = 0
    def progress():
        nonlocal calls
        calls += 1
        return int(calls > 50000)  # bounded at roughly fifty million VM instructions
    con.set_progress_handler(progress,1000)
    try:
        yield con
    finally:
        con.close()


def json_object(text):
    def bad_constant(_value):
        raise ImportFailure('nonfinite_json')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:
                raise ImportFailure('duplicate_json_key')
            result[key]=value
        return result
    try:
        return json.loads(text,parse_constant=bad_constant,object_pairs_hook=pairs)
    except (ValueError,UnicodeError,RecursionError) as exc:
        if isinstance(exc,ImportFailure):
            raise
        raise ImportFailure('invalid_json') from None


def json_rows(stream):
    while True:
        line=stream.readline(MAX_LINE+1)
        if not line:
            return
        if len(line)>MAX_LINE:
            raise ImportFailure('source_record_too_large')
        if not line.strip():
            continue
        row=json_object(line)
        if not isinstance(row,dict):
            raise ImportFailure('source_row_not_object')
        yield row


@contextmanager
def source_rows(path, kind, *, query=None, member=None):
    if kind=='sqlite':
        if not query or not isinstance(query,str):
            raise ImportFailure('missing_explicit_select_query')
        with database(path) as con:
            con.set_authorizer(authorizer)
            try:
                cursor=con.execute(query)
                names=[column[0] for column in cursor.description or []]
                if len(names)!=len(set(names)):
                    raise ImportFailure('ambiguous_query_column_names')
                yield (dict(zip(names,row)) for row in cursor)
            except sqlite3.Error:
                raise ImportFailure('source_select_rejected_or_failed') from None
    elif kind=='jsonl':
        with Path(path).open('rb') as stream:
            yield json_rows(stream)
    elif kind=='jsonl_zip':
        if not isinstance(member,str):
            raise ImportFailure('missing_explicit_zip_member')
        with ZipFile(path) as archive:
            if archive.namelist().count(member)!=1:
                raise ImportFailure('zip_member_missing_or_ambiguous')
            with archive.open(member) as stream:
                yield json_rows(stream)
    else:
        raise ImportFailure('unsupported_source_format')


def inspect_source(path, kind, *, member=None, sample_rows=1000):
    path=frozen_source(path)
    if type(sample_rows) is not int or sample_rows<1:
        raise ImportFailure('invalid_sample_limit')
    result={'schema':'note_capture_source_profile_v1','format':kind,'source_sha256':digest_file(path),
            'raw_rows_included':False,'training_approved':False}
    if kind=='sqlite':
        tables=[]
        with database(path) as con:
            names=con.execute("SELECT name,type FROM sqlite_schema WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
            for name,table_type in names:
                # Trusted metadata query, before restricting the explicitly supplied SELECT.
                columns=[{'name':r[1],'declared_type':r[2],'not_null':bool(r[3]),'primary_key':r[5]}
                         for r in con.execute('PRAGMA table_info('+quoted(name)+')')]
                tables.append({'name':name,'type':table_type,'columns':columns})
            result['tables']=tables
        signature=tables
    else:
        paths,scanned,truncated={},0,False
        def walk(value, path='$', depth=0):
            if depth>16 or len(paths)>4096:
                raise ImportFailure('structure_profile_limit_exceeded')
            label='null' if value is None else type(value).__name__
            paths.setdefault(path,set()).add(label)
            if isinstance(value,dict):
                for key,item in value.items():
                    safe=key if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,79}',key) or key in map(str,NOTE_IDS) else '*'
                    walk(item,path+'/'+safe,depth+1)
            elif isinstance(value,list):
                for item in value:
                    walk(item,path+'/*',depth+1)
        with source_rows(path,kind,member=member) as rows:
            for row in rows:
                if scanned==sample_rows:
                    truncated=True;break
                walk(row);scanned+=1
        signature={path:sorted(types) for path,types in sorted(paths.items())}
        result.update({'sampled_field_types':signature,'rows_sampled':scanned,
                       'source_fully_scanned':not truncated,'sample_limit':sample_rows})
        if kind=='jsonl_zip':
            result['member']=member
    result['structure_sha256']=hashlib.sha256(canonical(signature).encode()).hexdigest()
    if frozen_source(path)!=path or digest_file(path)!=result['source_sha256']:
        raise ImportFailure('source_changed_during_inspection')
    return result


def at(row,path):
    if not isinstance(path,list) or not path:
        raise ImportFailure('invalid_source_path')
    value=row
    for part in path:
        if isinstance(value,dict) and isinstance(part,str):
            value=value.get(part)
        elif isinstance(value,list) and type(part) is int and 0<=part<len(value):
            value=value[part]
        else:
            return None
    return value


def set_field(record,target,value):
    parts=target.split('.')
    for name in parts[:-1]:
        record=record.setdefault(name,{})
    record[parts[-1]]=value


def check_mapping(mapping,profile,allow_test_profile):
    if not isinstance(mapping,dict):
        raise ImportFailure('mapping_not_object')
    if mapping.get('schema')!='joint_note_import_mapping_v1':
        raise ImportFailure('unsupported_mapping_schema')
    binding=mapping.get('binding') or {}
    status=binding.get('status')
    if status=='synthetic_test_only':
        if not allow_test_profile:
            raise ImportFailure('test_profile_requires_explicit_opt_in')
    elif status!='reviewed_source_binding':
        raise ImportFailure('source_schema_not_bound')
    elif (binding.get('structure_sha256')!=profile['structure_sha256']
          or not isinstance(binding.get('producer_commit'),str)
          or not re.fullmatch(r'[0-9a-f]{40}',binding['producer_commit'])
          or not isinstance(binding.get('evidence'),str) or not binding['evidence']):
        raise ImportFailure('binding_missing_or_structure_changed')
    if not isinstance(mapping.get('namespace'),str) or not mapping['namespace']:
        raise ImportFailure('missing_dataset_namespace')
    identity=mapping.get('identity') or {}
    for key in ('run','action'):
        if not isinstance(identity.get(key),list) or not identity[key]:
            raise ImportFailure('missing_composite_identity_mapping')
    if not isinstance(mapping.get('fields'),dict) or not mapping['fields']:
        raise ImportFailure('missing_field_mapping')
    if (not isinstance(mapping.get('json_columns',[]),list)
            or any(not isinstance(name,str) for name in mapping.get('json_columns',[]))):
        raise ImportFailure('invalid_json_column_mapping')
    for target,spec in mapping['fields'].items():
        if target not in FIELDS or not isinstance(spec,dict) or set(spec)-{'path','transform','values'}:
            raise ImportFailure('unsupported_field_mapping')
        if not isinstance(spec.get('path'),list) or not spec['path']:
            raise ImportFailure('missing_source_field_path')
        if spec.get('transform') not in (None,'bool01','enum'):
            raise ImportFailure('unsupported_explicit_transform')
        if spec.get('transform')=='enum' and not isinstance(spec.get('values'),dict):
            raise ImportFailure('invalid_enum_mapping')
    return status


def map_record(row,mapping,number):
    errors=[]
    row=dict(row)
    for name in mapping.get('json_columns',[]):
        value=row.get(name)
        if not isinstance(value,(str,bytes)) or len(value)>MAX_LINE:
            errors.append('json_column_not_text');row[name]=None;continue
        try:
            row[name]=json_object(value)
        except ImportFailure:
            errors.append('invalid_json_column');row[name]=None
    record={'schema':'joint_note_transition_v1'}
    identity=[]
    for kind in ('run','action'):
        values=[at(row,path) for path in mapping['identity'][kind]]
        if any(value is None or type(value) not in (str,int) or value=='' for value in values):
            errors.append('missing_'+kind+'_identity')
            values=['unidentified_source_row',number,kind]
        identity.extend(values)
        record[kind+'_id']=kind+'-'+hashlib.sha256(canonical([mapping['namespace'],identity]).encode()).hexdigest()
    for target,spec in mapping['fields'].items():
        value=at(row,spec['path'])
        if spec.get('transform')=='bool01':
            if type(value) is int and value in (0,1):
                value=bool(value)
            else:
                errors.append('invalid_bool01_transform');value=None
        elif spec.get('transform')=='enum':
            choices=spec.get('values',{})
            if type(value) not in (str,int) or str(value) not in choices:
                errors.append('unmapped_enum');value=None
            else:
                value=choices[str(value)]
        set_field(record,target,value)
    try:
        canonical(record)
    except (TypeError,ValueError,RecursionError):
        raise ImportFailure('mapped_record_not_finite_json') from None
    return record,sorted(set(errors+validate(record)))


def import_source(path,mapping,output, *, allow_test_profile=False):
    if not isinstance(mapping,dict):
        raise ImportFailure('mapping_not_object')
    path=frozen_source(path);output=Path(output).resolve()
    if output.exists():
        raise ImportFailure('output_already_exists')
    if (mapping.get('binding') or {}).get('status')=='unbound':
        raise ImportFailure('source_schema_not_bound')
    kind=mapping.get('source_format')
    profile=inspect_source(path,kind,member=mapping.get('member'))
    status=check_mapping(mapping,profile,allow_test_profile)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='note-import-',dir=output.parent) as temporary:
        staging=Path(temporary)
        spool=sqlite3.connect(staging/'groups.sqlite')
        try:
            spool.execute('CREATE TABLE records (seq INTEGER,run TEXT,action TEXT,signature TEXT,record TEXT,errors TEXT,action_seq INTEGER)')
            total=0;reasons=Counter()
            with source_rows(path,kind,query=mapping.get('query'),member=mapping.get('member')) as rows:
                for total,row in enumerate(rows,1):
                    record,errors=map_record(row,mapping,total)
                    text=canonical(record)
                    action_seq=record.get('sequence')
                    if type(action_seq) is not int or not 0<=action_seq<2**63:
                        action_seq=None
                    spool.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?)',
                        (total,record['run_id'],record['action_id'],hashlib.sha256(text.encode()).hexdigest(),text,canonical(errors),action_seq))
                    reasons.update(errors)
            spool.commit()
            spool.execute('CREATE INDEX identity_idx ON records(run,action)')
            collisions=set(spool.execute('SELECT r.run,r.action FROM records r JOIN (SELECT run,action_seq FROM records WHERE action_seq IS NOT NULL GROUP BY run,action_seq HAVING COUNT(DISTINCT action)>1) bad ON r.run=bad.run AND r.action_seq=bad.action_seq'))
            accepted=Counter();repeated=conflicts=invalid_groups=sequence_collisions=0
            with (staging/'candidates.jsonl').open('w',encoding='utf-8') as good, (staging/'rejections.jsonl').open('w',encoding='utf-8') as bad:
                groups=spool.execute('SELECT run,action,COUNT(*),COUNT(DISTINCT signature),MAX(errors!=?) FROM records GROUP BY run,action ORDER BY MIN(seq)',('[]',))
                for run,action,count,variants,invalid in groups:
                    collision=(run,action) in collisions
                    if variants>1 or invalid or collision:
                        conflicts+=int(variants>1);invalid_groups+=int(invalid)
                        sequence_collisions+=int(collision)
                        bad.write(canonical({'run_id':run,'action_id':action,'source_rows':count,
                            'conflicting_observations':variants>1,'has_invalid_observation':bool(invalid),
                            'ambiguous_action_sequence':collision})+'\n')
                        continue
                    repeated+=count-1
                    record=json.loads(spool.execute('SELECT record FROM records WHERE run=? AND action=? LIMIT 1',(run,action)).fetchone()[0])
                    record['import_provenance']={'source_sha256':profile['source_sha256'],
                        'mapping_sha256':hashlib.sha256(canonical(mapping).encode()).hexdigest(),
                        'binding_status':status,'synthetic':status=='synthetic_test_only','training_approved':False}
                    if status=='synthetic_test_only':
                        record['training_eligible']=False  # Test rows must also fail legacy training filters.
                    good.write(canonical(record)+'\n');accepted[record['event_type']]+=1
        finally:
            spool.close()
        if frozen_source(path)!=path or digest_file(path)!=profile['source_sha256']:
            raise ImportFailure('source_changed_during_import')
        report={'schema':'joint_note_import_report_v1','source_profile':profile,'binding_status':status,
            'rows_read':total,'quality_passed_unique_actions':dict(accepted),
            'real_actions_passing_completeness_checks':0 if status=='synthetic_test_only' else sum(accepted.values()),
            'reobservations_not_independent':repeated,'conflicting_action_groups_excluded':conflicts,
            'groups_with_invalid_observations_excluded':invalid_groups,'rejection_reason_counts':dict(reasons),
            'sequence_collision_groups_excluded':sequence_collisions,
            'training_approved':False,'scope':'Completeness and deduplication only; no calibrated kernel or training approval.'}
        report['producer_binding_independently_verified']=False
        (staging/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        # The spool contains only temporary canonical rows. Publish three files
        # together after the complete read and source identity check succeed.
        (staging/'groups.sqlite').unlink()
        if staging.resolve().parent!=output.parent:
            raise ImportFailure('staging_outside_output_parent')
        staging.rename(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    inspect=commands.add_parser('inspect')
    inspect.add_argument('source',type=Path)
    inspect.add_argument('--format',choices=('sqlite','jsonl','jsonl_zip'),required=True)
    inspect.add_argument('--member')
    inspect.add_argument('--sample-rows',type=int,default=1000)
    inspect.add_argument('--output',type=Path,required=True)
    convert=commands.add_parser('import')
    convert.add_argument('source',type=Path)
    convert.add_argument('--mapping',type=Path,required=True)
    convert.add_argument('--output-directory',type=Path,required=True)
    convert.add_argument('--allow-test-profile',action='store_true')
    args=p.parse_args()
    try:
        if args.command=='inspect':
            result=inspect_source(args.source,args.format,member=args.member,sample_rows=args.sample_rows)
            if args.output.exists():
                raise ImportFailure('output_already_exists')
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        else:
            mapping=json_object(args.mapping.read_bytes())
            result=import_source(args.source,mapping,args.output_directory,allow_test_profile=args.allow_test_profile)
        keys=('schema','rows_sampled','source_fully_scanned','structure_sha256','training_approved') if args.command=='inspect' else (
            'schema','rows_read','quality_passed_unique_actions','real_actions_passing_completeness_checks','training_approved')
        print(json.dumps({key:result[key] for key in keys if key in result},ensure_ascii=True))
    except (ImportFailure,OSError,sqlite3.Error) as error:
        code=str(error) if isinstance(error,ImportFailure) else 'source_or_output_io_failed'
        p.exit(2,'Import refused: '+code+'\n')


if __name__=='__main__':
    main()
