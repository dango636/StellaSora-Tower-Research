"""Preview or stage a training-config revision at an unfrozen round boundary.

Never modifies checkpoints, teacher routing, score caches, completed manifests,
or retention settings. The existing Engine.apply_next_round_config performs
the actual versioned transition when it reaches a new, unfrozen round.

Staging configuration does not make replacing the trainer source safe during
an unfinished round. Use --check-code-upgrade with the old trainer stopped;
finish and commit any frozen round with its original code before replacing it.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile


def merge_overlay(base, overlay):
    result = copy.deepcopy(base)
    for key,value in overlay.items():
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key] = merge_overlay(result[key],value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _fit_artifacts(directory):
    """Only inspect known fit-stage markers; never open model arrays."""
    directory = Path(directory)
    paths = []
    for name in ('fit_identity.json', 'fit.json', 'fit_candidates.json'):
        path = directory / name
        if path.exists():
            paths.append(path)
    paths.extend(sorted((directory / 'fit_checkpoints').glob('*.npz')))
    for branch in sorted((directory / 'branches').glob('*')):
        if branch.is_dir():
            paths.extend(_fit_artifacts(branch))
    return paths


def _next_unfrozen_round(run_dir, number):
    while (run_dir / 'rounds' / f'round_{number:06d}' / 'manifest.json').exists():
        number += 1
    directory = run_dir / 'rounds' / f'round_{number:06d}'
    # A cache without the manifest that froze it cannot be reassigned to new
    # settings. Preserve all files and require the original run to be repaired.
    orphaned = _fit_artifacts(directory)
    for name in ('summary.json', 'official_candidate_scores.npz'):
        if (directory / name).exists():
            orphaned.append(directory / name)
    return number, orphaned


def _summary_materials(run_dir, state, directory, number):
    """Check commit metadata/references, not model contents or score validity.

    Even a complete summary is not a code-upgrade boundary until the original
    engine has committed it and advanced run_state.next_round.
    """
    from stella.io_utils import read_json
    from stella.run_paths import run_path
    required = ('round', 'teacher_bank', 'teacher_scores', 'candidate_id',
                'candidate_checkpoint', 'best_model_id', 'best_candidate_mean',
                'composite_teacher_mean', 'candidate_mean', 'champion_updated_count',
                'distinct_teacher_count', 'primary_delta')
    try:
        summary = read_json(directory / 'summary.json')
        missing = [key for key in required if key not in summary]
        if missing:
            return dict(complete=False, missing_fields=missing)
        if summary['round'] != number or len(summary['teacher_bank']) != 4960:
            return dict(complete=False, reason='summary round or teacher-bank size disagrees')
        models = dict(state.get('models', {}))
        models.update(summary.get('candidate_models', {}))
        models[summary['candidate_id']] = summary['candidate_checkpoint']
        needed = set(summary['teacher_bank']) | {summary['best_model_id'], summary['candidate_id']}
        missing_ids = sorted(needed - models.keys())
        references = [summary['teacher_scores']]
        for key in sorted(needed & models.keys()):
            value = models[key]
            # Historical metadata can contain a path record rather than a
            # string; this check is not a migration of either representation.
            references.append(value.get('checkpoint', value.get('path'))
                              if isinstance(value, dict) else value)
        missing_files = []
        for reference in references:
            if not isinstance(reference, str) or not run_path(run_dir, reference).is_file():
                missing_files.append(reference)
        return dict(complete=not missing_ids and not missing_files,
                    missing_model_ids=missing_ids, missing_files=missing_files,
                    referenced_files_checked=len(references),
                    verification='metadata and file existence only; original engine must commit')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return dict(complete=False, reason=str(exc))


def inspect_code_upgrade(run_dir):
    """Read-only snapshot of frozen rounds and code-sensitive fit identities.

    Dynamic diagnostics deliberately stay out of pending_config.json, so a
    long-running old round cannot change the identity of a staged revision.
    This does not acquire an engine lock or claim that a live process stopped.
    """
    from stella.config import resolve
    from stella.fit_cache import identity
    from stella.io_utils import digest, read_json
    run_dir = Path(run_dir).expanduser().resolve()
    state = read_json(run_dir / 'run_state.json')
    config = read_json(run_dir / 'resolved_config.json')
    if state.get('protocol') != 'catalogue4960_per_preset_champions_v1':
        raise ValueError('Expected an existing 4960-preset teacher-bank run')
    if len(state.get('teacher_bank', [])) != 4960:
        raise ValueError('Incomplete frozen teacher bank')
    number = int(state['next_round'])
    first = number
    frozen = []
    while (run_dir / 'rounds' / f'round_{number:06d}' / 'manifest.json').exists():
        directory = run_dir / 'rounds' / f'round_{number:06d}'
        record = dict(round=number, directory=str(directory),
                      status='unfinished_frozen_round')
        if (directory / 'summary.json').exists():
            material = _summary_materials(run_dir, state, directory, number)
            record.update(status='completed_pending_commit' if material['complete']
                          else 'incomplete_summary', commit_materials=material)
        artifacts = _fit_artifacts(directory)
        record['fit_artifacts'] = [str(path.relative_to(directory)) for path in artifacts]
        try:
            manifest = read_json(directory / 'manifest.json')
            expected = identity(manifest, config['seed'])
            saved = directory / 'fit_identity.json'
            if saved.exists():
                actual = read_json(saved)
                keys = sorted(set(actual) | set(expected or {})) if isinstance(actual, dict) else []
                differences = [key for key in keys if actual.get(key) != (expected or {}).get(key)]
                if actual == expected:
                    record['fit_identity_status'] = 'compatible_with_current_source'
                else:
                    record['fit_identity_status'] = ('implementation_mismatch'
                        if differences == ['implementation_sha256'] else 'identity_mismatch')
                    record['different_identity_fields'] = differences
            elif artifacts and expected is not None:
                record['fit_identity_status'] = 'unidentified_fit_cache'
            else:
                record['fit_identity_status'] = ('no_fit_cache_yet' if not artifacts
                                                 else 'legacy_untracked_fit_cache')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record.update(fit_identity_status='cannot_verify', fit_identity_error=str(exc))
        frozen.append(record)
        number += 1
    boundary, orphaned = _next_unfrozen_round(run_dir, number)
    config_hash = digest(config)
    consistent = (state.get('config_hash') == config_hash and
                  resolve(read_json(run_dir / 'config.json')) == config)
    # A live engine can freeze the next manifest between the loop and this
    # final boundary read. That is a blocked snapshot, not an empty safe run.
    changed_during_check = boundary != number
    ready = not frozen and not orphaned and consistent and not changed_during_check
    return dict(ready_at_snapshot=ready, state_next_round=first,
                earliest_unfrozen_round=boundary, frozen_rounds=frozen,
                boundary_changed_during_check=changed_during_check,
                active_config_sha256=config_hash,
                active_config_identity_consistent=consistent,
                orphaned_stage_artifacts=[str(path.relative_to(run_dir)) for path in orphaned],
                configuration_staging='pending configuration may wait for the next unfrozen round; it does not migrate source code or fit caches',
                next_step=('Stop the trainer, recheck this boundary, then replace source files and resume.' if ready else
                    'Keep all checkpoints, teachers and caches. Finish/commit frozen rounds with the original trainer, stop before the next manifest, then recheck. Do not delete fit_identity.json or clear caches.'),
                snapshot_only=True)


def prepare_revision(run_dir,overlay,*,revision=None,extra_config_path=None):
    from stella.config import resolve
    from stella.io_utils import digest,read_json
    run_dir = Path(run_dir).expanduser().resolve()
    state = read_json(run_dir/'run_state.json')
    previous = read_json(run_dir/'resolved_config.json')
    raw = read_json(run_dir/'config.json')
    if state.get('protocol')!='catalogue4960_per_preset_champions_v1':
        raise ValueError('Expected an existing 4960-preset teacher-bank run')
    if len(state.get('teacher_bank',[]))!=4960:
        raise ValueError('Incomplete frozen teacher bank')
    if resolve(raw)!=previous or state.get('config_hash')!=digest(previous):
        raise ValueError('Active config identity disagrees; resolve the existing revision first')
    allowed = {'training','catalogue_coverage'}
    if set(overlay)-allowed:
        raise ValueError('This helper only changes training and catalogue_coverage settings')
    updated_raw = merge_overlay(raw,overlay)
    updated = resolve(updated_raw)
    if updated==previous:
        return None
    for section in ('seed','compute','evaluation','sampling','retention','model','catalogue_mode'):
        if updated.get(section)!=previous.get(section):
            raise ValueError('Unrelated config change: '+section)
    # A manifest owns its frozen settings even when collection was interrupted.
    number, orphaned = _next_unfrozen_round(run_dir, int(state['next_round']))
    if orphaned:
        raise ValueError('Fit/summary stage artifacts exist without their frozen manifest; '
                         'preserve all caches and restore the original run before staging a revision')
    tag = revision or 'action_calibration_v3_'+digest(updated)[:12]
    if not tag or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for char in tag):
        raise ValueError('revision must be a simple filename-safe identifier')
    if (run_dir/'config_revisions'/(tag+'.json')).exists():
        raise ValueError('This revision identifier was already applied; use a new revision name')
    paths = [str(run_dir/'config.json')]
    if extra_config_path is not None:
        extra = Path(extra_config_path).expanduser().resolve()
        if not extra.is_file():
            raise ValueError('The optional launcher config must already exist')
        if str(extra) not in paths:
            paths.append(str(extra))
    return dict(revision=tag,effective_round=number,previous_config_hash=digest(previous),
        description='全4960预设独立训练/验证轨迹；冻结逐预设教师和正式选择流程保持原配置',
        config=updated_raw,config_paths=paths)


def atomic_create_json(path,value):
    """Publish a complete new pending file atomically, without replacement."""
    path=Path(path)
    encoded=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode('utf-8')
    fd,name=tempfile.mkstemp(prefix='.pending-calibration-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(name,path)  # atomic create-if-absent, fails if another writer won
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True,help='JSON training/coverage overlay')
    parser.add_argument('--trainer',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--revision')
    parser.add_argument('--extra-config-path',type=Path)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--write-pending',action='store_true',help='Default only previews; this writes pending_config.json')
    action.add_argument('--check-code-upgrade',action='store_true',
        help='Read-only check; exit 2 if a frozen round still needs its original source or commit')
    args=parser.parse_args()
    sys.path.insert(0,str(args.trainer.resolve()))
    from stella.io_utils import read_json,digest
    overlay=json.loads(args.profile.read_text(encoding='utf-8'))
    upgrade=inspect_code_upgrade(args.run_dir)
    if args.check_code_upgrade:
        summary={'status':'code_upgrade_ready' if upgrade['ready_at_snapshot'] else 'code_upgrade_blocked',
                 'code_upgrade':upgrade}
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        if not upgrade['ready_at_snapshot']:
            raise SystemExit(2)
        return
    change=prepare_revision(args.run_dir,overlay,revision=args.revision,extra_config_path=args.extra_config_path)
    if change is None:
        print(json.dumps({'status':'already_active','code_upgrade':upgrade},ensure_ascii=False,indent=2))
        return
    target=args.run_dir.resolve()/'pending_config.json'
    summary={'status':'preview','revision':change['revision'],'effective_round':change['effective_round'],
             'target':str(target),'training':change['config']['training'],
             'catalogue_coverage':change['config'].get('catalogue_coverage'),
             'teacher_bank_modified':False,'checkpoints_modified':False,'formal_evaluation_modified':False,
             'effective_round_is_earliest_eligible':True,'code_upgrade':upgrade}
    if args.write_pending:
        if target.exists():
            if read_json(target)==change:
                summary['status']='already_staged'
            else:
                raise ValueError('A different pending revision exists; it has not been overwritten')
        else:
            # Recheck immediately before writing; never overwrite an active
            # configuration transition that completed while previewing.
            if digest(read_json(args.run_dir/'resolved_config.json'))!=change['previous_config_hash']:
                raise ValueError('Active config changed during preparation; rerun the preview')
            # The old engine may have frozen another round since the first
            # preview. Never target that directory; Engine also rechecks the
            # manifest when it eventually applies this unchanged pending form.
            latest=inspect_code_upgrade(args.run_dir)
            if (not latest['active_config_identity_consistent'] or
                    latest['active_config_sha256'] != change['previous_config_hash']):
                raise ValueError('Active config changed during preparation; rerun the preview')
            if latest['orphaned_stage_artifacts']:
                raise ValueError('Fit/summary stage artifacts appeared without a frozen manifest; '
                                 'preserve them and rerun after restoring the original run')
            change['effective_round']=max(change['effective_round'],latest['earliest_unfrozen_round'])
            summary.update(effective_round=change['effective_round'],code_upgrade=latest)
            try:
                atomic_create_json(target,change)
                summary['status']='staged_for_new_round'
            except FileExistsError:
                if read_json(target)!=change:
                    raise ValueError('A different pending revision was created concurrently; it was preserved')
                summary['status']='already_staged'
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
