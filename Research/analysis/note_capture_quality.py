"""Conservative model-side completeness checks; never approve model training.

Canonical vocabulary follows the earlier concerto_joint capture validator.
No native Maa database schema is assumed here.
"""
NOTE_IDS = tuple(range(90011, 90024))
FLAGS = ('training_eligible', 'probability_eligible', 'sequence_complete',
         'source_complete', 'sequence_inference_eligible')


def note_vector(value):
    if isinstance(value, dict) and set(value) == set(map(str, NOTE_IDS)):
        value = [value[str(n)] for n in NOTE_IDS]
    if (not isinstance(value, list) or len(value) != 13
            or any(type(n) is not int or n < 0 for n in value)):
        raise ValueError('complete_observed_notes_required')
    return value


def validate(record):
    errors = []
    if not isinstance(record, dict) or record.get('schema') != 'joint_note_transition_v1':
        return ['unsupported_record_schema']
    for key in ('run_id', 'action_id'):
        if not isinstance(record.get(key), str) or not record[key]:
            errors.append('missing_'+key)
    if type(record.get('sequence')) is not int or record['sequence'] < 0:
        errors.append('invalid_action_sequence')
    if record.get('event_type') not in ('battle', 'exchange', 'event_reward', 'purchase'):
        errors.append('unknown_event_type')
    for key in ('operation_confirmed', 'response_linked') + FLAGS:
        if record.get(key) is not True:
            errors.append('unconfirmed_'+key)
    if record.get('confirmation_source') not in ('server_response', 'verified_ui_change'):
        errors.append('invalid_confirmation_source')
    if record.get('intervening_note_actions') != []:
        errors.append('not_an_isolated_transition')
    if record.get('recognition_status') != 'valid' or record.get('invalid_fields') != []:
        errors.append('unverified_or_invalid_observation')
    counts = {}
    for side in ('before', 'after'):
        state = record.get(side)
        if not isinstance(state, dict):
            errors.append(side+'_state_invalid'); continue
        if state.get('notes_complete') is not True:
            errors.append(side+'_notes_incomplete')
        try:
            counts[side] = note_vector(state.get('notes'))
        except ValueError:
            errors.append(side+'_notes_unknown_or_invalid')
    if len(counts) == 2:
        delta = [a-b for b,a in zip(counts['before'], counts['after'])]
        entries = record.get('note_infos')
        if entries is not None:
            actual = [0]*13
            if not isinstance(entries, list):
                errors.append('note_infos_invalid')
            else:
                for entry in entries:
                    if (not isinstance(entry, dict) or type(entry.get('Tid')) is not int
                            or entry['Tid'] not in NOTE_IDS or type(entry.get('Qty')) is not int):
                        errors.append('signed_note_info_invalid'); continue
                    actual[entry['Tid']-90011] += entry['Qty']
                if actual != delta:
                    errors.append('signed_infos_disagree_with_inventory')
        if record.get('event_type') == 'exchange':
            cost = {10301:5, 10401:10}.get(record.get('option_id'))
            if cost is None or sum(delta) != -cost or any(n > 0 for n in delta):
                errors.append('exchange_result_not_verified')
    conditions = record.get('conditions')
    if not isinstance(conditions, dict):
        return sorted(set(errors+['missing_conditions']))
    for key in ('client_version','server_region'):
        value=conditions.get(key)
        if not isinstance(value,str) or not value.strip() or value.lower() in ('unknown','unverified'):
            errors.append('missing_condition_'+key)
    if type(conditions.get('tower_id')) is not int or conditions['tower_id']<=0:
        errors.append('missing_condition_tower_id')
    if type(conditions.get('floor')) is not int or not 1<=conditions['floor']<=20:
        errors.append('missing_condition_floor')
    for key in ('main_discs','auxiliary_discs'):
        discs = conditions.get(key)
        if not isinstance(discs, list) or len(discs) > 3 or (key == 'main_discs' and len(discs) != 3):
            errors.append('incomplete_'+key); continue
        for disc in discs:
            if (not isinstance(disc, dict) or type(disc.get('disc_id')) is not int or disc['disc_id'] <= 0
                    or type(disc.get('phase')) is not int or not 0 <= disc['phase'] <= 8):
                errors.append('unverified_'+key+'_identity_or_phase')
    return sorted(set(errors))
