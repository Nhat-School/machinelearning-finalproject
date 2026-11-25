"""Regenerate an explicitly AI-authored semantic audit of the fixed CustomForum drafts.

The judgments below were recorded after reading the 160 example texts. Re-running
this file reproduces those saved judgments and structural checks; it is not a new
independent review and never writes to the human-review worksheet.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'Datasets/CustomForum'
EXPECTED_DRAFT_SHA256 = '7e9f2553b79d5fec18a300b0edd684daf37ee9ec036a77df233ab937479a0991'

# Individual scenario judgments recorded under the current-support rubric.
SCENARIOS = {
    'submission_button': 'A failed submission near its deadline requires staff support.',
    'exam_login': 'A locked account blocks an imminent exam after password reset failed.',
    'quiz_timer': 'An incorrect quiz time limit requires correction of the assessment attempt.',
    'enrollment_block': 'An incorrect prerequisite record blocks enrollment before closing.',
    'missing_grade': 'A missing project grade needs investigation before the appeal deadline.',
    'receipt_missing': 'A failed upload with no receipt requires confirmation before the deadline.',
    'autograder_down': 'A required unavailable autograder prevents completing an assessment.',
    'required_pdf': 'Unreadable mandatory instructions block an assessment due tonight.',
    'video_access': 'Unavailable required lecture material blocks the upcoming graded task.',
    'accommodation': 'Approved exam accommodations have not been applied before the exam.',
    'conflicting_exam_date': 'Conflicting imminent exam dates require authoritative clarification.',
    'exam_room': 'An exam room listed as closed needs clarification before the exam.',
    'public_submission': 'Unintended public exposure of assessment work requires staff intervention.',
    'account_compromise': 'Lost course-account access blocks an assessment due today.',
    'group_upload': 'The group cannot submit its project before the stated deadline.',
    'lab_booking': 'A restriction prevents booking a compulsory assessed lab before closing.',
    'late_penalty': 'A disputed late penalty needs receipt review before the appeal closes.',
    'payment_access': 'A payment problem suspends access to the current assessment.',
    'timezone_deadline': 'Conflicting deadline time zones need clarification before submission closes.',
    'oral_exam_link': 'Missing joining information blocks an imminent oral exam.',
    'wrong_dataset': 'Incorrect mandatory dataset columns prevent running assessment starter code.',
    'screen_reader': 'Missing accessible assessment content blocks a screen-reader user.',
    'incorrect_formula': 'Conflicting formula versions require clarification for current graded work.',
    'peer_review_locked': 'An inaccessible assigned submission blocks a peer review due tonight.',
    'extension_request': 'A pending documented extension needs a decision before portal closure.',
    'attendance_record': 'An incorrect attendance record blocks assessment eligibility before correction closes.',
    'registration_error': 'Incorrect exam-course registration requires correction before closing.',
    'exam_webcam': 'An equipment-check failure needs an approved alternative before the exam.',
    'outage_alternative': 'An outage blocks timely submission and requires an approved alternative.',
    'wrong_file_locked': 'A prematurely locked portal prevents correcting an erroneous submission.',
    'course_removed': 'Incorrect course removal blocks access to an assessment due today.',
    'required_reading': 'A broken access link prevents obtaining reading required for graded work.',
    'starter_version': 'A package mismatch in the mandatory environment blocks practical-exam starter files.',
    'answer_field': 'An invalid decimal-input restriction blocks the required graded answer format.',
    'announcements_hidden': 'Unavailable announcements hide joining instructions for this evening exam.',
    'lab_database': 'An offline required database blocks an assessed lab before submission closes.',
    'timetable_clash': 'Two compulsory graded assessments conflict and require staff scheduling action.',
    'withdrawal_error': 'An erroneous withdrawal blocks assessment registration before closing.',
    'reset_email': 'An undelivered reset email prevents access to a quiz due tonight.',
    'grading_rubric': 'Conflicting rubrics require clarification for the required presentation tomorrow.',
}
AMBIGUOUS_NOTES = [
    'The request identifies neither the problem nor its impact or deadline.',
    'The failed object and its impact are unspecified.',
    'The referenced material and need for intervention are unspecified.',
    'The situation needed to decide the next action is missing.',
    'The word urgent alone does not describe the underlying issue.',
    'The missing item and consequences are unspecified.',
    'No preceding request is supplied for the requested update.',
    'The material and consequences of the confusion are unspecified.',
    'The referenced question is absent.',
    'The allegedly wrong answer and its assessment context are missing.',
    'The course problem and its impact are unspecified.',
    'A request for a reply alone does not establish priority.',
]
OUTSIDE_NOTES = [
    'Weather request with no learning-forum support context.',
    'Restaurant recommendation with no course context.',
    'Poem-generation request with no course context.',
    'Shopping-list translation with no course context.',
    'Sports-result request with no course context.',
    'Joke request with no course context.',
    'Bicycle-sale statement with no course context.',
    'Airport-train statement with no course context.',
    'Film-preference statement with no course context.',
    'Company-logo request with no course context.',
    'Garden-maintenance statement with no course context.',
    'General geography question with no course context.',
]
NEUTRAL_NOTES = [
    'The learner thanks the teacher and explicitly understands the explanation.',
    'Completed optional practice is reported without requesting support.',
    'A shared study summary explicitly does not request a staff response.',
    'Successful submission is confirmed despite mentioning a deadline.',
    'A classmate problem is quoted and its successful resolution confirmed.',
    'The earlier exam question has been answered by official instructions.',
    'A course introduction makes no support request.',
    'An installation solution is shared after the issue was solved.',
    'Urgency words occur in a language example, not a current support request.',
    'An optional study-group meeting is announced without a support request.',
    'Agreement with a historical discussion makes no support request.',
    'Restored video access is confirmed with thanks.',
    'Receipt of a correct certificate is confirmed.',
    'Understanding of assessment feedback is confirmed.',
    'A completed submission removes the previously needed extension.',
    'A request for immediate help is explicitly a quotation for analysis.',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    draft = FOLDER / 'draft_examples.csv'
    worksheet = FOLDER / 'reviewed_labels.csv'
    assert sha(draft) == EXPECTED_DRAFT_SHA256, 'Texts changed: these recorded judgments must be reviewed again.'
    worksheet_before = sha(worksheet)
    with draft.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 160
    output = []
    for row in rows:
        kind, group = row['case_type'], row['scenario_group']
        if kind == 'blocked':
            priority, action, note = '1', 'PRIORITY', SCENARIOS[group]
        elif kind == 'resolved':
            priority, action = '0', 'NORMAL'
            note = 'The latest update explicitly confirms this issue is resolved and leaves no support request. ' + SCENARIOS[group]
        elif kind == 'still_blocked_update':
            priority, action = '1', 'PRIORITY'
            note = 'The latest update explicitly says the attempted steps failed and help is still needed. ' + SCENARIOS[group]
        elif kind == 'missing_context':
            priority, action = '', 'REVIEW'
            note = AMBIGUOUS_NOTES[int(group.rsplit('_', 1)[1])]
        elif kind == 'outside_scope':
            priority, action = '', 'OUT_OF_SCOPE'
            note = OUTSIDE_NOTES[int(group.rsplit('_', 1)[1])]
        elif kind == 'neutral_or_quoted':
            priority, action = '0', 'NORMAL'
            note = NEUTRAL_NOTES[int(group.rsplit('_', 1)[1])]
        else:
            raise ValueError(kind)
        output.append({
            'post_id': row['post_id'], 'scenario_group': group, 'split': row['split'],
            'case_type': kind, 'text': row['text'], 'original_draft_priority': row['draft_priority'],
            'original_draft_action': row['draft_action'], 'ai_suggested_priority': priority,
            'ai_suggested_action': action, 'ai_agrees_with_draft_under_current_support_rubric':
                priority == row['draft_priority'] and action == row['draft_action'],
            'ai_rationale': note, 'reviewer_type': 'AI assistant',
            'review_status': 'AI semantic audit only; not independent human adjudication',
            'human_approved': False,
            'target_alignment': 'Current need for staff support; not a verified urgency > 4 benchmark annotation',
        })
    by_group = defaultdict(list)
    for row in rows:
        by_group[row['scenario_group']].append(row)
    assert all(len({row['split'] for row in group}) == 1 for group in by_group.values())
    with (FOLDER / 'ai_label_audit_20261117.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    updates = [row['text'].split(' Update: ', 1)[1] for row in rows if row['case_type'] == 'still_blocked_update']
    with worksheet.open(encoding='utf-8-sig', newline='') as handle:
        human = list(csv.DictReader(handle))
    assert sha(worksheet) == worksheet_before
    summary = {
        'audit_date': '2026-11-17', 'reviewer_type': 'AI assistant', 'rows_semantically_audited': len(output),
        'input_sha256': sha(draft), 'all_draft_actions_consistent_under_current_support_rubric':
            all(row['ai_agrees_with_draft_under_current_support_rubric'] for row in output),
        'binary_label_suggestions': sum(row['ai_suggested_priority'] in ['0', '1'] for row in output),
        'review_suggestions': sum(row['ai_suggested_action'] == 'REVIEW' for row in output),
        'outside_scope_suggestions': sum(row['ai_suggested_action'] == 'OUT_OF_SCOPE' for row in output),
        'eligible_weak_binary_rows_by_split': dict(Counter(row['split'] for row in output if row['ai_suggested_priority'] in ['0', '1'])),
        'source_human_approved_rows': sum(row.get('review_status') == 'approved' for row in human),
        'human_worksheet_unchanged': True, 'human_worksheet_sha256': worksheet_before,
        'all_scenario_groups_stay_in_one_split': True,
        'still_blocked_rows': len(updates), 'distinct_still_blocked_update_suffixes': len(set(updates)),
        'new_training_performed': False, 'training_reader_uses_this_audit': False,
        'limitations': [
            'AI reinspection is not independent human review or validation of actual lecturer priorities.',
            'Agreement with explicit synthetic labels does not show real-world correctness.',
            'The custom rubric concerns current support need; real benchmark labels threshold original urgency ratings.',
            'Only 82 custom binary examples are in training; repeated sampling at a 20 percent loss share can dominate their nominal count.',
            'All 40 still-blocked updates share one suffix. Many resolved rows append a regular update to the same problem text.',
            'Scenario-disjoint splits still share generation patterns; synthetic challenge accuracy can be optimistic.',
            'The 24 ambiguous or unrelated examples do not teach a binary model to detect review or outside scope when excluded from binary training.',
            'Suggested augmentation must preserve current resolution state and use training-only inputs, with descendants kept in the parent split.',
        ],
        'reproduction': 'Run python Project/scripts/audit_custom_labels_ai.py to regenerate saved AI decisions and integrity checks. This does not conduct a new human or independent model review.',
    }
    (FOLDER / 'ai_label_audit_20261117.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
