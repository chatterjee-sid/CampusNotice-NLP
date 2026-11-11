# Annotation Rules (v1)

Labels live in `labels.json`, keyed by the `.txt` file name in `data/texts/`.
Labels were reviewed by the user for category, dates and fees on 2026-11-11 (`"status": "user_reviewed"`); free-text fields were not checked line by line.

## Category (exactly one)
| Label | Use for |
|---|---|
| `placement` | Campus recruitment / internship drives, interview and test schedules (CDC mails) |
| `exam` | Exam timetables, academic calendars, exam/course registration, project report submission |
| `hostel` | Hostel allotment, mess, hostel charges, hostel-resident instructions |
| `general` | Everything else: closures, events, committees, scholarships, fee remission, lectures, health, sports, programmes |

Tie-break: pick the category of the **action the student must take**. Fee and scholarship notices are `general`
(too few examples for their own class); their amounts and deadlines are still extracted as fields.

## Fields
- `deadline_dates` (ISO `YYYY-MM-DD`): last date by which an action is still possible (last date, on or before, till, by).
  Relative words ("tomorrow", "Monday") are resolved against the mail's date. Include every distinct deadline in the notice.
- `event_dates`: dates something happens (exam day, interview, orientation, effective-from date). Ranges list start and end
  (calendars/timetables list each exam day). Not deadlines, not the notice's issue date.
- `notice_date`: date the notice was issued (used for resolving relative dates; not an extraction target).
- `fees`: integer rupee amounts a student must pay or is charged (fee, fine, charges, stamp paper).
  **Not fees:** salary/CTC/LPA/stipend, income limits, scholarship amounts, room numbers, phone numbers.
- `eligibility`: one short string per condition (CGPA, percentage, backlogs, branch, batch, programme/year, category).
- `required_documents`: things the student must submit, bring or attach (forms, certificates, resume, receipts).
- `action_items`: one short sentence per thing the student is asked to do.
- Quoted/forwarded text: label the **latest** message's facts, plus deadlines/eligibility that the thread still relies on.

## Evaluation flags
- `group`: near-duplicate documents (same notice as mail + PDF, or attachments of one mail). **Members of a group must stay in
  the same train/dev/test split** to avoid leakage.
- `extraction_eval: false`: table-heavy documents (calendars, committee lists, branch tables) - category is evaluated,
  field extraction is not.
- `synthetic`: `true` only for documents written by us (none so far).

## Privacy
Student names, roll numbers, e-mail addresses and phone numbers are masked in `data/texts/` (`[STUDENT]`, `[ROLL]`,
`[EMAIL]`, `[PHONE]`, `[NAME]`). Some staff names in signatures may remain.
