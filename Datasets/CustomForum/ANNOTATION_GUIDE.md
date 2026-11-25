# CustomForum annotation guide

These are synthetic drafts. They are not an independently labelled real-world benchmark.

Read the whole text, including any final update. Judge the current need for prompt teaching-team support.
- PRIORITY / 1: a current, unresolved course issue needs a timely staff response.
- NORMAL / 0: informational, resolved, or no current actionable request.
- REVIEW / blank: context is insufficient or the correct label is debatable.
- OUT_OF_SCOPE / blank: the text is not a supported student forum post.

The original source benchmark still uses urgency > 4. Its labels are not overwritten.
Enter reviewer name, adjudicated action, and (for PRIORITY/NORMAL only) adjudicated_priority in reviewed_labels.csv.
Set review_status to approved only after review. For disputed cases, obtain a second independent judgement and record the resolution.
The draft file can be regenerated. The review worksheet is never overwritten by the generator.
Only approved train rows may enter a training ablation. Validation/challenge scenarios remain held out, including every variant of a scenario.
Performance on approved synthetic scenarios is reported separately from real UPenn/Stanford test performance.
