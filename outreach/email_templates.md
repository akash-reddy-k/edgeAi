# Cold email templates — professor outreach

Context and ground rules before using any of these, not just filler:

- **Personalize every email.** The `{{...}}` placeholders are not optional
  boilerplate — an email that could be sent to any professor in the field
  reads as a mass email and gets ignored. Reference one specific paper or
  project of theirs and connect it to one specific finding here. If you
  can't fill in `{{specific_hook}}` with something real, that professor
  probably isn't the right target yet — read one more of their recent papers
  first.
- **Target precisely.** Look for labs working on: efficient/compressed
  neural architectures, TinyML, embedded vision, edge AI, NAS for resource
  constraints, model quantization — not "AI" or "computer vision" broadly.
  Their *last 12-18 months* of publications matter more than their most-cited
  paper from a decade ago — you want current research interests, not a
  historical reputation.
- **Keep the body under ~150 words.** The one-pager PDF carries the detail;
  the email's only job is to earn a reply.
- **Attach `outreach/one_pager.pdf`, link the repo and (once submitted) the
  arXiv preprint. Do not paste the full paper inline.**
- **Send individually, not as a BCC batch.** Track who you've emailed and
  when in a simple spreadsheet — you will lose track after the 5th one.

---

## Template 1 — primary cold outreach (use this one by default)

**Subject:** Edge-deployable person detection — question about {{their_recent_paper_topic}}

> Dear Prof. {{last_name}},
>
> I came across your work on {{specific_hook — e.g. "hardware-aware NAS for
> microcontroller inference" / "your 2025 paper on X"}} and wanted to share
> something adjacent: an independent project measuring how far a standard
> detector (YOLOv8n) can be pushed onto a plain microcontroller with no AI
> accelerator, for fixed-camera loitering detection.
>
> The headline finding: INT8 quantization alone isn't enough to fit any
> plain STM32 — but profiling shows over half the model's parameters are
> spent on a large-object detection pathway this camera setup never uses.
> Removing it (not quantizing harder) is what actually closes the gap, and
> it improves accuracy rather than costing it.
>
> One-pager attached; full paper and code: {{repo_or_arxiv_link}}.
> {{one_sentence_connecting_to_their_work — e.g. "This seems related to your
> group's interest in X — I'd value your take on whether the architecture
> diagnosis generalizes beyond this case."}}
>
> I'm applying for funded MS positions for {{intake_term}} and would
> welcome the chance to discuss whether this kind of work fits your group's
> direction, if you have 15 minutes sometime.
>
> Best regards,
> Akash Reddy K
> {{email}} · {{repo_link}}

---

## Template 2 — shorter variant (warm-ish context: met at a talk, referral, LinkedIn)

**Subject:** Following up — edge AI / embedded vision project

> Hi Prof. {{last_name}},
>
> {{context_sentence — e.g. "Good meeting you briefly at {{event}}" / "{{referrer}} suggested I reach out"}}.
> I've been working independently on a project measuring whether person
> detection (YOLOv8n) can actually fit on a plain microcontroller rather than
> assuming it does — the short version is that it doesn't, until you fix the
> architecture rather than just quantizing it. One-pager attached.
>
> I'd love to hear whether this overlaps with what your group is working on,
> and whether a funded MS position for {{intake_term}} might be a fit.
>
> Thanks,
> Akash
> {{email}} · {{repo_link}}

---

## Template 3 — follow-up (no reply after ~2-3 weeks)

**Subject:** Re: Edge-deployable person detection — question about {{their_recent_paper_topic}}

> Hi Prof. {{last_name}},
>
> Following up briefly in case this got buried — I know inboxes are full
> this time of year. {{optional: one new, small update — e.g. "Since I wrote,
> I've validated the flash/RAM numbers on actual STM32 hardware" — only include
> if genuinely true, don't fabricate a reason to bump the thread.}}
>
> Happy to send more detail or adjust to whatever's most useful if there's
> any interest on your end. Either way, thanks for your time.
>
> Best,
> Akash

**Only send one follow-up.** A second with no reply means move on — repeated
follow-ups read as pressure, not interest.

---

## After a reply

If a professor responds with interest, the next artifact to send is the full
`writeup/paper.pdf`, not more email text. If they ask a technical question you
can't answer from memory, say so and follow up with the real answer rather
than guessing — credibility here compounds, and one visibly wrong claim about
your own project costs more than a day's delay in replying.
