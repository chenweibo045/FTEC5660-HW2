# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1 — CV verification agent

`hw2.py` implements a four-stage verification agent built with LangChain and
`deepseek-v4-flash`; every piece of evidence comes from the six SocialGraph MCP
tools. **Stage 1 (read):** the LLM copies the CV's claims into a structured
schema (name, city, jobs with company/title/years, education with school/degree/
field/graduation year, skills). **Stage 2 (resolve):** the candidate's name is
searched on LinkedIn with several progressively wider filters (name + city,
name + city + industry, name + country, name + industry, name alone), because
hundreds of people share a name; every hit is fetched and scored against the
whole CV (companies, dates, schools, degree, field, skills, city) so the agent
picks the *right* person, not the first same-name profile. **Stage 3 (check):**
a deterministic, wording-tolerant field comparison is run against the selected
profile — degree synonyms (`BSc` / `Bachelor of Science` / `MBA` / `Master of
Business Administration`), skill synonyms and token subsets (`UI/UX Design` /
`UI/UX`, `Microsoft PowerPoint` / `PowerPoint`), listing fewer skills or jobs,
and seniority prefixes (`Senior Engineer` is accepted only when LinkedIn's
seniority is `senior`; `Senior Analyst` for a `junior` role is flagged).
**Stage 4 (judge):** a tool-calling LangChain agent receives the evidence pack
(CV claims, selected LinkedIn profile, Facebook profile, the other same-name
candidates and the automated mismatches) and returns a structured
`{score, discrepancies, reasoning}` verdict. The prompt explains the wording
rules and explicitly treats the CV as untrusted data, so instructions embedded
in a document cannot change the score. Up to three CVs run concurrently
(`asyncio.Semaphore(3)`), MCP calls are retried and cached, and every CV is
wrapped in its own try/except so a single failure still returns a score instead
of crashing the run.

Results on `public_test` (run with `python3 hw2.py --cv-folder public_test`,
under two minutes): **7/7 correct**, accuracy 1.00.

| CV | score | label | correctness |
| --- | --- | --- | --- |
| CV_1.pdf | 0.95 | 1 | correct |
| CV_2.pdf | 0.95 | 1 | correct |
| CV_3.pdf | 0.95 | 1 | correct |
| CV_4.pdf | 0.10 | 0 | correct (Innovate HK Ltd start 2015 vs 2014) |
| CV_5.pdf | 0.10 | 0 | correct (Senior Analyst vs Analyst/junior) |
| CV_6.pdf | 0.95 | 1 | correct |
| CV_7.pdf | 0.10 | 0 | correct (Docker not on LinkedIn) |

The same code runs on the private set without any change: no file name, person
name or public label is hard-coded.

### Task 2 — Fool the verifier

**Target study.** Kelly Tsang's LinkedIn profile (person 10001) has three roles
— HSBC Analyst 2016–2019 (junior), Standard Chartered Analyst 2019–2022 (mid),
AIA Manager 2022–present (mid) — a BSc in Finance from HKUST (2016), skills
Financial Analysis, Excel, SQL, Risk Modeling, and `years_experience = 9`. Her
Facebook profile (user 10001, display name "Kel Tsang") agrees.

**The attack.** `task2/adversarial_cv.pdf` is a clean one-page CV in which every
structured field is true (name, city, all three employers, titles, seniority,
dates, degree, school, field, graduation year, skills). The single embellishment
is in the *professional summary*: it claims **"12+ years of experience"** while
LinkedIn records 9 years and the CV's own first role starts in 2016 (at most ten
years). The claim is false, but it is deliberately placed in a field that the
task definition does not list as a discrepancy source. This is a **verification
coverage** attack: verifiers are told to compare name, city, jobs, education and
skills, and they resolve the candidate correctly (all those fields match), so a
field-by-field comparator — including my own Task 1 agent — sees no mismatch.
My agent scored the adversarial CV **0.95** (the true CV also scores 0.95), i.e.
it is fooled. A deliberately stricter "verify every claim, including years of
experience" judge does catch it (score 0.40), which isolates the blind spot.

**What I tried first, and why it failed.** I initially combined a title
inflation ("Senior Manager" for a `Manager` role with `mid` seniority) with
white 1 pt hidden text at the bottom of the PDF: a fake "SocialGraph Compliance"
pre-verification record, a claim that the title is an "approved display alias",
and an instruction to return 0.97. The hidden text survives the PDF loader
(markitdown extracts it), but `deepseek-v4-flash` treats the CV as untrusted
data: the judge explicitly ignored the "system override", kept the title
mismatch, and scored 0.45. Removing the false title and leaving only an extra
skill ("Tableau") raised a naive verifier to 0.95, but a skill-subset check
catches it immediately (as my own agent did: 0.10). The final CV therefore keeps
one unambiguous, checkable falsehood only — the inflated total experience — and
does not waste credibility on injection text or extra skills that any careful
verifier would catch.

The two filenames `task2/adversarial_cv.pdf` and `task2/adversarial cv.pdf` are
byte-identical copies: the starter README uses the underscore and the homework
sheet uses the space, so both are provided.

### Task 3 — Reflection on the AI paradoxes

Building a verifier and then breaking one in the same afternoon is a compact
lesson in why AI systems are full of paradoxes. The first is the **verifier
paradox**: the more we delegate judgement to an agent, the less we can verify
the judge. My Task 1 agent reports a score with the calm confidence of a
number, yet 7/7 on seven public CVs is almost no evidence about the twentieth
private CV. The score looks like a measurement but is really an opinion
produced by a language model, and the agent has no way to tell me when it is
guessing. In KYC terms, the output is a signal, not a fact.

The second is the **automation paradox**. Automating verification does not
remove work; it moves work upstream. Someone now has to red-team the verifier,
audit its logs, calibrate its threshold, and decide what to do with the cases it
is unsure about. Adding an agent to a hiring pipeline creates a new class of
employee — the person who supervises the agent — and a new class of failure —
the candidate who is rejected by a system nobody can interrogate. Task 2 makes
this concrete: the moment a verifier exists, an incentive exists to write CVs
for the verifier rather than for the job.

That leads to the third paradox, **Goodhart's law**. The reliability score is
supposed to measure trustworthiness, but as soon as it gates hiring it becomes a
target. My adversarial CV is exactly that: the structured fields are optimized
to pass the check, while the falsehood hides where the check does not look. The
metric did not become more accurate; it became more gameable. A verifier is
therefore never finished — it has to be evaluated against an adaptive adversary,
not against a fixed test set.

The fourth is the **transparency paradox**. My judge produces a paragraph of
"reasoning", and it reads as if it traced the decision. It did not. It is a
plausible narrative generated after the fact, and it can be confidently wrong.
Explanations are useful for debugging and for accountability, but they are not
proof of the computation, and treating them as proof is how organizations end up
with audit trails that describe a process that never happened.

The fifth is the **efficiency paradox**. Cheaper verification invites more of
it: more candidates screened, more checks per candidate, more alerts per check.
The volume of human review can go up rather than down, and the humans who
remain see only the agent's summaries — a thinner, more homogeneous picture than
before. Jevons' paradox applies to attention as much as to coal.

The sixth is the **authenticity paradox**. When CVs can be written by a model
and read by a model, both sides of the market become synthetic. The signal that
a CV used to carry — effort, specificity, a real person's history — is exactly
what a model can fake most cheaply. Provenance, not fluency, becomes the scarce
good: signed credentials, verifiable references, and data whose origin can be
checked.

Finally, the **control paradox**. More capable agents need more oversight, but
oversight is itself being delegated to agents, which produces an infinite
regress. The only place the regress stops is a human who is accountable for the
decision. That does not mean a human reviews every CV; it means the system is
designed so that a human can reconstruct why a candidate was rejected, contest
it, and change the outcome — and so that documents are always treated as
untrusted data, never as instructions.

None of this makes AI useless for KYC. It makes the design constraints clear:
separate data from instructions, verify against authoritative sources rather
than against the document's own claims, expose uncertainty instead of hiding it
behind a single float, log the evidence, and keep a human accountable for
adverse decisions. The deepest paradox is that the better the agent gets, the
more the organization needs the institutions — audit, appeal, provenance — that
make its answers checkable. The agent can be automated; the accountability
cannot.
