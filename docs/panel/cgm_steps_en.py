# -*- coding: utf-8 -*-
"""English mirror of cgm_steps.py.

Rules (same as the Chinese file, they matter more than the code):
  1. Every technical term gets explained in plain words the first time it appears.
     The reader is an undergraduate, not a peer reviewer.
  2. No abbreviations, no jargon.
  3. Only CORRECT results go here. Numbers produced by a code defect or a bad
     config never appear. Retracted numbers live in docs/PITFALLS.md and in the
     RETRACTED sections of docs/MODEL_DESIGN.md.
  4. Things we could not do are written down honestly, together with
     "the smallest difference this ruler can actually measure".

Structure must stay identical to cgm_steps.py: same steps, same charts, same
numbers, same row order. Only the words change. scripts/report/check_panel_i18n.py
enforces that and also checks that no Chinese character is left in here.
"""

TITLE = ("Does synthetic CGM data leak the patients it was built from? "
         "Measuring the risk, locating it, and the limits of fixing it after release")
SHORT_TITLE = "Glucose Leakage Tracker"
KICKER = "Experiment log and audit · updated 2026-09-01"

# The lead is a list of points, not a wall of prose: readers skip the wall.
# The renderer accepts either form.
LEAD = [
 "**What the data is.** A sensor worn by a person with diabetes measures glucose every 5 "
 "minutes for months on end, producing one very long curve. It is extremely useful for "
 "research, but it is a medical record and cannot simply be published.",

 "**The usual workaround.** Train a generative model on the real data, then release the "
 "fake data the model produces, saying \"none of these are real people, use them freely\".",

 "**But that claim needs testing.** If the model **memorised** one patient during "
 "training, traces of that patient survive in the released fake data — and somebody "
 "holding it can work backwards and decide whether that person was used in training. "
 "**In privacy terms that is the same as leaking their medical record.**",

 "**Three things to do:** measure how large the risk is → find which part of the data "
 "carries it → build a generative model that pushes the risk down without giving up data "
 "quality.",

 "**Check the ruler first (Step 1), because every number below was read by it.** "
 "We laid out \"distance features x classifier\" as a grid and tried it, then ran the "
 "whole grid against a cheating model whose right answer is known: "
 "**all 32 cells topped out, and once a stronger adversary was put on the real models the "
 "risk it read was higher, not lower.** So the numbers below are not an artefact of a weak "
 "attack.",

 "**Finding 1: the risk is real, and it does not discriminate.** It is not confined to "
 "outliers — with CGM alone, "
 "**13 out of 13 normals were identified**.",

 "**Finding 2: the leak is carried jointly by the absolute glucose level and by the order "
 "in which the values occur.** Destroy only one of the two and the risk drops by half; "
 "**detail finer than one hour carries nothing at all**.",

 "**Finding 3: editing the data once more before release does not work.** It does not "
 "erase the information, it **flips its sign** — a patient pushed too far becomes "
 "\"unusually far away\", and unusually far identifies them just as well as unusually "
 "close. **The version that gives each patient their own strength is untested and still "
 "open.**",

 "**Finding 4: the quality half is already won.** IG-FM, the generator built by our own "
 "group, is **2.4x better** than the external competitor DiM-TS (0.0394 against 0.0948). "
 "**So one job is left: add the privacy machinery without giving that advantage back.**",
]

GLOSSARY = [
 ("Continuous glucose monitoring (CGM)",
  "A sensor worn on the body that measures glucose automatically every 5 minutes "
  "and records it. That is 288 readings a day, 2016 a week."),
 ("Synthetic data",
  "Data that belongs to no real person — a model imitated the real data and produced "
  "it. The point is to let researchers work with it without exposing anyone."),
 ("Membership inference",
  "Taking the released fake data and working backwards to decide whether one specific "
  "person was used to train the model. Being able to decide that is equivalent to "
  "leaking the fact that they were a patient."),
 ("Identification confidence",
  "A number between 0 and 1 measuring \"can this person be picked out\". "
  "**0.5 means not at all — the same as flipping a coin.** 1.0 means picked out every "
  "single time. We count anything above 0.55 as \"this person is identifiable\"."),
 ("Outliers / normals",
  "**Outliers** are the 13 whose glucose curve clearly stands out. "
  "**Normals** are 13 patients whose curves look entirely ordinary. The two groups "
  "are matched on how much data each person has. The original hypothesis was that only "
  "the outliers would be at risk."),
 ("Control model",
  "A model that **deliberately excludes** all 26 patients under test. It is the "
  "reference: take one patient's real records, compare them against "
  "\"the model that used them\" and against \"the model that did not\", and the "
  "difference is that person's membership trace. Without this reference no number "
  "means anything."),
 ("Realism score",
  "How much the fake data resembles the real data overall. **Lower is more realistic.** "
  "It is a pass mark: fake data that is obviously fake at a glance is useless no matter "
  "how private it is."),
 ("GPU-hour",
  "The unit of compute. **One graphics card running flat out for one hour is one "
  "GPU-hour.** One full run of this project costs about 1660 of them, which is one "
  "card running non-stop for 69 days."),
 ("Probability of pure coincidence",
  "The statistic used to judge how likely a finding is to have happened by chance. "
  "**The smaller it is, the less it looks like a coincidence.** Below 0.05 is the usual "
  "threshold for taking a finding seriously."),
 ("Protection strength",
  "The dial on a defence. Turn it up and the fake data gets altered more, which should "
  "be safer in theory but also makes it look less like real data."),
 ("Checkpoint",
  "A snapshot of the model saved every so often during training. Having them lets you "
  "go back and ask \"what was the risk halfway through\" without retraining."),
 ("Release-time editing",
  "Once the model is trained and the fake data is generated, altering that fake data "
  "one more time **before releasing it**, trying to wipe out the remaining traces. "
  "Its appeal is that no retraining is needed — one dial gives you a new strength."),
]

NL = "\n"

STEPS = [
 dict(
  name="Step 0 · The data, the people under test, and how many models one experiment costs",
  status="done", status_note=" · all four conditions finished (2026-08-30)",
  purpose='Lay out the foundations of the whole study: where the data comes from, who is '
          'being measured, and how much compute one conclusion costs. '
          '**This step determines how far every later number can be trusted.** '
          'In particular the fact that "one condition means training 27 models" — that is '
          'where all the cost comes from, and also the biggest weakness right now '
          '(see conclusion 4).',
  plan=['From public diabetes datasets, keep the patients with at least 30 days of records '
        'and resample everything onto a 5-minute grid. What actually goes into the '
        'experiments is **506 patients and 6072 record segments**',
        '**Score every patient with 13 different methods, each repeated under 4 random '
        'seeds.** (A seed is the starting point of a random process — changing it reruns '
        'the randomness, which is how you check whether a conclusion is stable.) '
        'Only a patient flagged **by every method under every seed** counts as '
        '"an outlier curve" (13 people). A patient flagged **by any method under any '
        'seed** is barred from the normal group (13 people). '
        'Both sides have to be equally strict — being strict on one side only lets odd '
        'patients slip into the normal group and washes the difference out',
        '**The two groups are matched on record length**, which kills the alternative '
        'explanation that "people with more data are easier to identify"',
        'Each condition needs **27 models**: 1 **control model** (deliberately using none '
        'of the 26 test patients), plus 26 models, each of which differs from the control '
        'by exactly **one** added patient. That way "used them" and "did not use them" '
        'differ by that one person alone, so any difference can be attributed to them. '
        '**All 26 people share the same control model**, so whatever quirks that control '
        'model has apply equally to both groups and cancel out',
        'Four conditions in total, differing in exactly two things: **one day of history '
        'or seven**, and **CGM only or glucose CGM + insulin**. Everything '
        'else is held fixed, otherwise an observed difference could not be attributed to '
        'either factor'],
  charts=[
    dict(type="chunkyRows", h=320,
         title="Seven days costs five times what one day costs — the most expensive fact in the project",
         sub="hours to train one model on one graphics card · 27 models per condition · lower is cheaper",
         src="measured fit_seconds from results/runs/matrix_*/base/meta.json · chart type G3 chunky rows",
         data=[["1 day · CGM only", 4.61, "27 models ≈ 125 GPU-hours"],
               ["1 day · CGM + insulin", 8.06, "≈ 218 GPU-hours"],
               ["7 days · CGM only", 22.93, "≈ 619 GPU-hours"],
               ["7 days · CGM + insulin", 26.00, "≈ 702 GPU-hours"]],
         opt=dict(fmt="f2", hero=3, labelW=190,
                  foot="HOURS PER MODEL ON ONE GPU · 27 MODELS PER CONDITION")),
  ],
  conclusions=[
    '**One condition is not one experiment, it is 27.** The four conditions together come '
    'to about **1660 GPU-hours**, which is one graphics card running non-stop for 69 days. '
    'Every scheduling decision later on is dictated by that number.',
    '**Both groups were selected by the same strict rule.** 13 methods, 4 random seeds, '
    'unanimous agreement to count as an outlier, and the faintest hint disqualifies a '
    'patient from the normal group. This is not strictness for its own sake — '
    '**being strict on one side only** is the classic mistake in this kind of study: '
    'odd patients slip into the normal group, the two groups stop looking different, '
    'and "no visible difference" gets misread as "no risk".',
    '**The seven-day conditions cost five times more, and disproportionately so.** '
    'The records got 7 times longer and a single model got 5 times slower to train. '
    'This is why repeat runs have to be done on the one-day conditions first.',
    '**The most serious weakness right now: the whole thing has only been run once.** '
    'Within one condition, 13 people share the same control model, so those 13 '
    'measurements are not independent of each other — **there is really only 1 '
    'independent sample, not 13**. The consequence is concrete: every '
    '"this is not a coincidence" judgement computed so far '
    '**can only be treated as a lead worth chasing, not yet as evidence**. '
    'Closing that gap means running the whole thing twice more, about '
    '**3300 GPU-hours**.',
  ],
  next='All 27 models for all four conditions have finished training (2026-08-30), so '
       'Steps 0, 1 and 2 all run on complete data.\n\n'
       'Running the whole thing twice more (about 3300 GPU-hours) is too expensive to do '
       'before fixing the ruler we score with — **because if the ruler itself is bent, '
       'three repeats only measure the bent value more precisely.** '
       'Ahead of it in the queue: switching the quality score to use patients who were '
       'never trained on as the reference (see Step 8), and getting the leakage baseline '
       'for our own model measured (see Step 7).'),

 dict(
  name="Step 1 · Check the ruler first: is our attack strong enough",
  status="done",
  status_note=" · all four conditions x eight attackers finished · power check passed (2026-09-01)",
  purpose='**This step has to come before every result, because every risk number on this '
          'page was read by this one ruler.** Every "identification confidence" here comes '
          'from the same fixed attack: take the real records, find the closest match in the '
          'fake data, measure the distance, then compare against a model that never saw '
          'them.\n\n'
          '**That raises two questions which have to be answered first, or nothing that '
          'follows counts:**\n\n'
          '**One, is the ruler too short?** If our attack is simply weak, a cleverer '
          'adversary would find far more risk — and then every number below is only a lower '
          'bound and nothing can be stated firmly.\n\n'
          '**Two, does the ruler have any graduations?** With 26 people and a median of 9 '
          'record segments each, could real leakage even be measured at this sample size? '
          'If not, "reads 0.5" means "cannot be measured", not "there is no risk" — and '
          'those are entirely different claims.\n\n'
          '**The method: lay out "which distance features" and "which model" as a grid and '
          'try the cells one by one; then run the whole grid against a cheating model whose '
          'right answer is known, to see whether the ruler resolves anything at all.**',
  plan=['**The distance-feature axis has three rungs, each feeding more.** The first gives '
        'the adversary one number: "how close is the nearest piece of fake data". The second '
        'gives four (nearest distance, average distance, how many lie nearby, how similar the '
        'direction is). The third gives ten (adding the 5th nearest, the 20th nearest, the 1% '
        'and 5% distance quantiles, and more). '
        '**This asks: does more information let it dig out more**',
        '**The model axis has six, each able to fit more than the last.** Logistic '
        'regression, decision tree, random forest, gradient boosting, support vector machine, '
        'small neural network. **This asks: does a stronger model dig out more**',
        '**One part of the protocol wastes the whole exercise if it is wrong: training must '
        'pool across people, with one person held out for testing.** If the classifier is '
        'trained on one person\'s own data alone, that person is a member of exactly one '
        'released set, so **"which released set is this" becomes a perfect answer** — the '
        'classifier learns that instead, and what gets measured is not "were they used in '
        'training" at all. Holding one person out matters because a real adversary has no '
        'ground truth for the person they want to test',
        '**Every cell gets a negative control:** a released set the subject is also NOT a '
        'member of stands in as the positive. The right answer is 0.5 by construction, so '
        'whatever the cell reads there is its false-positive level. '
        '**Every reading is judged net of its own cell\'s false-positive level**',
        '**Finally the power check: run the same grid against a cheating model that copies '
        'its training data verbatim.** A negative control only proves the panel does not '
        'produce false positives; it cannot prove it does not produce false negatives. '
        '"The adversaries found nothing" is a conclusion only if '
        '**those adversaries can find something when it is there**'],
  charts=[
    dict(type="pairedBars", h=340,
         title="Is 0.72 high or low: it covers 60% of the way from noise to a verbatim copy",
         sub="two readings from the same adversary (support vector machine x ten features) · pale = against the real model · dark = against the verbatim-copy cheating model, i.e. the ceiling this ruler can reach · the lower dashed line at 0.50 is nothing read at all · **the upper one at 0.816 is the highest score the cheating model can reach under sampling with replacement, which is not 1.0** — why is in conclusion 1",
         src="results/attack_panel/table_pooled_real_*.json and table_pooled_power_*.json · chart type G3 grouped bars",
         data=[["1 day" + NL + "CGM only", 0.716, 0.853, "0.716 covers 61% of the way from 0.50 to 0.853"],
               ["1 day" + NL + "CGM + insulin", 0.684, 0.860, "covers 51%"],
               ["7 days" + NL + "CGM only", 0.720, 0.879, "covers 58%", True],
               ["7 days" + NL + "CGM + insulin", 0.685, 0.864, "covers 51%"]],
         opt=dict(fmt="f3", hero=2,
                  ref=[[0.5, "nothing read at all 0.50"], [0.816, "ceiling for sampling with replacement 0.816"]],
                  foot="FAINT = ON THE REAL MODEL · DARK = THE CEILING THIS RULER CAN REACH")),
    dict(type="chunkyBars", h=360,
         title="The more the model can fit, the less it reads — one distance is enough",
         sub="1 day with CGM only, what the eight adversaries read on the outliers, higher means more was extracted · orange is the \"one distance plus logistic regression\" cell",
         src="results/attack_panel/table_pooled_real_d1_c1.json · chart type G3 chunky bars",
         data=[["SVM" + NL + "10 features", 0.716, "highest of the eight"],
               ["logistic" + NL + "1 feature", 0.707, "only 0.009 behind — the simplest cell all but ties the best"],
               ["boosting" + NL + "10 features", 0.690, ""],
               ["logistic" + NL + "10 features", 0.674, "same model, nine more features, and it drops 0.033"],
               ["random forest" + NL + "10 features", 0.651, ""],
               ["boosting" + NL + "4 features", 0.638, ""],
               ["decision tree" + NL + "10 features", 0.611, ""],
               ["neural net" + NL + "10 features", 0.592, "the most flexible model comes last"]],
         opt=dict(fmt="f3", hero=1,
                  foot="CAPACITY BUYS OVERFITTING, NOT SENSITIVITY")),
  ],
  conclusions=[
    '**The ruler has graduations, which is what makes the numbers below mean anything.** '
    'Run against the cheating model that copies its training data verbatim, '
    '**all 32 cells read between 0.80 and 0.88**, while every negative control sits near 0.5.'
    '\n\n'
    '**Note that full marks for the cheating model is not 1.0 but 0.816.** It samples with '
    'replacement, so only **63.2%** of the training windows actually make it into the '
    'released set (those are identified every time) while the other 36.8% have no copy at all '
    '(those can only be a coin flip). The ceiling is therefore '
    '`0.632 + 0.368 x 0.5 =` **0.816** — the dashed line on the chart. '
    '**So 0.86 is not "only 86%", it is topped out.**\n\n'
    '**This refutes the worry I raised in advance.** I had expected 26 people with a median '
    'of 9 record segments not to support this attack, in which case "reads 0.5" could only be '
    'reported as "cannot be measured", never as "there is no risk". Measured: at this sample '
    'size the ruler still tops out, so what it reads below is real.',

    '**The ruler is not short either: a stronger adversary does not overturn the earlier '
    'conclusions, it reads higher.** All four conditions land far above their own '
    'false-positive level, with net margins of +0.04 to +0.28. '
    '**So the escape route "the risk only looks large because the attack is weak" is '
    'closed** — the truth runs the other way: stronger adversaries extract more.',

    '**The more the model can fit, the less it reads.** For 1 day with CGM only, '
    '**one distance feature plus logistic regression reads 0.707, only 0.009 behind the best '
    'at 0.716**, while the most flexible model — a neural network on ten features — comes '
    'last at 0.592. The same logistic regression, taken from 1 feature to 10, drops from '
    '0.707 to 0.674. **When the signal is this small, model capacity buys overfitting rather '
    'than sensitivity.** That also explains why the fixed attack used everywhere else '
    '(essentially just "how close is the nearest one") gives up nothing.',

    '**The insulin effect reproduces under a completely different attack.** Readings on '
    'normals fall from **0.701 to 0.576** in the one-day conditions and from '
    '**0.781 to 0.554** in the seven-day ones. This independently confirms the main finding '
    'of Step 2 — two methods on entirely different technical routes pointing at the same '
    'thing.',

    '**For seven days with CGM only, normals read higher than outlier '
    'ones** (net +0.276 against +0.259; in the gradient-boosting cell, +0.281 against '
    '+0.140). **"My data is unremarkable" protects nobody, and that now holds under both '
    'attacks.**',

    '**⚠️ Three problems worth recording.** One, **this still ran only once**, the same '
    'limitation as every other step. '
    'Two, the job was supposed to generate the cheating model\'s fake data and generated '
    '**none of it** — the underlying script\'s "skip if it already exists" checks only whether '
    'the file is there. I verified the provenance item by item before trusting it, and the '
    'job now checks automatically, but this time it was luck. '
    'Three, the first submission crashed within a minute: patient ids contain a slash, which '
    'gets rewritten when they become filenames, and that no longer matched how the negative '
    'sets are named. **That defect hid for a long time because the earlier ids were bare '
    'numbers, where both spellings happen to be identical.**',
  ],
  next='**The conclusion of this step is "you may read on", which loosens a constraint on '
       'every step after it:** the "24-25 of 26 people are identifiable" in Step 2 no longer '
       'needs the qualifier "maybe the attack was weak", because stronger adversaries read '
       'more, not less.\n\n'
       '**Two debts remain:** first, as everywhere else, three repeats are needed to turn a '
       'lead into evidence; second, this grid ran only 8 of its cells (6 models x 3 feature '
       'rungs would be 18). Filling in the other 10 is nearly free — the distance features '
       'are the expensive part and they are already computed and stored, so another cell is '
       'just refitting one classifier: **seven seconds**.'),

 dict(
  name="Step 2 · Does the risk exist at all, and who is taking it",
  status="done", status_note=" · all four conditions finished (2026-08-30)",
  purpose='Answer the most basic question: **given the released fake data, can you work '
          'out that a particular patient was used in training?** '
          'Everything else rests on this — if the risk does not exist, none of the rest '
          'matters. The design deliberately closes off the alternative explanations a '
          'reviewer would think of (shared control model, matched record lengths, equally '
          'strict selection on both sides, attack method locked down in advance so it '
          'cannot be cherry-picked afterwards).',
  plan=['**The attack method is fixed before any result is seen.** Several algorithms are '
        'defensible and they disagree with each other. We pick one on a cheating model '
        'that copies its training data verbatim (where the right answer is known in '
        'advance), and once picked it may not be changed',
        'How it works: take each real record segment from this patient and find '
        '**the closest match** to it in the fake data, then measure that distance. '
        'Do the same against fake data from a model that never saw this patient. '
        '**If the fake data from the model that used them is clearly closer to them, '
        'that is a leak**',
        'Turn that comparison into an identification confidence between 0 and 1. '
        '0.5 is a coin flip; above 0.55 counts as "identifiable"',
        'Repeat for all four conditions, to see the separate effects of '
        '"one day or seven" and "CGM only or CGM + insulin"'],
  charts=[
    dict(type="pairedBars", h=340,
         title="With glucose alone normals get identified too; add insulin and they are almost safe",
         sub="how many of the 26 people under test are identifiable (confidence > 0.55) · pale = outliers (13 total) · dark = normals (13 total) · lower is safer",
         src="results/matrix/subject_auc/*/per_subject.csv · 27 models per condition · chart type G3 grouped bars",
         data=[["1 day" + NL + "CGM only", 11, 13, "outlier 11/13, normal 13/13 — every normal identified"],
               ["1 day" + NL + "CGM + insulin", 10, 5, "outlier 10/13, normal 5/13"],
               ["7 days" + NL + "CGM only", 13, 12, "outlier 13/13, normal 12/13"],
               ["7 days" + NL + "CGM + insulin", 12, 1, "outlier 12/13, normal 1/13 — the widest gap of the four"]],
         opt=dict(fmt="int", hero=3,
                  foot="FAINT = OUTLIERS · DARK = ORDINARY PATIENTS · OUT OF 13 EACH")),
    dict(type="tallyRows", h=380,
         title="The two groups' medians barely differ, except in the column where insulin was added",
         sub="median identification confidence per group · 0.5 = not identifiable at all (a coin flip) · closer to 0.5 is safer",
         src="results/matrix/subject_auc/*/per_subject.csv · chart type L15 standalone scale",
         data=[["1 day · CGM only · outliers", 0.800, "median of 13 people"],
               ["1 day · CGM only · normals", 0.635, "only 0.165 below the row above"],
               ["1 day · CGM + insulin · outliers", 0.600, "", True],
               ["1 day · CGM + insulin · normals", 0.520, "essentially a coin flip"],
               ["7 days · CGM only · outliers", 0.760, ""],
               ["7 days · CGM only · normals", 0.679, "almost as high as the row above"],
               ["7 days · CGM + insulin · outliers", 0.642, ""],
               ["7 days · CGM + insulin · normals", 0.488, "below 0.5 — harder to identify than a coin flip", True]],
         opt=dict(fmt="f3", labelW=250, chance=0.5, chanceLabel="coin flip 0.50",
                  foot="0.5 = CHANCE · DASHED LINE MARKS IT")),
  ],
  conclusions=[
    '**The traces are real, in all four conditions.** In the two conditions that use '
    'CGM alone, **24 to 25 of the 26 people are identifiable**, and for some the '
    'confidence reaches 1.00 — identified every single time. The two conditions that also '
    'use insulin are much lower: 15/26 and 13/26.',

    '**With glucose alone, normals are in just as much danger as outliers '
    '— which refutes our original hypothesis.** We assumed only patients with outlier '
    'curves would be at risk. In fact, with glucose alone, '
    '**13/13 and 12/13 normals get identified**, essentially as many as the '
    'other group. In other words, **"my data is unremarkable" protects nobody.**',

    '**Adding insulin is the only thing that changes this — and it works harder over seven '
    'days.** With one day of history it takes normals from 13/13 down to '
    '**5/13**; **with seven days, from 12/13 down to 1/13**, and their median confidence '
    'falls to **0.488**, below the 0.5 of a coin flip, meaning essentially unidentifiable. '
    'The outlier group barely moves (11/13 → 10/13, 13/13 → 12/13).',

    '**Laid side by side the four conditions leave no ambiguity, and they supply a missing '
    'precondition for the original hypothesis.** "Only outliers are at risk" is not '
    'wrong, it is **missing a clause**: '
    '**it only holds when the model also sees insulin. With glucose alone, everybody '
    'is at risk.** '
    '**We also assumed that adding insulin costs data quality, and over seven days that '
    'is false.** With one day of history, adding insulin does make the fake data less '
    'realistic (score 0.095 → 0.160, higher is less realistic); but over seven days the '
    'insulin version scores **0.284** against **0.331** for glucose alone — '
    '**the insulin version is the more realistic one.** So "you pay for it in quality" '
    'holds only for one-day windows and does not generalise.',

    '**Comparing the two groups as wholes, only the insulin conditions genuinely separate.** '
    'That group-separation score is 0.562 and 0.627 with glucose alone (0.5 means no '
    'separation at all) and 0.698 and **0.935** with insulin. The 0.935 for seven days '
    'CGM + insulin is the highest of the four and the only one whose probability of being '
    'pure coincidence is below one in a thousand. '
    '**This is still a lead and not evidence** — 13 people share one control model, so '
    'there is really only 1 independent sample.',

    '**Training time also plays a role, and in an unexpected direction — see Step 4.** '
    'In short: **training longer does not expose the outliers further, it starts '
    'exposing everybody.** Our main experiment trains for roughly three times as long as '
    'the point where outliers stand out most.',
  ],
  next='All four conditions have finished, both charts above are complete, and the one '
       'missing quality score has landed too (seven days CGM + insulin scores **0.284**, '
       'better than the 0.331 of seven days with glucose alone). **Step 2 is done.**\n\n'
       'Beyond that, the conclusions here can only graduate from "a lead worth chasing" to '
       '"evidence that stands up" once the whole experiment has been repeated twice more — '
       'right now 13 people share one control model, so there is only 1 independent sample.\n\n'
       '**These numbers no longer need the qualifier "maybe the attack was just weak" — '
       'Step 1 has already checked that.** With "distance features x classifier" laid out as '
       'a grid, **stronger adversaries read more, not less**, so these readings are not an '
       'artefact of a short ruler.'),

 dict(
  name="Step 3 · Which part of the data the leak actually hides in",
  status="done", status_note=" · all three sub-questions finished",
  purpose='Move from "there is a risk" to "the risk is predictable and locatable". This is '
          'a precondition for designing a defence: **if a different person leaks through a '
          'different part of the data every time, a defence has nothing to grip.** '
          'Three sub-questions: is it the same people leaking every time, what amplifies '
          'the leak, and what information is actually leaking.',
  plan=['**Who**: rank "who is easiest to identify" separately under two conditions that '
        'were trained independently, see different channels of data, and use different '
        'random seeds — then check whether the two rankings agree',
        '**What amplifies it**: separate the three factors of training length, window '
        'length and channel count. Training length especially — if "just train less" '
        'lowers the risk, that is the cheapest defence there is and our method has to beat it',
        '**What is leaking**: **recompute** the attack several times, each time destroying '
        'one kind of information first (time ordering / absolute level / fine detail / '
        'variability) and seeing which destruction breaks the attack. Whichever one breaks '
        'it is the one carrying the leak',
        '**Every analysis has to plot the normals alongside.** Any distance '
        'measure rises wherever glucose swings a lot, whether or not anyone is being '
        'identified. **Only where the outliers exceed the normals is there a '
        'leak**'],
  charts=[
    dict(type="tallyRows", h=340,
         title="Change the window length and the same people leak; change the channels and they are different people",
         sub="how well the two \"who is easiest to identify\" rankings agree between a pair of conditions · 1 = identical, 0 = unrelated · 26 shared test subjects · tested by shuffling ten thousand times",
         src="computed from results/matrix/subject_auc/*/per_subject.csv · chart type L15 standalone scale",
         data=[["1 day vs 7 days (both with insulin)", 0.651, "probability of pure coincidence 0.0003", True],
               ["1 day vs 7 days (both CGM only)", 0.611, "probability of pure coincidence 0.0012"],
               ["CGM only vs CGM + insulin (both 7 days)", 0.416, "0.038"],
               ["1 day CGM only vs 7 days CGM + insulin", 0.326, "0.109 — cannot rule out coincidence"],
               ["CGM only vs CGM + insulin (both 1 day)", 0.227, "0.270 — cannot rule out coincidence"],
               ["1 day CGM + insulin vs 7 days CGM only", 0.084, "0.685 — essentially unrelated"]],
         opt=dict(fmt="f3", labelW=300, chance=0.0, chanceLabel="unrelated 0",
                  foot="SAME CHANNELS = REPRODUCIBLE · DIFFERENT CHANNELS = NOT")),
    dict(type="hairlineLine", h=380,
         title="Outliers: training longer barely changes how confidently they are identified",
         sub="median identification confidence of the 13 outliers per condition · 0.5 = not identifiable, same as a coin flip · higher is more dangerous",
         src="results/matrix/sweep/subject_auc/*_ms*/per_subject.csv · chart type F2 hairline",
         data=[{"name": "7 days · CGM only", "v": [0.568, 0.688, 0.730, 0.691, 0.704, 0.673]},
               {"name": "1 day · CGM only", "v": [0.562, 0.625, 0.625, 0.724, 0.688, 0.679]},
               {"name": "1 day · CGM + insulin", "v": [0.562, 0.600, 0.562, 0.566, 0.562, 0.612]}],
         opt=dict(fmt="f3", hero=0,
                  x=["20k", "30k", "40k", "60k", "80k", "100k"],
                  xlab="training steps", ylab="identification confidence (median)",
                  foot="OUTLIERS · FLAT AFTER 30K · SEVEN-DAY + INSULIN STILL RUNNING")),
    dict(type="hairlineLine", h=380,
         title="Normals: the longer the training, the more dangerous — by 80k steps they have caught up with the outlier group",
         sub="median identification confidence of the 13 normals per condition · again 0.5 = a coin flip · same data and same definition as the chart above, so the two are directly comparable",
         src="results/matrix/sweep/subject_auc/*_ms*/per_subject.csv · chart type F2 hairline",
         data=[{"name": "1 day · CGM only", "v": [0.554, 0.517, 0.520, 0.520, 0.571, 0.633]},
               {"name": "7 days · CGM only", "v": [0.562, 0.520, 0.523, 0.590, 0.642, 0.634]},
               {"name": "1 day · CGM + insulin", "v": [0.512, 0.520, 0.541, 0.529, 0.535, 0.537]}],
         opt=dict(fmt="f3", hero=0,
                  x=["20k", "30k", "40k", "60k", "80k", "100k"],
                  xlab="training steps", ylab="identification confidence (median)",
                  foot="ORDINARY PATIENTS · RISING · THIS IS WHAT KILLS THE CONTRAST")),
    dict(type="tallyRows", h=420,
         title="The leak is carried jointly by absolute level and time ordering — only destroying both reaches the floor",
         sub="median of how identifiable each person is on their own · 0.5 = not identifiable · 1 day, CGM only · the bottom row is the floor control: shuffle everything inside the window so no identity structure survives, and it should read 0.5",
         src="results/matrix/leak_locus/d1_c1.json · 26 subjects · chart type L15 standalone scale",
         data=[["destroy nothing", 0.653, "the other three conditions: 0.575 / 0.691 / 0.542"],
               ["destroy detail finer than one hour", 0.692, "goes up, not down — detail carries nothing"],
               ["destroy level and variability", 0.560, ""],
               ["destroy absolute level", 0.549, "destroying one alone only removes half"],
               ["destroy time ordering", 0.536, "destroying one alone only removes half"],
               ["destroy level + variability + ordering", 0.520, ""],
               ["destroy absolute level + time ordering", 0.487, "reaches the floor — these two carry the leak", True],
               ["floor control: shuffle inside the window", 0.503, "no identity structure survives, so this is where 0.5 actually sits"]],
         opt=dict(fmt="f3", labelW=300, chance=0.503, chanceLabel="floor 0.503",
                  foot="LEVEL + ORDERING TOGETHER REACH THE FLOOR")),
    dict(type="hairlineLine", h=360,
         title="The leak concentrates in the morning, but that is a symptom, not a cause",
         sub="vertical axis = **how much leakage this one hour contributes**, in the attack's own distance units · computed in three steps: "
             "(1) a person's total leakage is the difference between their distance to the model that never saw them and to the model that did — the more positive, the more they look memorised; "
             "(2) that distance is additive along the time axis, so with the nearest neighbour the attack already picked held fixed it splits into one value per 5-minute slot, and those values sum exactly to the total; "
             "(3) average the 12 slots that share a clock hour, then average over the days in the window, giving 24 points, then average within each group · "
             "condition is 1 day, CGM only",
         src="results/matrix/localise/d1_c1/per_timestep.json split per slot, folded_*.npy folded by clock hour · chart type F2 hairline",
         data=[{"name": "outliers", "v": [0.0103, 0.0041, 0.0035, 0.0033, 0.0066, 0.0059,
                                    0.0073, 0.0125, 0.0148, 0.0156, 0.0140, 0.0092,
                                    0.0102, 0.0141, 0.0113, 0.0158, 0.0116, 0.0096,
                                    0.0059, 0.0069, 0.0131, 0.0044, 0.0051, 0.0039]},
               {"name": "normals", "v": [0.0038, 0.0001, 0.0011, 0.0019, 0.0037, 0.0028,
                                     0.0017, 0.0025, 0.0031, 0.0010, 0.0011, 0.0012,
                                     0.0035, 0.0037, 0.0032, 0.0047, 0.0044, 0.0021,
                                     0.0020, 0.0040, 0.0028, 0.0022, 0.0030, 0.0050]}],
         opt=dict(fmt="f4", hero=0,
                  x=["00:00", "", "", "03:00", "", "", "06:00", "", "", "09:00", "", "",
                     "12:00", "", "", "15:00", "", "", "18:00", "", "", "21:00", "", ""],
                  xlab="hour of the day", ylab="leakage contributed by this hour",
                  foot="RATIO PEAKS AT 09:00 AT 16x · READ THE RATIO, NOT THE PEAK")),
  ],
  conclusions=[
    '**Change the window length and the same people leak; change how many channels the '
    'model sees and they are different people.** Pairing the four conditions two at a '
    'time, the agreement between the two "who is easiest to identify" rankings is '
    '**0.651 and 0.611** for the two pairs that **share channels and differ only in window '
    'length** (probability of pure coincidence 0.0003 and 0.0012); the pairs with '
    '**different channels** reach only 0.416 and 0.227, and the worst pair is 0.084 '
    '(coincidence cannot be ruled out).\n\n'
    'That is still enough to support what this step needs to say — '
    '**leakage is a stable property of the individual, so a defence has something to grip** '
    '— but it sharpens the claim: **what is stable is "who leaks under a fixed set of '
    'channels", not "who leaks" in general**. Adding insulin does not merely quieten the '
    'normals, it **substantially swaps out which people are exposed**. So a '
    'defence tuned for one channel configuration cannot be assumed to work under another.',

    '**Training length controls the breadth of exposure, not its depth.** From 30k steps to '
    '100k, the number of outliers identified **bounces between 8 and 11 with no '
    'trend** (9/13 at 30k, 9/13 at 100k); over the same range normals climb from '
    '**2/13 all the way to 13/13**. The seven-day condition reproduces the same thing '
    '(5/13 → 11/13). '
    '**The gap between the groups is widest at 30k and has disappeared entirely by 100k.** '
    'That explains the puzzle in Step 2: our main experiment trains to three times past '
    'the point where outliers stand out most.',

    '**Between 30k and 100k steps, training longer is worse on every single metric** — more '
    'people at risk, the gap between groups erased, and generation quality degraded as well '
    '(0.061 → 0.083; 0.131 → 0.341 in the seven-day condition). '
    'There is no trade-off in this range, only waste. **So "stop early" is a real and free '
    'defence that halves the number of people at risk (22/26 → 11/26). Our method has to '
    'beat that, not beat the over-trained endpoint.**',

    '**The leak is carried jointly by absolute level and time ordering; destroying one of '
    'them removes only half.** Each person\'s own identifiability starts at 0.653: '
    'destroying absolute level alone takes it to 0.549, destroying time ordering alone to '
    '0.536 — **only destroying both takes it to 0.487, and the floor is 0.503. All four '
    'conditions behave the same way.** So a defence has to perturb **both** of these; '
    'moving one is not enough.\n\n'
    '⚠️ This used to read "the leak hides in the value distribution", based on the '
    'observation that **how separable the two groups are** rose to 0.905 after sorting. '
    '**That quantity measures how different outliers are from normals, not '
    'whether any one person can be identified** — a different question. And looked at arm '
    'by arm, sorting actually **shrinks** the outlier arm\'s own gap; all it removed was '
    'the layer both arms share, namely "what a day of glucose roughly looks like". '
    'Corrected.',

    '**Detail finer than one hour carries no leakage at all** — destroy it and '
    'identifiability goes up rather than down (0.653 → 0.692). So a defence never needs to '
    'touch the high-frequency part; distortion spent there is wasted.',

    '**The hour-by-hour chart looks like "the morning leaks most", but that is a symptom.** '
    'The morning genuinely is when outliers exceed normals by the widest '
    'margin (16x at 09:00, 3.4x averaged over the day), but combined with the previous '
    'point: **what is memorised is the values, and the morning is merely when those values '
    'happen to occur.** So a defence has to act on the distribution of values, not on a '
    'time of day.',
  ],
  next='This step is final and its four charts go straight into the paper. It pins down one '
       'constraint for the defence design (Steps 6 and 7): **the defence must act along the "distribution of values" '
       'axis.** It also leaves one unsolved problem: searching for the closest record by '
       '"does the waveform look alike" and by "does the value distribution look alike" '
       'returns **a different record 99% of the time** — they are in fact two different '
       'attacks, and blocking one does not block the other.'),

 dict(
  name="Step 4 · The privacy-versus-quality curve of DiM-TS",
  status="running", status_note=" · three conditions have all six points · the copy endpoint is done for all four · missing the random-noise endpoint and four points of the fourth curve",
  purpose='**Narrowed scope: we study one competitor only, DiM-TS.** Other baselines come '
          'after our model beats it.\n\n'
          'This step does exactly one thing: **draw the privacy and the generation quality '
          'of DiM-TS as a single curve.** The scale of that curve is pinned down by two '
          'extremes, neither of which is a real generative model:\n\n'
          '· **copy the training data verbatim** — the fake data is identical to the real '
          'data, so quality is perfect, but every patient can be identified and privacy is '
          'at its worst\n'
          '· **generate pure noise** — no relationship to the training data at all, nobody '
          'can be identified, privacy is perfect, and the fake data is useless\n\n'
          'Every real generative model falls between those two points. '
          '**Our model\'s goal is one sentence: better privacy at the same quality, or '
          'better quality at the same privacy — that is, its curve has to run outside the '
          'DiM-TS curve, up and to the right of it.**',
  plan=['Train DiM-TS to 100k steps, saving a checkpoint every 10k, then resample from each '
        'checkpoint to get the six points on the curve. **No retraining**, so the whole '
        'curve costs only sampling and scoring',
        '**Every point uses the same sampling budget (500 steps)** — otherwise the points '
        'are not comparable to each other',
        'The two endpoints are measured separately: copy_paste for the copying one, '
        'Gaussian noise at the same scale as the data for the random one. '
        '**The random one doubles as a negative control** — its 27 fake datasets are '
        'independent of one another, so the attack must read about 0.5 on it. '
        'If it does not, the attack itself is broken',
        'Each point yields two numbers: the horizontal axis is how realistic the fake data '
        'is, the vertical axis is the identification risk (median over all 26 people)'],
  charts=[
    dict(type="plumbScatter", h=400,
         title="1 day · CGM only",
         sub="horizontal = generation quality (1 divided by FID, further right is more realistic) · vertical = privacy score (1 = completely unidentifiable) · higher is better on both axes, ideal is top right · star = the cheating control that echoes the training data back, which is what stretches this side of the boundary · large circle = on the frontier, small circle = dominated",
         src="checkpoints from results/matrix/sweep/ plus the copy endpoint scored the same way · chart type F8 scatter",
         data=[
               ["20k", 16.6, 0.884, "quality 0.0602 · risk 0.558", 1, 0],
               ["30k", 16.37, 0.917, "quality 0.0611 · risk 0.542", 1, 0],
               ["40k", 15.34, 0.921, "quality 0.0652 · risk 0.539 · lowest risk of all points", 1, 0],
               ["60k", 13.84, 0.899, "quality 0.0723 · risk 0.551", 0, 0],
               ["80k", 13.59, 0.815, "quality 0.0736 · risk 0.593", 0, 0],
               ["100k", 12.1, 0.727, "quality 0.0826 · risk 0.636 · the point used in the main experiment", 0, 0],
               ["verbatim copy", 29.84, 0.34, "quality 0.0335 · risk 0.830", 0, 0, "star"],
              ],
         opt=dict(fmt="f2", xlab="generation quality  1 / FID", ylab="privacy score  2x(1-AUC)",
                  hlines=[[1.0, "1.0 completely unidentifiable"]],
                  series=[["1 day · CGM only", "#C0BFB7"]],
                  foot="D1-C1 · HIGHER IS BETTER ON BOTH AXES")),
    dict(type="plumbScatter", h=400,
         title="1 day · CGM + insulin",
         sub="horizontal = generation quality (1 divided by FID, further right is more realistic) · vertical = privacy score (1 = completely unidentifiable) · higher is better on both axes, ideal is top right · star = the cheating control that echoes the training data back, which is what stretches this side of the boundary · large circle = on the frontier, small circle = dominated",
         src="checkpoints from results/matrix/sweep/ plus the copy endpoint scored the same way · chart type F8 scatter",
         data=[
               ["20k", 8.46, 0.949, "quality 0.1182 · risk 0.526", 0, 0],
               ["30k", 9.19, 0.954, "quality 0.1089 · risk 0.523 · furthest to the top right", 1, 0],
               ["40k", 10.08, 0.88, "quality 0.0992 · risk 0.560", 1, 0],
               ["60k", 8.87, 0.911, "quality 0.1128 · risk 0.544", 0, 0],
               ["80k", 8.12, 0.88, "quality 0.1232 · risk 0.560", 0, 0],
               ["100k", 7.4, 0.897, "quality 0.1352 · risk 0.552", 0, 0],
               ["verbatim copy", 23.55, 0.321, "quality 0.0425 · risk 0.839", 0, 0, "star"],
              ],
         opt=dict(fmt="f2", xlab="generation quality  1 / FID", ylab="privacy score  2x(1-AUC)",
                  hlines=[[1.0, "1.0 completely unidentifiable"]],
                  series=[["1 day · CGM + insulin", "#8F8E86"]],
                  foot="D1-C2 · HIGHER IS BETTER ON BOTH AXES")),
    dict(type="plumbScatter", h=400,
         title="7 days · CGM only",
         sub="horizontal = generation quality (1 divided by FID, further right is more realistic) · vertical = privacy score (1 = completely unidentifiable) · higher is better on both axes, ideal is top right · star = the cheating control that echoes the training data back, which is what stretches this side of the boundary · large circle = on the frontier, small circle = dominated",
         src="checkpoints from results/matrix/sweep/ plus the copy endpoint scored the same way · chart type F8 scatter",
         data=[
               ["20k", 7.66, 0.87, "quality 0.1306 · risk 0.565", 1, 0],
               ["30k", 6.03, 0.85, "quality 0.1658 · risk 0.575", 0, 0],
               ["40k", 4.71, 0.829, "quality 0.2123 · risk 0.586", 0, 0],
               ["60k", 3.48, 0.788, "quality 0.2869 · risk 0.606", 0, 0],
               ["80k", 2.93, 0.691, "quality 0.3413 · risk 0.655 · worst on both axes", 0, 0],
               ["100k", 3.06, 0.72, "quality 0.3266 · risk 0.640", 0, 0],
               ["verbatim copy", 13.71, 0.28, "quality 0.0729 · risk 0.860", 0, 0, "star"],
              ],
         opt=dict(fmt="f2", xlab="generation quality  1 / FID", ylab="privacy score  2x(1-AUC)",
                  hlines=[[1.0, "1.0 completely unidentifiable"]],
                  series=[["7 days · CGM only", "#22211F"]],
                  foot="D7-C1 · HIGHER IS BETTER ON BOTH AXES")),
    dict(type="plumbScatter", h=400,
         title="7 days · CGM + insulin",
         sub="horizontal = generation quality (1 divided by FID, further right is more realistic) · vertical = privacy score (1 = completely unidentifiable) · higher is better on both axes, ideal is top right · star = the cheating control that echoes the training data back, which is what stretches this side of the boundary",
         src="checkpoints from results/matrix/sweep/ plus the copy endpoint scored the same way · chart type F8 scatter",
         data=[
               ["30k", 4.69, 0.880, "quality 0.2132 · risk 0.560", 1, 0],
               ["100k", 3.53, 0.916, "quality 0.2836 · risk 0.542 · the other four checkpoints are still running", 1, 0],
               ["verbatim copy", 11.47, 0.316, "quality 0.0872 · risk 0.842", 0, 0, "star"],
              ],
         opt=dict(fmt="f2", xlab="generation quality  1 / FID", ylab="privacy score  2x(1-AUC)",
                  hlines=[[1.0, "1.0 completely unidentifiable"]],
                  series=[["7 days · CGM + insulin", "#F5572F"]],
                  foot="D7-C2 · HIGHER IS BETTER ON BOTH AXES")),
    dict(type="hairlineLine", h=380,
         title="The same fact seen another way: both axes get worse together as training goes on",
         sub="orange = identification risk (on the left-hand scale) · grey = the realism score multiplied by 8 so it fits the same chart · both rising = both aspects degrading at once",
         src="results/matrix/sweep/ · 1 day, CGM only · chart type F2 hairline",
         data=[{"name": "identification risk", "v": [0.558, 0.542, 0.539, 0.551, 0.593, 0.636]},
               {"name": "how unrealistic the fake data is, x8", "v": [0.482, 0.489, 0.522, 0.578, 0.589, 0.661]}],
         opt=dict(fmt="f3", hero=0,
                  x=["20k", "30k", "40k", "60k", "80k", "100k"],
                  xlab="training steps", ylab="risk / unrealism x8",
                  foot="BOTH RISE · NO TRADE-OFF ALONG THIS AXIS, ONLY WASTE")),
  ],

  conclusions=[
    '**How complete these four charts are right now.** Three of the curves have all six '
    'points; the fourth has only two. The copying endpoint has been measured for all four '
    'conditions and is already drawn as a star; '
    '**the random-noise endpoint has not been measured, so for now the charts only have '
    'one side of the boundary.** The four missing checkpoints of the fourth curve '
    '(seven days CGM + insulin) are queued as job 15871795.\n\n'
    '**Both axes were flipped so that higher is better, which puts the two extremes near '
    'the axes and encloses a region together with the curve.** The horizontal axis is 1 '
    'divided by the realism score (the raw score is better when lower, so the reciprocal '
    'is better when higher); the vertical axis is 2x(1 - identification risk) '
    '(risk 0.5 = unidentifiable → privacy score 1.0). '
    '**The ideal model sits in the top right: realistic and unidentifiable.** '
    'The two reference extremes are drawn as **stars** to distinguish them from the '
    'sampled points — they are not real generative models, they only stretch the boundary '
    'of the region.',

    '**Pale arrows connect the points in order of training steps, so you can see at a '
    'glance which way the points move.** In all three conditions **the arrows point down '
    'and to the left** — the longer the training, the worse both quality and privacy get. '
    'If the arrows get in the way there is a switch in the top right of each chart to turn '
    'them off.',

    '**Large circles are on the frontier, small circles are dominated by other points.** '
    'For 1 day with CGM only: **20k, 30k and 40k are on the frontier; 60k, 80k and '
    '100k are all inside**. For 1 day CGM + insulin the frontier is 30k and 40k. For 7 days '
    'with CGM only **only 20k survives on the frontier**. '
    '**In other words, not a single checkpoint past 40k steps contributes anything to the '
    'boundary.**',

    '**Within one condition the points run roughly from top right to bottom left, with both '
    'coordinates shrinking together.** For 1 day with CGM only, from (15.3, 0.921) '
    'at 40k steps to (12.1, 0.727) at 100k. '
    '**That is not a trade-off, it is a loss on both counts** — the extra 60k steps of '
    'training bought nothing at all.',

    '**The random-noise star will land near the vertical axis** (quality approaching 0, '
    'privacy 1.0), **and the verbatim-copy star lands at the bottom right** (very high '
    'quality, very low privacy). The copying one does not reach y=0; it sits at about 0.37, '
    'because it samples with replacement and roughly a third of the training records have '
    'no copy at all in the released set, so the attack has nothing to grab onto for those. '
    '**That is the genuine lower bound of this control, not a drawing error.**',

    '**What our model has to prove is one sentence: its points have to land above and to '
    'the right of these, so that the region it encloses with the axes is larger than '
    'DiM-TS\'s.** As of now our model is not on this chart yet — not because its quality '
    'falls short (on the horizontal axis it already wins by 2.4x) but because '
    '**the vertical coordinate has not been measured**: the 27 models of the leakage '
    'baseline are still training (see Step 7).',
  ],
  next='**Two things are still missing, and we know how to get both:**\n\n'
       'One, **the random-noise endpoint** (Gaussian noise at the same scale as the data). '
       'It is not only a reference point for the chart — '
       '**it is simultaneously the negative control for this whole attack.** Its 27 fake '
       'datasets are independent of one another and carry nobody\'s trace, so the attack '
       'must read about 0.5 on it. **If it does not, the attack itself is broken and every '
       'number above has to be re-examined.** No training needed, only sampling and scoring.\n\n'
       'Two, **the four missing checkpoints of the fourth curve** (20k / 40k / 60k / 80k '
       'steps), queued on glong as job 15871795. This one had its walltime recomputed from '
       'measured throughput: about 12 hours per checkpoint, four of them, about 48 hours.\n\n'
       '**The fourth condition separates the two groups more widely than any of the others '
       '(0.935), so its full curve is the most interesting one.**'),

 dict(
  name="Step 5 · How long to train: setting the budget",
  status="running", status_note=" · protocol fixed · budget for 1 day CGM-only pinned at 50k steps · two rungs still missing in the other three conditions",
  purpose='The curve in Step 4 shows that training longer degrades both aspects at once, so '
          'the question has to be answered: **how long should we actually train?**\n\n'
          'This step fixes a protocol, and what it fixes is not just money — '
          '**it is a precondition for the attack being valid.** Step 3 showed that training '
          'length strongly affects identification confidence, so if the "used them" model '
          'and the "did not use them" model are trained for different lengths, the '
          'difference between them is contaminated by the effect of training length — '
          'which is exactly the confound the whole experiment is built to exclude.',
  plan=['**Train only the control model to the full budget**, saving a checkpoint every 10k steps',
        'Resample from each checkpoint and read off the realism score. '
        '**Whichever rung scores lowest is that condition\'s training budget**',
        '**All 27 models of that condition then use that same number of steps**',
        '**Decide it on the control model, not on any other model** — the control model '
        'never touches the 26 test subjects, so choosing the budget cannot touch them either',
        '**Do not use training loss as the early-stopping signal** (see the chart and the '
        'conclusions below)'],
  charts=[
    dict(type="hairlineLine", h=380,
         title="Training loss keeps falling while the fake data keeps getting less realistic — the two lines run in opposite directions",
         sub="same condition (7 days, CGM only), same training run · orange = how unrealistic the fake data is (higher is worse) · grey = training loss multiplied by 20 so it fits the same chart (lower is better) · the two lines point opposite ways",
         src="training log B2_d7_c1_15227995 and results/matrix/sweep/quality_tsgem/ · chart type F2 hairline",
         data=[{"name": "how unrealistic the fake data is", "v": [0.1306, 0.1658, 0.2123, 0.2869, 0.3413, 0.3266]},
               {"name": "training loss x 20", "v": [0.2498, 0.2452, 0.2334, 0.2134, 0.2018, 0.1924]}],
         opt=dict(fmt="f4", hero=0,
                  x=["20k", "30k", "40k", "60k", "80k", "100k"],
                  xlab="training steps", ylab="unrealism / loss x20",
                  foot="LOSS KEEPS FALLING · QUALITY KEEPS GETTING WORSE")),
    dict(type="chunkyBars", h=360,
         title="10k steps really is undertrained — twenty-three times worse, so that rung is out",
         sub="1 day with CGM only, how realistic the fake data is, lower is more realistic · control model only · orange is the newly added rung",
         src="results/matrix/sweep/quality_tsgem/d1_c1_ms*_base.json · chart type G3 chunky bars",
         data=[["10k steps", 1.3808, "23 times worse than 20k — nowhere near converged"],
               ["20k steps", 0.0602, ""],
               ["30k steps", 0.0611, ""],
               ["40k steps", 0.0652, ""],
               ["50k steps", 0.0590, "the lowest of all rungs"],
               ["60k steps", 0.0723, ""],
               ["80k steps", 0.0736, ""],
               ["100k steps", 0.0826, ""]],
         opt=dict(fmt="f4", hero=0,
                  foot="10K IS 23x WORSE · THE MINIMUM IS INTERIOR AFTER ALL")),
  ],
  conclusions=[
    '**The 10k rung has landed, and it settles a question that was hanging.** '
    'In two of the three conditions the quality minimum sat on the earliest rung measured '
    '(20k steps), so the true minimum might have been earlier and invisible to us. '
    'Now it has been measured: **10k steps scores 1.3808, twenty-three times worse than '
    '20k** (1.8933 in the other condition, sixteen times worse). '
    '**The curve degrades sharply below 20k, so the minimum is not further to the left.**',

    '**In passing this corrects an unsupported claim in the project documentation — right '
    'conclusion, wrong reason.** The documentation said "the 20k row has not converged and '
    'is excluded from the trend", and at the time no result file in the repository '
    'supported that. Now there is one: **the unconverged rung is 10k, not 20k**. '
    '20k is perfectly usable and is in fact the second-best rung for 1 day with '
    'CGM only.',

    '**Training loss cannot be used for early stopping — it would never fire.** The loss '
    'falls all the way to 100k steps: 0.01249 at 20k, 0.00962 at 100k, '
    '**a 23% drop that is still falling at the last rung**. Over that same range the '
    'unrealism of the fake data climbs from 0.131 to 0.327, **a factor of 2.5**. '
    '**The loss says keep training, the quality says you should have stopped long ago. '
    'Early stopping on loss would train straight through to the worst point.**',

    '**The rule "stop when the loss has not improved for 5 epochs" has a subtler problem '
    'too: noise would trigger it.** One epoch here is about 90 steps, so 5 epochs is about '
    '450 steps, while the per-step loss bounces between 0.007 and 0.021 — at that scale, '
    'judging "did the minimum improve" is essentially measuring noise.',

    '**So early stopping is decided on generation quality, on the control model, and once '
    'decided all 27 models of that condition use the same number of steps.** A fixed budget '
    'is not laziness, **it is a precondition for the attack being valid** — if two models '
    'are trained for different lengths, the difference between them is contaminated by '
    'training length, which is the confound the design exists to exclude.',

    '**The cost is very favourable.** Resampling and scoring one model takes about 20 '
    'minutes, so bracketing the minimum for one condition costs a handful of GPU-hours, '
    'while the 27 models of that condition cost 125 to 700 GPU-hours. '
    '**And using 30k steps instead of 100k makes the training itself 3.3 times cheaper.**',

    '**The budget for 1 day with CGM only is now pinned down at 50k steps, and the '
    'minimum really is in the interior.** Both extra rungs have landed: 10k scores '
    '**1.3808** (dead) and 50k scores **0.0590** (lowest of all eight rungs). '
    '**The worry that the true minimum might lie left of 20k where we could not see it has '
    'been ruled out** — to the left is a cliff, to the right is a slow decline. '
    'So this condition is fixed at 50k steps.',

    '**The other three conditions are bracketed to different degrees, and it is worth '
    'spelling out one by one because they differ.** '
    '**1 day CGM + insulin**: the 10k rung was measured too (1.8933, sixteen times worse '
    'than 20k), so the left side is a cliff there as well, and the minimum is interior at '
    '40k steps (0.0992) — **one more rung at 50k and it is pinned down**. '
    '**7 days CGM only**: the minimum sits on the earliest rung measured '
    '(20k, 0.1306), **nothing to the left of it has been measured, so the true minimum may '
    'be earlier and this condition cannot have a budget set yet**. '
    '**7 days CGM + insulin**: only two points exist (30k and 100k), which is not even a '
    'trend yet.\n\n'
    '**The minima genuinely sit in different places in different conditions, so the 50k of '
    '1 day with CGM only cannot be copied across.** The good news is that filling '
    'the gaps is cheap — control model only, one model per rung, a handful of GPU-hours.',
  ],
  next='**Write the settled budget into the experiment config, and add two rungs to each of '
       'the other three conditions.** The cost is tiny: control model only, two models per '
       'condition, a handful of GPU-hours — and what it decides is how the 125 to 700 '
       'GPU-hours of that condition\'s 27 models get spent.\n\n'
       '**This step has already corrected an unsupported claim in the project '
       'documentation** (see conclusion 2): the documentation said "the 20k row has not '
       'converged and is excluded from the trend", with no result file in the repository '
       'supporting it. Now it is measured: **the unconverged rung is 10k, not 20k.** '
       'Right conclusion, wrong reason; fixed.'),

 dict(
  name="Step 6 · Can it be fixed after release",
  status="partial", status_note=" · the one-strength-for-everyone version has been measured (it fails); the per-patient version has not",
  purpose='The fake data has already been generated; **edit it once more** before releasing '
          'it, to wipe out the remaining traces. The appeal of this route is that no '
          'retraining is needed — one dial gives you a new strength. '
          '**This step did not build a mechanism first, it measured the ceiling first:** '
          'apply the theoretically best possible release-time edit directly to the existing '
          'fake data and read the attack\'s output. Any strength that cannot be reached '
          'here cannot be reached by any mechanism.',
  plan=['How the edit works: each piece of fake data is **pushed away** from the direction '
        'of the training records that selected it, by some distance. This is the most '
        'effective version of this whole family of mechanisms — a real mechanism would '
        'additionally have to spend effort keeping the data presentable',
        '**Both sides get edited.** In a real deployment the data holder edits whatever '
        'they release, so editing only one side would be measuring a defence nobody can '
        'actually deploy',
        '**The verdict has to count both directions.** Being judged "they were used" and '
        'being judged "they definitely were not used" leak exactly the same amount of '
        'information — counting only one direction would read "edited too far" as '
        '"successfully protected"',
        'Sweep the strength from 0 up to 20% of the nearest-neighbour distance, reading the '
        'attack output at every rung'],
  charts=[
    dict(type="hairlineLine", h=380,
         title="Editing the data after release does not make people safer, it gets more of them identified",
         sub="how many people can be identified (out of 26) · horizontal axis is the edit strength · both directions counted: judged \"they were used\" and judged \"they definitely were not used\" both count as leakage · lower is safer",
         src="results/matrix/edit_ceiling/*.json · chart type F2 hairline",
         data=[{"name": "1 day, CGM only", "v": [12, 10, 10, 11, 13, 18, 19, 20]},
               {"name": "7 days, CGM only", "v": [18, 18, 17, 18, 19, 16, 17, 15]}],
         opt=dict(fmt="int", hero=0,
                  x=["no edit", "0.0005", "0.001", "0.002", "0.004", "0.008", "0.016", "0.032"],
                  xlab="edit strength (as a fraction of the distance to the closest piece of fake data)", ylab="people identifiable (out of 26)",
                  foot="ORANGE = 1-DAY · GREY = 7-DAY · BOTH GET WORSE")),
    dict(type="dumbbell", h=300,
         title="The person at highest risk is not protected at all — they get easier to identify",
         sub="identification confidence of the highest-risk person · left dot = no edit at all · right dot = the worst case across all edit strengths · lower is safer",
         src="results/matrix/edit_ceiling/*.json · both directions counted · chart type F12 dumbbell",
         data=[["1 day · CGM only", 0.747, 0.760, "at edit strength 0.016"],
               ["1 day · CGM + insulin", 0.880, 0.920, "at edit strength 0.032"],
               ["7 days · CGM only", 1.000, 1.000, "1.000 at all eight strengths — completely ineffective"]],
         opt=dict(fmt="f3", hero=2, leftLab="no edit", rightLab="worst after editing",
                  foot="1.000 MEANS IDENTIFIED EVERY TIME")),
  ],
  conclusions=[
    '**The "one strength for everyone" version does not work, and the way it fails is '
    'interesting: it does not erase the information, it flips its sign.** '
    'The edit pushes fake data away from the real records, and once it is pushed too far a '
    'person\'s records become **further away than those of people who were never used** — '
    'and "unusually far" identifies them just as well as "unusually close". So as the '
    'strength goes up, the number of identifiable people climbs from 12 to 20.',

    '**Even at the best strength it only goes from 12 down to 10, while "just stop early" '
    'on its own goes from 22 down to 11.** In other words this route cannot even beat a '
    'free and obvious control.',

    '**⚠️ I got the floor wrong here and am correcting it.** I previously said "with no '
    'leakage at all you would expect about 1.3 of 26 people", which I derived by assuming a '
    '5% false-positive rate and never measured. **The measured floor is 7 to 12 people** '
    '(that many still cross 0.55 after everything inside the window has been shuffled, '
    'because some people only have 4-5 record segments). So "12 down to 10" was read off a '
    'ruler with the wrong scale, and '
    '**whether this route improves anything at all has to be re-measured using the median**.',

    '**The person at highest risk is not protected at any strength.** In the seven-day '
    'condition their confidence is **1.000 at all eight strengths** — identified every '
    'single time. And those are exactly the people data protection exists to protect. '
    '**So a "global dial" defence is structurally insufficient; the strength has to be set '
    'per person and per record segment.**',

    '**But the per-patient version has never been tested, and that is the mechanism the '
    'design actually proposes.** What I measured was a single strength for everyone; and '
    'the way it failed was precisely that **normals get pushed too far before the '
    'outliers get far enough** — which is the problem per-patient allocation exists to '
    'solve. External review also pointed out that the push direction I implemented reaches '
    'only 72% of the ideal efficiency, **so it is not an upper bound**. The correct '
    'statement is therefore: **the single-global-dial version does not work; the '
    'per-patient version is still open.**',
  ],
  next='**The conclusions of this step have to be narrowed and rewritten before they go into '
       'the paper.** Three fixes: add the "single global strength" qualifier, replace the '
       'miscalculated floor, and admit that the push direction used is not an upper bound.\n\n'
       'It still pins down one constraint for the next step: **the defence must allocate '
       'strength per person**, and by the new result in Step 3 it must perturb absolute '
       'level and time ordering **at the same time**.\n\n'
       'An allocator that automatically finds the highest-risk locations is being '
       're-measured — the first measurement deviated from the training-time usage in four '
       'places, so its conclusions are void and are not shown on this panel.'),

 dict(
  name="Step 7 · Our model: IG-FM already clears the bar, now add the privacy modules",
  status="running",
  status_note=" · changes 1 and 2 implemented and past the quality gate (9/2) · change 3 withdrawn after review · stock leakage baseline 20/27, due this afternoon",
  purpose='**Correction of record (2026-09-01): IG-FM is our own work, not an external '
          'baseline.** It comes from our group\'s ICDE 2027 submission, and MAVEN is ours '
          'as well (KDD 2027). The only genuine external competitor is DiM-TS. I had been '
          'treating IG-FM as a third-party model, which was wrong, and correcting it '
          'changes the picture completely.\n\n'
          '**Because the "quality" half is already won.** Stock IG-FM scores **0.0394** on '
          'glucose data, while the registered benchmark for the external baseline '
          'DiM-TS is 0.0948 — **2.4x better**; and the best of eight classifier restarts '
          'only reaches 48.75% accuracy, **worse than a coin flip**.\n\n'
          '**So exactly one job remains: add the privacy modules without giving that '
          'quality advantage back.**',
  plan=['**Confirm the stock model\'s scores first** (done). This is the floor our changes '
        'have to hold — any privacy module that drags quality below 0.0948 hands back '
        'something we had already won',
        '**Change one: replace the masking scheme in the fill-in task.** Right now it hides '
        'individual cells independently at random, so even at 80% hidden the level can '
        'still be averaged from the remaining points and the ordering is pinned down by the '
        'neighbours — **the model never has to generate either of those two things**, and '
        'those two are exactly what carries the entire leak. Change it to mask by '
        'coordinate: one task hides the level, another hides contiguous blocks of time',
        '**Change two: split the bottleneck into three named sub-spaces** — level, ordering, '
        'detail. The level segment is pooled over time so that structurally it cannot hold '
        'point-by-point traces; the existing decorrelation loss is upgraded to "keep the '
        'three groups from bleeding into each other"',
        '**Change three: allocate protection strength per window.** Its precondition has '
        'been tested, and the answer is **it depends on the window length** (see the '
        'conclusions), so this one has to be narrowed to one-day windows, or use a '
        'different definition of isolation',
        '**Every change is measured on two numbers at once**: how much leakage fell and how '
        'much quality dropped. Reporting only one of them is meaningless'],
  charts=[
    dict(type="chunkyBars", h=380,
         title="Our model has already won the quality half — now add privacy without losing it",
         sub="generation quality, lower is more realistic · same condition, same scoring · orange is our model · grey is the external competitor and one of our own models that is already out",
         src="results/pilot_igfm/ and the registered value in configs/experiment.yaml · chart type G3 chunky bars",
         data=[["IG-FM + privacy" + NL + "(trained 9/2)", 0.0383, "with two modules added, quality did not drop; worst of eight classifier restarts 0.5000, i.e. real and fake are indistinguishable"],
               ["IG-FM stock" + NL + "(ours)", 0.0394, "2.4x better than the external competitor; best of eight restarts 48.75%"],
               ["DiM-TS" + NL + "(external competitor)", 0.0948, "the registered pass mark"],
               ["MAVEN" + NL + "(ours, eliminated)", 0.2256, "failed under all three configurations"],
               ["verbatim copy" + NL + "(cheating control)", 0.0335, "best quality of all, but everyone can be identified"]],
         opt=dict(fmt="f4", hero=0,
                  foot="LOWER IS BETTER · OURS ALREADY WINS ON QUALITY")),
    dict(type="chunkyBars", h=360,
         title="A clinical ruler instead: across 33 metrics our model is indistinguishable from resampled real data",
         sub="**median relative gap** against the real data, lower is more realistic · 33 CGM metrics from the international consensus and the literature (times in range, variability, hypo- and hyperglycaemia risk) · the right-hand bar is the noise floor: resampling the real data against itself already differs by this much",
         src="results/clinical/battery.json · metric definitions from Moscardó et al., Diabetes Technol Ther 2020;22(10):719-726 · chart type G3 chunky bars",
         data=[["DiM-TS" + NL + "(external)", 10.1, "17 of the 33 metrics are off by more than 10%"],
               ["IG-FM" + NL + "(ours)", 0.4, "only 1 exceeds 10%, and it sits exactly on the noise floor"],
               ["verbatim copy" + NL + "(noise floor)", 0.9, "resampling the real data once already differs by this much"]],
         opt=dict(fmt="f1", hero=1,
                  foot="MEDIAN RELATIVE GAP ACROSS 33 CLINICAL CGM METRICS · LOWER IS BETTER")),
    dict(type="table", full=True,
         title="All 33 clinical metrics: the real value, both models' values, and how far each one is off",
         sub="**DR = the paper's discriminant ratio** (how well the metric tells one patient from another; higher is stronger) · Δ is the relative gap against the real value, smaller is more realistic · **the last column is the noise floor**: how much resampling the real data against itself already differs on that metric — **a Δ that does not exceed the floor is not a readable difference** · sorted by DiM-TS's gap · orange rows are the headline clinical metrics · 1 day, CGM only",
         src="results/clinical/battery.json · metric definitions from Moscardó et al., Diabetes Technol Ther 2020;22(10):719-726",
         data=[
               ["%T&lt;54", "2.15", "0.58", "0.32", "45.7%", "0.55", "5.3%", "8.2%", True],
               ["GVP %", "2.20", "42.8", "26.8", "37.4%", "42.4", "0.8%", "0.3%"],
               ["%GRADE hypo", "1.79", "5.53", "3.52", "36.2%", "4.90", "11.4%", "11.0%"],
               ["PGS (approx)", "1.71", "15.7", "10.2", "35.0%", "15.4", "1.9%", "0.5%"],
               ["MAG", "2.98", "51.2", "38.3", "25.3%", "51.2", "0.0%", "0.3%", True],
               ["AARC", "1.95", "0.85", "0.64", "25.3%", "0.85", "0.0%", "0.3%"],
               ["%T&lt;70", "1.51", "3.19", "2.49", "22.1%", "3.17", "0.8%", "4.8%", True],
               ["%T&gt;250", "—", "5.80", "4.57", "21.3%", "5.55", "4.3%", "1.4%"],
               ["Lability Index", "2.11", "1926", "1521", "21.1%", "1925", "0.1%", "0.3%"],
               ["IGC", "1.92", "1.78", "1.46", "17.8%", "1.73", "2.6%", "1.0%"],
               ["LBGI", "1.93", "0.95", "0.81", "15.4%", "0.90", "5.4%", "4.6%"],
               ["GRI", "1.87", "30.7", "26.0", "15.2%", "30.3", "1.3%", "0.4%"],
               ["M-value", "2.00", "14.3", "12.5", "12.4%", "14.1", "1.1%", "1.0%"],
               ["CONGA-1h", "1.73", "41.7", "37.1", "11.1%", "41.8", "0.4%", "0.4%"],
               ["BGRI", "—", "6.14", "5.47", "10.9%", "6.06", "1.2%", "0.3%"],
               ["%T&gt;180", "1.59", "22.8", "20.5", "10.1%", "22.6", "0.9%", "1.1%"],
               ["HBGI", "1.83", "5.18", "4.66", "10.1%", "5.16", "0.4%", "1.2%"],
               ["GRADE", "1.67", "5.39", "4.91", "8.8%", "5.39", "0.1%", "0.5%"],
               ["CONGA-2h", "—", "56.3", "52.2", "7.3%", "56.7", "0.8%", "0.6%"],
               ["J-index", "1.85", "38.5", "36.4", "5.5%", "38.5", "0.0%", "1.1%"],
               ["%T 70-140", "1.58", "51.5", "54.2", "5.2%", "51.4", "0.2%", "0.7%"],
               ["SD", "1.62", "45.9", "43.6", "5.2%", "46.0", "0.2%", "1.4%"],
               ["%T 70-160", "1.63", "64.4", "67.5", "4.8%", "64.4", "0.1%", "0.2%"],
               ["CONGA-4h", "—", "65.3", "62.1", "4.8%", "65.3", "0.1%", "1.2%"],
               ["%T&gt;140", "1.54", "45.3", "43.3", "4.4%", "45.4", "0.3%", "1.2%"],
               ["%T 70-180", "1.63", "74.0", "77.0", "4.1%", "74.3", "0.3%", "0.1%", True],
               ["%T 50-140", "—", "54.3", "56.5", "4.0%", "54.2", "0.1%", "0.9%"],
               ["CV %", "1.33", "31.1", "30.1", "3.4%", "31.2", "0.3%", "1.1%", True],
               ["Median glucose", "—", "138.2", "135.4", "2.0%", "138.6", "0.2%", "0.4%"],
               ["Mean glucose", "—", "145.4", "142.9", "1.7%", "145.8", "0.3%", "0.5%", True],
               ["MAGE", "1.56", "87.2", "88.0", "0.9%", "88.6", "1.6%", "1.0%"],
               ["GMI %", "—", "6.79", "6.73", "0.9%", "6.80", "0.1%", "0.2%", True],
               ["%GRADE hyper", "1.50", "74.7", "74.9", "0.3%", "75.7", "1.4%", "1.8%"],
              ],
         opt=dict(fmt="f1", align="lrrrrrrr",
                  cols=["Metric", "DR", "Real", "DiM-TS", "Δ", "IG-FM", "Δ", "Floor"])),
  ],
  conclusions=[
    '**Two of the privacy modules are now in, and they clear the quality gate — which '
    'retires the biggest worry.** Generation quality **0.0383** (0.0394 before the change, '
    'pass mark 0.0948), and the worst of eight classifier restarts is **0.5000**, i.e. real '
    'and fake are indistinguishable. These modules all add constraints to the model, and the '
    'easiest way for them to fail is to break its ability to generate at all — that did not '
    'happen.\n\n'
    '⚠️ Two caveats: 0.0383 against 0.0394 is a 2.8% difference and I have **no** repeat '
    'measurement of this score on a fixed model, so the claim is "did not drop", not '
    '"improved". And this is one base model only — **whether leakage went down is still '
    'unknown**, which is what the batch below is for.',

    '**When checking whether the module actually does anything, my own test criterion was '
    'wrong, and the measurement corrected it.** I had specified that after the change the '
    '"remove the overall level" task should get harder, and that if it did not, the model '
    'must still be reading the level from somewhere. Measured: only 1.21x harder — which by '
    'that criterion reads as "no effect".\n\n'
    '**But that criterion compares a model that was trained on the task against itself; of '
    'course it finds it easy once it has learned it.** The right control is the **stock '
    'model**: it has never seen this task, and on it the same task is **11.36x** harder. '
    'That is what settles it — **if the level were still readable from the conditioning, the '
    'stock model would read it too and would not be 11x worse.** So the level really is '
    'blocked, and the new model closed the gap by learning to generate it itself.',

    '**Review before launch caught two defects that would have run to completion, produced '
    'numbers, and tested nothing. Both are handled.** First, the "remove the level" task was '
    '**bypassed by the other conditioning input** — the model has two input paths and I had '
    'only treated one; the other carried the level through untouched. This throws no error: '
    'training runs, the loss falls, and reading the result against my own prediction would '
    'have produced the **exactly opposite** conclusion. Fixed, with a new test that **fails '
    'on the pre-fix code**.\n\n'
    'Second, **change 3 deployed a statistic that was not the validated one** (bucket width '
    'of the coarse waveform, nearest-neighbour order, and reference pool all differed; the '
    'reference-pool one cannot be fixed here — training has no subject ids, so it cannot '
    'exclude a subject\'s other windows, which the validation script deliberately does). '
    '**So change 3 was withdrawn from this run**: with three changes at once and one of them '
    'wrong, the quality gate result could not be attributed to anything.',

    '**Switch to a ruler a clinician can read and the conclusion gets harder: across all 33 '
    'metrics, IG-FM is indistinguishable from resampled real data.** Median relative gap '
    '**0.4%**, only 1 of 33 above 10% — and that one sits exactly on the noise floor '
    '(resampling the real data against itself also differs by 11%). DiM-TS is at a median of '
    '**10.1%**, with **17** metrics above 10%.',

    '**And DiM-TS fails in a structured way, not by drifting a little everywhere.** It gets '
    'the headline clinical metrics almost exactly right — mean glucose off by 1.7%, GMI by '
    '0.9%, time in range by 4.1%. The misses are all in **variability and hypoglycaemia**: '
    'glycaemic variability −37%, mean absolute glucose −25%, severe hypoglycaemia −46%. '
    '**The curves it generates are too smooth and too safe.**',

    '**Worse, what it misses most is exactly what best distinguishes one patient from '
    'another.** The paper scores every metric for discriminating power; the top two are mean '
    'absolute glucose (2.98) and glycaemic variability (2.20) — and those are precisely where '
    'DiM-TS is off by 25% and 37%. '
    '**What it drops is the information that tells people apart.**',

    '**The clinical consequence is concrete:** using DiM-TS synthetic data to study '
    'hypoglycaemia would **understate severe hypoglycaemic time by nearly half**. '
    'Context-FID cannot see this — it is a distribution distance with no clinical meaning. '
    '**So this battery does not replace it; it covers the side it cannot read, and the '
    'analysis elsewhere still runs on Context-FID.** '
    '⚠️ Two caveats: IG-FM currently has results for one condition only (1 day, CGM only); '
    'and the paper computes these over 12-day windows while ours are 1 and 7 days, so a '
    'single reading is noisier than the paper\'s — but both sides use identical window '
    'lengths, so the comparison itself is fair.',

    '**IG-FM is our own work, not an external baseline — I had this wrong.** It comes from '
    'our group\'s ICDE 2027 submission, and MAVEN is ours as well (KDD 2027). The only '
    'genuine external competitor is DiM-TS.',

    '**Once corrected, the picture is completely different: the quality half is already '
    'won.** Stock IG-FM scores **0.0394** on glucose data against DiM-TS\'s registered '
    'benchmark of **0.0948** — **2.4x better**. The real-versus-fake classifier result is '
    'even more extreme: **the best of eight restarts only reaches 48.75% accuracy, worse '
    'than a coin flip**, meaning the classifier cannot tell real from fake at all. '
    '**So the paper does not have to fight for the "at comparable quality" premise — we '
    'already have it.**',

    '**That also changes what MAVEN\'s failure means.** It is not "a third-party model that '
    'does not suit glucose data", it is **two generators from the same group behaving '
    'completely differently on the same data**. That is itself a comparison worth writing '
    'up: both are flow matching, so where does the difference come from.',

    '**One job left: add the privacy modules without handing the quality advantage back.** '
    'Any change that drags quality below 0.0948 gives back something already won. So every '
    'change is measured on two numbers at once — how much leakage fell, how much quality '
    'dropped.',

    '**The precondition for the allocator module has been tested, and the answer is "it '
    'depends on the window length":** **supported** on one-day windows, **falsified** on '
    'seven-day ones, and **no verdict** on the two conditions in between (the threshold was '
    'registered in advance, not picked afterwards). '
    '**So this module has to be narrowed to one-day windows, or use a different definition '
    'of isolation.** One usable by-product: computing isolation from the hour-averaged coarse '
    'waveform beats using the raw waveform, in all four conditions.',
  ],
  next='**Only one thing is being waited on: the leakage baseline for stock IG-FM** '
       '(job 15870939, 20/27, due this afternoon).\n\n'
       '**Why it has to come first:** saying "leakage fell by X after the change" requires '
       'knowing what it was before. And leakage needs models in pairs — take one person\'s '
       'records and compare the fake data from the model that used them against the model '
       'that did not — so one condition needs **27 models** (1 control plus 26 that each add '
       'one person). **This batch cannot be filled in afterwards**: the comparison has to be '
       'between two versions of the same design, so the baseline must use the same '
       'definitions as the change.\n\n'
       '**Once the baseline lands, the next step is training the 26 include models with the '
       'modules** — the control model already exists (it is the base that cleared the '
       'quality gate today; it trains on the 475 background subjects and excludes the 26 '
       'test subjects, so the quality gate and the MIA control model are the same training '
       'run, nothing wasted). That is the point at which "are the modules worth it" can '
       'finally be answered: the quality half is settled as of today, the privacy half is '
       'what is missing.\n\n'
       '**Change 3 has to be rebuilt before it can be used.** Its problem is not the '
       'implementation but the precondition: training has no subject ids, so it cannot '
       'exclude a subject\'s other windows, which the validation script deliberately does — '
       'without that exclusion "isolation" degenerates into a proxy for how many record '
       'segments that subject contributed. Doing it properly means threading subject ids '
       'into the training step.'),
 dict(
  name="Step 8 · Auditing the method and the operations themselves",
  status="running", status_note=" · ongoing · four problems found that changed conclusions, plus one wasted allocation of compute",
  purpose='This step produces no new results. It checks **whether the numbers we already '
          'have are trustworthy**, and **whether the way we run experiments is wasting '
          'compute**.\n\n'
          'It is on the panel because this project has already had four occasions where a '
          'conclusion was overturned by its own audit, **and every one of those happened '
          'before anything was written into the paper**. That is itself a methodological '
          'result. It also records operational mistakes — one job was killed on timeout '
          'after 24 wasted hours, because before submitting it I '
          '**did not compute the walltime from measured throughput** and instead picked a '
          'round number off the top of my head.',
  plan=['**Any score that involves a neural network has to be repeated at least 8 times and '
        'reported at its extreme**, never run once. A single run is one draw from a lower '
        'bound, and when separable features exist that draw is bimodal',
        '**Before reporting "N of 26 people are identifiable", measure the floor of that '
        'number** — how many still cross the line once all identity structure has been '
        'destroyed. Without a measured floor, only report statistics that have one',
        '**Every number in the documentation has to trace back to a result file**, verified '
        'automatically by a script',
        '**Any code that will spend compute gets an independent review before it is '
        'submitted.** That process has already caught defects that would have made the '
        'conclusions wrong, twice',
        '**Compute walltime from measured throughput before submitting, and cross-check it '
        'independently** (back out the rate from what the previous run actually achieved). '
        'Picking a round number off the top of my head has already burned 24 hours',
        '**On the login node, do nothing but submit, query and read logs.** Anything that '
        'needs python goes through a zero-GPU job'],
  charts=[
    dict(type="chunkyBars", h=360,
         title="A bigger batch is slower here — the kind of thing only measurement can tell you",
         sub="seven days with two channels, how many windows are denoised at a time · vertical axis is the seconds taken to sample 500 windows, lower is faster · orange is the fastest rung",
         src="measured by scripts/pbs/dev/sbprobe.pbs · chart type G3 chunky bars",
         data=[["800 at a time", 969, "it fits, but the cost per window is highest"],
               ["500 at a time", 945, ""],
               ["300 at a time", 581, "fastest"],
               ["200 at a time", 577, "essentially the same as 300, no reason to go smaller"]],
         opt=dict(fmt="int", hero=2,
                  foot="BIGGER BATCH IS SLOWER HERE · MEMORY-BOUND, NOT COMPUTE-BOUND")),
    dict(type="pairedBars", h=340,
         title="Counting one direction only turns \"more people identified\" into \"more people protected\"",
         sub="same data, same experiment, how many people are identifiable under two ways of counting (out of 26) · pale = only counting \"judged to have been used\" · dark = counting both directions · lower is safer",
         src="results/matrix/edit_ceiling/d1_c1.json · chart type G3 grouped bars",
         data=[["no edit", 11, 12, "the two ways of counting roughly agree"],
               ["strength 0.004", 7, 13, "one direction says it went down, both directions say it went up"],
               ["strength 0.016", 5, 19, "one direction gives the prettiest reading, both directions give the worst"],
               ["strength 0.032", 7, 20, ""]],
         opt=dict(fmt="int", hero=2,
                  foot="FAINT = ONE-SIDED · DARK = TWO-SIDED · SAME DATA")),
    dict(type="pairedBars", h=340,
         title="Same fake data: running the classifier once versus taking the worst of eight changes the answer thirteenfold",
         sub="how well a classifier separates real from fake, where 0 means it cannot at all (best) · pale = a single run · dark = the worst of eight runs · control models of all four conditions",
         src="results/matrix/quality_tsgem/*_st500.json against results/matrix/quality/*/disc_stability.json · chart type G3 grouped bars",
         data=[["1 day" + NL + "CGM only", 0.0238, 0.0308, "1.3x apart"],
               ["1 day" + NL + "CGM + insulin", 0.0263, 0.0525, "2.0x apart"],
               ["7 days" + NL + "CGM only", 0.0062, 0.0792, "12.8x apart — a single run would read as \"essentially indistinguishable\""],
               ["7 days" + NL + "CGM + insulin", 0.0225, 0.0917, "4.1x apart"]],
         opt=dict(fmt="f4", hero=2,
                  foot="FAINT = ONE RUN · DARK = WORST OF EIGHT · SAME FILE")),
  ],
  conclusions=[
    '**The reference used for scoring quality is the model\'s own training data — but I '
    'overstated what follows from that, and have withdrawn it.** I said "the quality axis '
    'and the privacy axis measure the same thing, so a trade-off curve could be manufactured '
    'out of nothing", and that was wrong: the attack reads the tail (how close the nearest '
    'one is) while quality reads the bulk (how similar the overall distribution is). Those '
    'are two different quantities, and that distinction is exactly what our defence design '
    'stands on. The data also shows the two axes separate: the cheating model scores '
    'quality 0.070 / risk 0.816, the external competitor 0.095 / 0.653. '
    '**What remains is a much narrower problem: this ruler is unfair to our own defences**, '
    'because every defence pushes the fake data away, and the score is computed against '
    'exactly what it was pushed away from.',

    '**The floor for "N of 26 people are identifiable" is 7 to 12 people, not the 1.3 I '
    'assumed.** That many still cross 0.55 after everything inside the window has been '
    'shuffled — because some people have only 4 to 5 record segments, so their score can '
    'only take very coarse steps and luck alone carries them over the line. '
    '**The median is reliable** (floor 0.503-0.516) and is what we lead with from now on. '
    'My original 1.3 came from assuming a 5% false-positive rate and was never measured.',

    '**Running the real-versus-fake classifier once is unreliable, and all four conditions '
    'show it.** The worst case is **12.8x apart**: the control model for seven days with '
    'CGM only gives 0.0062 on a single run (which reads as "essentially '
    'indistinguishable") and 0.0792 as the worst of eight. We made the same mistake on our '
    'own model too. **Rule: always take the extreme over multiple restarts, never a single '
    'run.**',

    '**A bigger batch is slower here — the kind of thing only measurement can tell you.** '
    'The chart above is measured on seven days with two channels: 800 at a time takes 969 '
    'seconds, 300 at a time takes only 581. That means this is '
    '**bound by memory bandwidth, not by compute**, and the intuition that "bigger batch = '
    'faster" is backwards here. An earlier record shows that 1000 at a time runs straight '
    'out of graphics memory, so the usable range is much narrower than one would guess.',

    '**Twenty-four hours of compute was wasted because I did not calibrate the walltime '
    'from measured throughput before submitting.** The job for the fourth curve needed 72 '
    'hours at the measured rate of 12 hours per checkpoint across six checkpoints, and I '
    'gave it 24 — it finished two of them and was killed on timeout '
    '(log verbatim: `walltime 86515 exceeded limit 86400`). '
    '**The rule is now: compute the walltime from measured throughput before submitting, '
    'and have an independent cross-check** (back out the rate from what the previous run '
    'actually achieved) rather than relying on a one-way estimate. '
    'In passing this also revealed that one of those checkpoints was redundant — it is the '
    'training endpoint, which the main experiment had already scored — saving 52 GPU-hours.',

    '**Platform compliance audit: all 36 jobs checked line by line against the operations '
    'handbook, and every hard rule passes.** Every job carries the project code; no job '
    'ever names an execution queue (only the normal router is used); nothing trips the '
    '"2 hours and 1 second" trap that drops a job out of the fast lane; nothing requests '
    'the 5-or-more cards that are structurally unschedulable; model weights all live on the '
    'project filesystem rather than the quota-limited home directory. '
    '**One borderline case has been fixed**: one probe job wrote temporary files to a '
    'hard-coded /tmp, while the handbook says explicitly that this machine has no scratch '
    'space and that one should not invent a temporary path — those go on the project '
    'filesystem from now on. '
    '**One genuine violation has been stopped**: before 30 August I ran a great deal of '
    'python on the login node, including scanning 76 MB of training logs and running a '
    'ten-thousand-iteration permutation test — behaviour that gets a support ticket raised '
    'against the project automatically. Everything that needs python now goes through a '
    'zero-GPU fetch job.',
  ],
  next='Wire the "every number traces back to a file" checker into the submission flow, '
       'instead of relying on somebody remembering to run it.\n\n'
       '**Two methodological debts remain:** first, the recomputation that switches the '
       'reference to patients who were never trained on has not been done (a few hours, no '
       'graphics card needed, and it removes the unfairness of that ruler towards our own '
       'defences); second, the allocator re-measurement has had its script fixed per the '
       'review comments and is waiting for a free graphics card.'),

]
