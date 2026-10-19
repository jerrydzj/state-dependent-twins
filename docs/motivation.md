# Motivation and preliminary thoughts: LLM digital twins on state-dependent outcomes

I started with the research behind Simile, mainly Park et al. (2024) and Kolluri et al. (2025), which I've read closely. My main takeaway was that everything happens within the universe of LLM digital twins. These papers mainly show that richer inputs (Park) or fine-tuning (Kolluri) beat demographic prompting. But they compare twin configurations with one another, never against supervised models, so they don't really answer whether twins are worth it at all.

That led me to the Peng, Toubia et al. mega-study (arXiv 2509.19088), which I know from its abstract and a secondhand summary. It does run the supervised comparison. Twins matched an XGBoost model trained on roughly 180 labelled participants, but only on cross-person correlation.

Then I noticed that all three studies, and the literature more broadly, use trait-like outcomes: stable attitudes, personality, and choices in fixed experimental paradigms, where a person's answer depends mainly on who they are. The questions that matter most for simulation are often state-dependent instead: the answers shift with economic conditions, news, and other external circumstances. That's the gap I want to study. My question is under what conditions LLM digital twins outperform supervised baselines on state-dependent survey responses. My hypothesis is that the twins' advantage grows with regime novelty, because an LLM's world knowledge can substitute for training data from the relevant regime.

I want to test this on the University of Michigan Surveys of Consumers microdata, where responses depend heavily on the state of the economy at the time.

My initial idea is to build matched ladders for twins and supervised baselines with the same four information sets:
- demographics only
- demographics plus macro conditions
- demographics plus news context
- demographics plus both

For the supervised models, news has to be turned into features, and any LLM-derived features should be labelled as a hybrid. For re-interviewed panel respondents, we could add their prior response and the conditions at that time as features for both, comparing that subset with and without them.

The main dimension of the study is how out-of-sample the evaluation is. I'm thinking of three tiers: in-time (random holdout of respondents within the same months as training), out-of-time within the same regime (train on earlier months, test on later months before any known shift), and out-of-regime (train before a known shift, test during it). I'd report a headline comparison of all configurations at each tier, for both individual-level and distributional accuracy.

Constraints to work through, as separate concerns:
- pretraining contamination and lookahead bias
- the trade-off between older or dated models, which are cleaner but weaker, and frontier models
- the 2024 phone-to-web mode change
- limited post-cutoff data
