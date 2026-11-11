# RoboJudge CoRL TAG Workshop Full Talk Script

Target length: 30 minutes, written for about 28-30 minutes

## Slide 1: Title

Hi everyone, I am Siyuan. This talk is about RoboJudge: Multimodal Language Models as Judges for Embodied Video Generation. I will motivate why embodied video generation needs its own evaluation target, introduce RoboJudge-Bench and RoboJudge-Train, and then walk through the results of RoboJudge-9B and its use as a scalable evaluator.

## Slide 2: Roadmap

The talk has four parts. First, the problem: what exactly is missing in existing video evaluation when the video is an embodied rollout. Second, the benchmark and rubric. Third, the training recipe for RoboJudge-9B. Fourth, the results and how the judge can be used to compare video generation models.

## Slide 3: Context

The background is that video generation is moving closer to robotics. We use generated videos as imagined futures, as synthetic data, and sometimes as part of planning or evaluation loops. In that setting, a generated clip is not only a visual artifact. It is supposed to represent a possible physical transition under a task instruction. That is why the evaluation target has to become more embodied.

## Slide 4: Two questions

A single score hides two questions. First: is this physically believable? Does the robot stay consistent, does the scene remain stable, and does contact look realistic? Second: did it follow the task? Is it the right object, the right agent, and the right outcome? RoboJudge separates these as PA and IA because the failures are different and the model developer needs different feedback for each one.

## Slide 5: Existing judges

We evaluate a broad set of judges. Closed-source MLLMs are strong, but their agreement with human PA and IA ratings is still limited. Open-weight MLLMs are easier to reproduce but start lower. Embodied MLLMs have domain knowledge, but that does not automatically make them good judges. And specialized video metrics often target different notions of quality, so they transfer poorly to embodied physical and instructional judgments.

## Slide 6: Contributions

The contributions are fourfold. We introduce a benchmark for human-aligned evaluation of embodied generated videos. We study a broad set of existing model judges. We build RoboJudge-Train, a larger training set with verified labels and rationales. Finally, we train RoboJudge-9B, which achieves the strongest human agreement among the evaluated judges.

## Slide 7: Benchmark contents

RoboJudge-Bench contains 800 generated embodied videos. They come from eight source corpora and ten generators. Every video receives PA and IA annotations, so evaluation can be reported per axis and overall. The distribution is intentionally diverse across sources, embodiments, and simulator or real-source settings.

## Slide 8: Physical Adherence

Physical Adherence has three parts. Agent consistency checks whether the robot or hand remains the same entity through time. Scene consistency checks whether the environment and object states remain coherent. Interaction realism checks contact, grasping, collision, and object motion. These are the kinds of failures that can make a generated rollout useless for robot learning even if the clip looks visually polished.

## Slide 9: Instruction Alignment

Instruction Alignment is different. It checks whether the correct agent is performing the requested task, whether the correct object is manipulated, and whether the intended goal is completed. This is especially important because a physically plausible video can still be irrelevant to the instruction.

## Slide 10: Representative outcomes

These project-page examples show why the two-axis rubric is useful. The left example has strong physical adherence but lower instruction alignment. The middle one follows the instruction more strongly but has weaker physical quality. The right example fails both. These are not a separate split; they are illustrative records from the benchmark.

## Slide 11: Annotation target

The annotation target is not just a number. Annotators score PA and IA separately and provide rationales. This matters for training because the model sees the reasoning behind each score. It also matters for evaluation: we report correlation with human ratings, exact-match accuracy, and MSE, so we can tell whether the judge is aligned in rank, exact score, and calibration.

## Slide 12: Evaluation protocol

In evaluation, every judge is asked the same embodied PA and IA question. Pearson correlation is the main alignment metric because it measures whether a judge ranks videos like humans. Exact accuracy is stricter and asks whether the discrete score matches. MSE penalizes large numerical deviations. We also keep PA and IA separate so the result is diagnostic.

## Slide 13: Judge ranking

This figure summarizes the model-judge ranking. A useful observation is that model family alone is not enough. Strong general MLLMs do relatively well, but they do not solve the benchmark. Embodied-specialized models do not automatically become strong judges. Specialized video judges also do not transfer cleanly to this target. This motivates training directly on the embodied PA and IA rubric.

## Slide 14: Main result

Quantitatively, RoboJudge-9B reaches 0.719 overall Pearson correlation with human ratings. The strongest proprietary baseline in our evaluation is Gemini-3.7-Flash at 0.630, so the gain is 0.089. PA and IA are both strong, with 0.645 on PA and 0.775 on IA. This suggests that training on the embodied rubric gives a specialized judge a clear advantage.

## Slide 15: Training data

RoboJudge-Train has 12,351 videos. Most are generated videos from fourteen generators, which expose many failure modes. We also include 831 verified successful real demonstrations. Those are important because they anchor the high-score end of the rubric. Since every video has both PA and IA labels, the model does not have to compress everything into a generic preference.

## Slide 16: Training recipe

The training recipe is deliberately simple. We start from Qwen3.5-9B. Supervised fine-tuning teaches the model the output format, the PA and IA rubric, and how to produce rationales. Then GRPO reinforcement learning uses exact score agreement as the reward. The goal of RL is to sharpen the discrete scoring behavior after the model already knows the rubric.

## Slide 17: Ablation

The ablation is important because it tells us what is driving the result. The base model starts at 0.401 overall Pearson. SFT is the major change, moving the model to 0.700 by teaching the embodied rubric. RL adds a smaller but meaningful gain to 0.719, mainly by sharpening the discrete score agreement.

## Slide 18: OOD

We also test out of domain. On VideoPhy2, both physical consistency and semantic alignment improve from the base model through SFT and RL. On Physion-Eval the results are more mixed because the task is a different binary physical-reasoning setup. The takeaway is that RoboJudge training helps beyond the exact benchmark, but the target distribution still matters.

## Slide 19: Non-MLLM metrics

We compare against non-MLLM and specialized metrics because one might hope that existing video-quality metrics are enough. The result is that they are not well matched to embodied PA and IA. The failures often require understanding task intent and physical interaction, not just frame quality or embedding similarity.

## Slide 20: Failure cases

RoboJudge is not perfect, and the failure cases are useful. They show where video understanding, temporal grounding, and score calibration are still hard. The practical recommendation is to use RoboJudge as a scalable aligned signal, especially for screening and leaderboard-scale evaluation, while keeping human spot checks for high-stakes claims.

## Slide 21: Leaderboard figure

The first downstream application is a model leaderboard. Using the same PA and IA rubric, RoboJudge-9B scores a set of video generation models. This is useful because model developers can see not only who has the best overall score, but which axis is limiting a model.

## Slide 22: PA/IA tradeoffs

The most useful part of the leaderboard is not just the overall ranking. It is the PA and IA split. For example, a model can have a relatively high physical score but low instruction alignment, which means the clip looks plausible but does not execute the requested task. That distinction gives developers a much more actionable signal.

## Slide 23: Scope

I want to be precise about scope. RoboJudge is a human-aligned video judge for embodied generated rollouts. It is not a direct simulator or policy evaluator. So the right next question is how RoboJudge scores relate to downstream policy learning under matched generated-data or rollout studies. That is a natural follow-up, but we should keep it separate from the core claim of this paper.

## Slide 24: TAG relevance

For the TAG workshop, the broader relevance is that evaluation can become part of the training and selection loop for embodied video generation. If we know whether a model fails PA or IA, we can improve the generator, filter synthetic data, and design better downstream validation. The key is to keep the evidence honest: RoboJudge gives a scalable video-quality signal, and policy claims should be validated with policy metrics.

## Slide 25: Closing

I will close with three messages. First, embodied video generation needs an evaluation target that separates physical adherence and instruction alignment. Second, RoboJudge-Bench gives us a human-annotated way to test model judges on that target. Third, RoboJudge-9B shows that a specialized 9B judge can outperform much larger or proprietary judges on human agreement. Thank you, and I am happy to take questions.

## Slide 26: Backup: policy scope

This is a backup slide for Q&A. The precise answer is that RoboJudge evaluates videos, not policies. It can be used as a quality signal for generated rollouts, but any claim about downstream policy improvement needs a matched policy-learning or policy-evaluation experiment.

## Slide 27: Backup: bias

If asked about bias, I would emphasize that RoboJudge is evaluated against human ratings rather than assumed correct. The two-axis rubric reduces generic aesthetics bias, and rationales make failures easier to inspect. For high-stakes comparisons, it should be used with human spot checks.

## Slide 28: Backup: practical loop

This is the practical loop I would describe if someone asks how to use the model. Generate candidate rollouts, score PA and IA, filter or diagnose failure modes, and then validate with humans and downstream policy metrics when the claim is about robot control.
