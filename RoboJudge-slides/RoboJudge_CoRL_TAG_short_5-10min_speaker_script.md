# RoboJudge CoRL TAG Workshop Short Talk Script

Target length: 5-10 minutes, written for about 7 minutes

## Slide 1: Title

Hi everyone, I am Siyuan. Today I will present RoboJudge, our work on human-aligned evaluation for embodied video generation. The short version is simple: if video generators are going to become useful world models for robotics, we need judges that understand both physics and task intent.

## Slide 2: Motivation

The core motivation is that embodied videos fail in ways that generic video evaluation often misses. A clip can be visually nice but physically wrong, for example by ignoring contact or changing the robot. Or it can obey physics but perform the wrong task. So we evaluate two axes: Physical Adherence and Instruction Alignment. The goal is not just to rate beauty; the goal is to match what trained humans care about for embodied rollouts.

## Slide 3: System overview

RoboJudge has three pieces. First, RoboJudge-Bench: 800 generated embodied videos with human ratings and rationales. Second, RoboJudge-Train: about twelve thousand videos for training an embodied-specialized judge. Third, RoboJudge-9B, trained from Qwen3.5-9B with supervised learning and reinforcement learning. The key headline is an overall Pearson correlation of 0.719 with human ratings on the benchmark.

## Slide 4: Rubric and examples

The rubric separates PA and IA. Physical Adherence asks whether the agent, scene, and interactions remain physically plausible. Instruction Alignment asks whether the right agent, objects, and goal are actually followed. These examples from the project page show why this matters: PA and IA can move independently, and a single overall impression would hide that.

## Slide 5: Benchmark coverage

RoboJudge-Bench is designed around embodied video rather than generic internet clips. It covers eight source corpora and ten video generators, including different embodiments such as grippers, dexterous hands, and human hands. The point is to expose a judge to the kinds of contact, manipulation, and task-following failures that matter in robot settings.

## Slide 6: Training

The model starts from Qwen3.5-9B. We first use supervised fine-tuning so the model learns the rubric, rationales, and score format. Then we apply GRPO reinforcement learning with a direct score-agreement reward. The training set has over twelve thousand videos and two axis-level labels per video, so the model learns PA and IA explicitly instead of inferring a generic quality score.

## Slide 7: Main result

The main result is that RoboJudge-9B reaches 0.719 overall Pearson correlation with human ratings. In our evaluation, the strongest proprietary baseline is Gemini-3.7-Flash at 0.630, so the absolute gain is 0.089. The split is also useful: PA is 0.645 and IA is 0.775, which tells us the model is learning both physical and instruction-level judgments.

## Slide 8: Ablation and generalization

The ablation shows where the gains come from. The base Qwen3.5-9B model is at 0.401. Supervised training moves it to 0.700, which is the major jump. RL further improves it to 0.719. We also test out-of-domain behavior on VideoPhy2 and Physion-Eval. The gains are smaller but generally positive. A careful wording point: RoboJudge evaluates generated video quality; it is not itself a simulator or policy evaluator.

## Slide 9: Leaderboard use case

Once the judge is aligned to humans, one immediate application is scalable evaluation of video generation models. The leaderboard shows PA, IA, and overall scores. Importantly, the two axes reveal tradeoffs. Some models are physically more stable but less task-aligned, while others follow the task better but have weaker physical dynamics. This gives developers a more actionable signal than a single generic video-quality number.

## Slide 10: Takeaways

To close, the main message is that embodied video generation needs evaluation that understands both physics and task intent. RoboJudge-Bench gives us a testbed for that. RoboJudge-Train lets us train a specialized judge. And RoboJudge-9B reaches the best human agreement among the judges we evaluated. Thanks, and I would be happy to discuss how to connect this more tightly to downstream robot policy learning.
