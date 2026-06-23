### Heuristic Agents

This repository implements agents for the Caracssonne Environment which are using a heuristic to guide the search. They all use the default heuristic. The values for the default heuristic were obtained by running the optimizer from the VerificationHelper

#### Agent Greedy Heuristic

This Agent does not perform some sort of search but rather picks the action solely based on the heuristic. The agent calculates a distribution by applying a softmax function based on the heuristic values of the available actions, then samples from this distribution to pick the final action. For the sampealing a very low Temperature is used. The experiments have shown that sampeling improves the overall score immensely compared to picking using argmax


#### Agent-Naive-Heuristic

Assumes the other agent performs as good as itself and searches normally though the game tree. Uses the heuristic for guided search. Uses the PUCT formula to incorporate the heuristic as a prior. Apart from that the heuristic is used in the rollout to guide the simulations. During the rollout the action are chosen similarly to the AgentGreedyHeuristicAgent but additionally with a epsilon greedy policy

#### Agent-Naive-Heuristic-Ensemble

Uses the Agent-Naive-Heuristic as a subagent to view multiple possible tile orders.
