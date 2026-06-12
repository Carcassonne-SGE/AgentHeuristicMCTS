### Heuristic Agents

This repository implements agents for the Caracssonne Environment which are using a heuristic to guide the search. They all use the default heuristic. The values for the default heuristic were obtained by running the optimizer from the VerificationHelper

#### Agent Greedy Heuristic

Agent that does no kind of search for the action selection but rather uses the heuristic completely. View the heuristic as a distribution of which action is the best one. Puts the values into a softmax function and uses a very low temperature to sample from that

The sampling part did show in experiments to double the points

#### Agent-Naiv-Heuristic

Assumes the other agent performs as good as itself and searches normally though the game tree. Uses the heuristic for guided search. Takes inspiration form alpha go and uses the PUCT formula to incorporate the heuristic as a prior. Apart from that the heuristic is used in the rollout to guide the simulations. During the rollout the action are chosen similarly to the AgentGreedyHeuristicAgent

#### Agent-Naiv-Heuristic-Ensemble

Uses the Agent-Naiv-Heuristic as a subagent to view multiple possible tile orders.
