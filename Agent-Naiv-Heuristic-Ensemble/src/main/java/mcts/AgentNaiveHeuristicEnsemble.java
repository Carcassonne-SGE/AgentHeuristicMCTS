package mcts;

import abstractDeterminized.AbstractDeterminizedEnsembleAgent;
import at.ac.tuwien.ifs.sge.engine.Logger;
import core.AbstractAgentConfiguration;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;

import java.util.Random;

/**
 * AgentNaiveHeuristicEnsemble implements an ensemble of MCTS agents that
 * utilize
 * standard naive heuristic assumptions and configurable PUCT rollout
 * strategies.
 */
public class AgentNaiveHeuristicEnsemble extends AbstractDeterminizedEnsembleAgent {

    public AgentNaiveHeuristicEnsemble() {
        this(null);
    }

    public AgentNaiveHeuristicEnsemble(Logger logger) {
        this(logger, 0.4f);
    }

    public AgentNaiveHeuristicEnsemble(float rolloutGreedyProbability) {
        this(null, rolloutGreedyProbability);
    }

    public AgentNaiveHeuristicEnsemble(Logger logger, float rolloutGreedyProbability) {
        this(
                logger,
                new Random(99),
                // 330 4.0
                new AbstractAgentConfiguration(10f, 10, 0.1f, 1),
                HeuristicManager.createDefaultHeuristic(),
                rolloutGreedyProbability);
    }

    public AgentNaiveHeuristicEnsemble(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic) {
        this(logger, rand, config, heuristic, 0.85f);
    }

    public AgentNaiveHeuristicEnsemble(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic,
            float rolloutGreedyProbability) {
        super(
                logger,
                config,
                10,
                AbstractDeterminizedEnsembleAgent.pooledByVisitsAggregationMode(),
                rand,
                (subLogger, subConfig, subRand) -> new AgentNaiveHeuristic(subLogger, subRand, subConfig, heuristic,
                        rolloutGreedyProbability));
    }
}
