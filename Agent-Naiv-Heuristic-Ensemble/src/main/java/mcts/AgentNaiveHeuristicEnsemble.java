package mcts;

import abstractDeterminized.AbstractDeterminizedEnsembleAgent;
import at.ac.tuwien.ifs.sge.engine.Logger;
import core.AbstractAgentConfiguration;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;

import java.util.Random;

/// AgentNaiveHeuristicEnsemble
///
/// An ensemble agent that runs multiple AgentNaiveHeuristic sub-agents
/// searches and aggregates their search statistics and picks one action. Should try to model
/// multiple possible futures
public class AgentNaiveHeuristicEnsemble extends AbstractDeterminizedEnsembleAgent {

    /// AgentNaiveHeuristicEnsemble
    ///
    /// Default constructor creating an ensemble agent with default rollout greedy
    /// probability.
    public AgentNaiveHeuristicEnsemble() {
        this(null);
    }

    /// AgentNaiveHeuristicEnsemble
    ///
    /// @param logger the logger instance
    public AgentNaiveHeuristicEnsemble(Logger logger) {
        this(logger, 0.4f);
    }

    /// AgentNaiveHeuristicEnsemble
    ///
    /// @param rolloutGreedyProbability the custom rollout greedy probability
    public AgentNaiveHeuristicEnsemble(float rolloutGreedyProbability) {
        this(null, rolloutGreedyProbability);
    }

    /// AgentNaiveHeuristicEnsemble
    ///
    /// @param logger the logger instance
    /// @param rolloutGreedyProbability the custom rollout greedy probability
    public AgentNaiveHeuristicEnsemble(Logger logger, float rolloutGreedyProbability) {
        this(
                logger,
                new Random(99),
                // 330 4.0
                new AbstractAgentConfiguration(10f, 10, 0.1f, 1),
                HeuristicManager.createDefaultHeuristic(),
                rolloutGreedyProbability);
    }

    /// AgentNaiveHeuristicEnsemble
    ///
    /// @param logger the logger instance
    /// @param rand the random number generator
    /// @param config the agent configuration
    /// @param heuristic the heuristic configuration
    public AgentNaiveHeuristicEnsemble(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic) {
        this(logger, rand, config, heuristic, 0.85f);
    }

    /// AgentNaiveHeuristicEnsemble
    ///
    /// @param logger the logger instance
    /// @param rand the random number generator
    /// @param config the agent configuration
    /// @param heuristic the heuristic configuration
    /// @param rolloutGreedyProbability the custom rollout greedy probability
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
