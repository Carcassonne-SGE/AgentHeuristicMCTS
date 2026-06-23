package mcts;

import abstractDeterminized.AbstractDeterminizedAgent;
import at.ac.tuwien.ifs.sge.engine.Logger;
import core.AbstractAgentConfiguration;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;
import model.state.State;

import java.util.Random;

/// AgentNaiveHeuristic
///
/// An MCTS agent that integrates a naive heuristic configuration to guide search node selections
/// and simulation/rollout decisions.
public class AgentNaiveHeuristic extends AbstractDeterminizedAgent<DeterminizedActionNodePUCT> {
    final HeuristicConfiguration heuristic;
    final float rolloutGreedyProbability;
    private final float heuristicCoefficient;

    /// AgentNaiveHeuristic
    ///
    /// Default constructor creating an agent with default rollout greedy probability.
    public AgentNaiveHeuristic() {
        this(0.85f);
    }

    /// AgentNaiveHeuristic
    ///
    /// Constructor creating an agent with a logger and default rollout greedy probability.
    ///
    /// @param logger the logger instance
    public AgentNaiveHeuristic(Logger logger) {
        this(logger, 0.85f);
    }

    /// AgentNaiveHeuristic
    ///
    /// Constructor creating an agent with a custom rollout greedy probability.
    ///
    /// @param rolloutGreedyProbability the custom rollout greedy probability
    public AgentNaiveHeuristic(float rolloutGreedyProbability) {
        this(null, rolloutGreedyProbability);
    }

    /// AgentNaiveHeuristic
    ///
    /// Constructor creating an agent with a logger and custom rollout greedy probability.
    ///
    /// @param logger the logger instance
    /// @param rolloutGreedyProbability the custom rollout greedy probability
    public AgentNaiveHeuristic(Logger logger, float rolloutGreedyProbability) {
        this(
                logger,
                new Random(99),
                new AbstractAgentConfiguration(2f, 5, 0.1f, 5),
                HeuristicManager.createDefaultHeuristic(),
                rolloutGreedyProbability);
    }

    /// AgentNaiveHeuristic
    ///
    /// Constructor creating an agent with custom random generator, configurations, and default greedy probability.
    ///
    /// @param logger the logger instance
    /// @param rand the random number generator
    /// @param config the agent configuration
    /// @param heuristic the heuristic configuration
    public AgentNaiveHeuristic(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic) {
        this(logger, rand, config, heuristic, 0.85f);
    }

    /// AgentNaiveHeuristic
    ///
    /// Constructor creating an agent with custom random generator, configurations, and custom greedy probability.
    ///
    /// @param logger the logger instance
    /// @param rand the random number generator
    /// @param config the agent configuration
    /// @param heuristic the heuristic configuration
    /// @param rolloutGreedyProbability the custom rollout greedy probability
    public AgentNaiveHeuristic(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic,
            float rolloutGreedyProbability) {
        this(logger, rand, config, heuristic, rolloutGreedyProbability, 1.0f);
    }

    /// AgentNaiveHeuristic
    ///
    /// Constructor creating an agent with custom random generator, configurations, greedy probability, and heuristic coefficient.
    ///
    /// @param logger the logger instance
    /// @param rand the random number generator
    /// @param config the agent configuration
    /// @param heuristic the heuristic configuration
    /// @param rolloutGreedyProbability the custom rollout greedy probability
    /// @param heuristicCoefficient the coefficient scaling heuristic priors during selection
    public AgentNaiveHeuristic(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic,
            float rolloutGreedyProbability,
            float heuristicCoefficient) {
        super(logger, config, rand);
        this.heuristic = heuristic;
        this.rolloutGreedyProbability = rolloutGreedyProbability;
        this.heuristicCoefficient = 2.5f;// 242
    }

    /// getHeuristicCoefficient
    ///
    /// Returns the heuristic coefficient used to scale action selection scores.
    ///
    /// @return the heuristic coefficient
    public float getHeuristicCoefficient() {
        return heuristicCoefficient;
    }

    /// rolloutCount
    ///
    /// Returns the number of rollouts to perform per simulation.
    ///
    /// @return the rollout count
    int rolloutCount() {
        return config.rollouts();
    }

    /// rootFactory
    ///
    /// Factory method to construct the root node of the MCTS tree.
    ///
    /// @param initialState the initial game state
    /// @return the constructed root node
    @Override
    protected DeterminizedActionNodePUCT rootFactory(State initialState) {
        return new DeterminizedActionNodePUCT(this, null, 0, initialState, 0f, config.c(), heuristic);
    }

    /// childFactory
    ///
    /// Factory method to construct a new child node in the MCTS tree.
    ///
    /// @param parent the parent node
    /// @param action the action taken
    /// @param checkpoint the checkpointed state of the child node
    /// @return the constructed child node
    @Override
    public DeterminizedActionNodePUCT childFactory(DeterminizedActionNodePUCT parent, int action, State checkpoint) {
        return new DeterminizedActionNodePUCT(this, parent, action, checkpoint, 0f, config.c(), heuristic);
    }
}
