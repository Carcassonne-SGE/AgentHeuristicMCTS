package mcts;

import abstractDeterminized.AbstractDeterminizedAgent;
import at.ac.tuwien.ifs.sge.engine.Logger;
import core.AbstractAgentConfiguration;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;
import model.state.State;

import java.util.Random;

public class AgentNaiveHeuristic extends AbstractDeterminizedAgent<DeterminizedActionNodePUCT> {
    final HeuristicConfiguration heuristic;
    final float rolloutGreedyProbability;
    private final float heuristicCoefficient;

    public AgentNaiveHeuristic() {
        this(0.85f);
    }

    public AgentNaiveHeuristic(Logger logger) {
        this(logger, 0.85f);
    }

    public AgentNaiveHeuristic(float rolloutGreedyProbability) {
        this(null, rolloutGreedyProbability);
    }

    public AgentNaiveHeuristic(Logger logger, float rolloutGreedyProbability) {
        this(
                logger,
                new Random(99),
                new AbstractAgentConfiguration(2f, 5, 0.1f, 5),
                HeuristicManager.createDefaultHeuristic(),
                rolloutGreedyProbability);
    }

    public AgentNaiveHeuristic(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic) {
        this(logger, rand, config, heuristic, 0.85f);
    }

    public AgentNaiveHeuristic(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic,
            float rolloutGreedyProbability) {
        this(logger, rand, config, heuristic, rolloutGreedyProbability, 1.0f);
    }

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

    public float getHeuristicCoefficient() {
        return heuristicCoefficient;
    }

    int rolloutCount() {
        return config.rollouts();
    }

    @Override
    protected DeterminizedActionNodePUCT rootFactory(State initialState) {
        return new DeterminizedActionNodePUCT(this, null, 0, initialState, 0f, config.c(), heuristic);
    }

    @Override
    public DeterminizedActionNodePUCT childFactory(DeterminizedActionNodePUCT parent, int action, State checkpoint) {
        return new DeterminizedActionNodePUCT(this, parent, action, checkpoint, 0f, config.c(), heuristic);
    }
}
