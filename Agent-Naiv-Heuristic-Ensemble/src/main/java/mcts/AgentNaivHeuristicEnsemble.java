package mcts;

import abstractDeterminized.AbstractDeterminizedEnsembleAgent;
import at.ac.tuwien.ifs.sge.engine.Logger;
import core.AbstractAgentConfiguration;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;

import java.util.Random;

/**
 * AgentNaivHeuristicEnsemble implements an ensemble of MCTS agents that utilize
 * standard naive heuristic assumptions and configurable PUCT rollout strategies.
 */
public class AgentNaivHeuristicEnsemble extends AbstractDeterminizedEnsembleAgent {

    public AgentNaivHeuristicEnsemble() {
        this(null);
    }

  
    public AgentNaivHeuristicEnsemble(Logger logger) {
        this(logger, 0.4f);
    }

    
    public AgentNaivHeuristicEnsemble(float rolloutGreedyProbability) {
        this(null, rolloutGreedyProbability);
    }

    
    public AgentNaivHeuristicEnsemble(Logger logger, float rolloutGreedyProbability) {
        this(
                logger,
                new Random(99),
                //330 4.0
                new AbstractAgentConfiguration(10f, 10, 0.1f, 1),
                HeuristicManager.createDefaultHeuristic(),
                rolloutGreedyProbability
        );
    }

   
    public AgentNaivHeuristicEnsemble(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic
    ) {
        this(logger, rand, config, heuristic, 0.85f);
    }

   
    public AgentNaivHeuristicEnsemble(
            Logger logger,
            Random rand,
            AbstractAgentConfiguration config,
            HeuristicConfiguration heuristic,
            float rolloutGreedyProbability
    ) {
        super(
                logger,
                config,
                10,
                AbstractDeterminizedEnsembleAgent.pooledByVisitsAggregationMode(),
                rand,
                (subLogger, subConfig, subRand) -> new AgentNaivHeuristic(subLogger, subRand, subConfig, heuristic, rolloutGreedyProbability)
        );
    }
}
