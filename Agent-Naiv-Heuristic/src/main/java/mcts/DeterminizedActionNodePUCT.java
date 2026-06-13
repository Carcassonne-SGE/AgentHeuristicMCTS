package mcts;

import abstractDeterminized.AbstractDeterminizedAgent;
import abstractDeterminized.AbstractPuctActionNode;
import model.collections.ActionSet;
import model.heuristic.HeuristicConfiguration;
import model.state.State;

public class DeterminizedActionNodePUCT extends AbstractPuctActionNode<DeterminizedActionNodePUCT> {

    public DeterminizedActionNodePUCT(
            AbstractDeterminizedAgent<DeterminizedActionNodePUCT> agent,
            DeterminizedActionNodePUCT parent,
            int action,
            State checkpoint,
            float heuristic,
            float explorationCoefficient,
            HeuristicConfiguration heuristicConfiguration) {
        super(agent, parent, action, checkpoint, heuristic, explorationCoefficient, heuristicConfiguration);
    }

    @Override
    public float getSelectionScore() {
        if (visits == 0) {
            return Float.MAX_VALUE;
        }
        float q = agent.ucbTransform(value / (float) visits);
        if (parent == null || parent.getVisits() <= 1) {
            return q;
        }
        float b = (float) (Math.sqrt(parent.getVisits()) / (1 + visits));
        AgentNaiveHeuristic heuristicAgent = (AgentNaiveHeuristic) agent;
        float hCoef = heuristicAgent.getHeuristicCoefficient();
        return q + (explorationCoefficient + hCoef * heuristic) * b;
    }

    @Override
    public DeterminizedActionNodePUCT select() {
        if (children == null || children.length == 0) {
            return self();
        }

        DeterminizedActionNodePUCT best = selectUnvisitedOrBestUsb(children, children.length);
        assert best != null;
        return best.getVisits() == 0 ? best : best.select();
    }

    @Override
    protected DeterminizedActionNodePUCT[] newChildrenArray(int size) {
        return new DeterminizedActionNodePUCT[size];
    }

    @Override
    public DeterminizedActionNodePUCT expand(State state) {
        return expandWithHeuristicScores(
                state,
                true,
                (childAction, normalizedPrior) -> new DeterminizedActionNodePUCT(
                        agent,
                        this,
                        childAction,
                        null,
                        normalizedPrior,
                        explorationCoefficient,
                        heuristicConfiguration));
    }

    @Override
    public float simulate(State state) {
        AgentNaiveHeuristic heuristicAgent = (AgentNaiveHeuristic) agent;
        return averageHeuristicRollout(state, heuristicAgent.rolloutCount(),
                (rolloutState, actions) -> chooseRolloutAction(rolloutState, actions, heuristicAgent));
    }

    private int chooseRolloutAction(State state, ActionSet actions, AgentNaiveHeuristic heuristicAgent) {
        if (agent.rand.nextFloat() < heuristicAgent.rolloutGreedyProbability) {
            return chooseGreedyHeuristicAction(state, actions, 0.017);
        } else {
            return actions.get(agent.rand.nextInt(actions.size()));
        }
    }
}
