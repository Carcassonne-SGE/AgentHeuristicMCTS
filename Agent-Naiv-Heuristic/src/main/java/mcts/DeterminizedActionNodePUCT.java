package mcts;

import abstractDeterminized.AbstractDeterminizedAgent;
import abstractDeterminized.AbstractPuctActionNode;
import model.collections.ActionSet;
import model.heuristic.HeuristicConfiguration;
import model.state.State;

/// DeterminizedActionNodePUCT
///
/// PUCT-based action node implementation for AgentNaiveHeuristic. Modifies selection scores
/// based on heuristic coefficient priors. Duruing Rollout the agent uses an epsilon-greedy
/// strategy to choose between random actions and greedy heuristic actions.
public class DeterminizedActionNodePUCT extends AbstractPuctActionNode<DeterminizedActionNodePUCT> {

    /// DeterminizedActionNodePUCT
    ///
    /// Constructor for DeterminizedActionNodePUCT.
    ///
    /// @param agent the determinized agent managing the search tree
    /// @param parent the parent node in the search tree
    /// @param action the action represented by this node
    /// @param checkpoint the checkpointed board state (only stored at checkpoints/root)
    /// @param heuristic the prior heuristic probability/score of this node's action
    /// @param explorationCoefficient the constant controlling exploration vs exploitation
    /// @param heuristicConfiguration the heuristic configuration used for rollout evaluation
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

    /// getSelectionScore
    ///
    /// Computes the selection score based on PUCT heuristic prior weighted by the
    /// agent's heuristic coefficient.
    ///
    /// @return the selection score
    /// getSelectionScore
    ///
    /// Computes the selection score based on PUCT heuristic prior weighted by the
    /// agent's heuristic coefficient.
    ///
    /// @return the selection score
    @Override
    public float getSelectionScore() {
        if (visits == 0) {
            return Float.MAX_VALUE;
        }
        // Map value to UCB domain
        float q = agent.ucbTransform(value / (float) visits);
        if (parent == null || parent.getVisits() <= 1) {
            return q;
        }
        // Calculate standard PUCT exploration multiplier
        float b = (float) (Math.sqrt(parent.getVisits()) / (1 + visits));
        AgentNaiveHeuristic heuristicAgent = (AgentNaiveHeuristic) agent;
        float hCoef = heuristicAgent.getHeuristicCoefficient();
        // Blend exploration coefficient and prior heuristic weight
        return q + (explorationCoefficient + hCoef * heuristic) * b;
    }

    /// select
    ///
    /// Selects the best child node based on PUCT scores, falling back to unvisited children first.
    ///
    /// @return the selected child node
    @Override
    public DeterminizedActionNodePUCT select() {
        if (children == null || children.length == 0) {
            return self();
        }

        DeterminizedActionNodePUCT best = selectUnvisitedOrBestUsb(children, children.length);
        assert best != null;
        return best.getVisits() == 0 ? best : best.select();
    }

    /// newChildrenArray
    ///
    /// Helper factory method to instantiate the children node array.
    ///
    /// @param size the size of the array
    /// @return the created array of children nodes
    @Override
    protected DeterminizedActionNodePUCT[] newChildrenArray(int size) {
        return new DeterminizedActionNodePUCT[size];
    }

    /// expand
    ///
    /// Expands the current node, initializing children with heuristic priors.
    ///
    /// @param state the current board state
    /// @return the expanded node
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

    /// simulate
    ///
    /// Performs simulations/rollouts from the current node using a combination of random
    /// and greedy heuristic action selections.
    ///
    /// @param state the board state to start the simulation from
    /// @return the evaluated reward/score from the simulation
    @Override
    public float simulate(State state) {
        AgentNaiveHeuristic heuristicAgent = (AgentNaiveHeuristic) agent;
        return averageHeuristicRollout(state, heuristicAgent.rolloutCount(),
                (rolloutState, actions) -> chooseRolloutAction(rolloutState, actions, heuristicAgent));
    }

    /// chooseRolloutAction
    ///
    /// Selection strategy during simulation rollouts, choosing between greedy heuristic
    /// decisions and random actions based on configured rollout greedy probability.
    ///
    /// @param state the current rollout board state
    /// @param actions the set of legal actions
    /// @param heuristicAgent the AgentNaiveHeuristic instance
    /// @return the chosen action
    private int chooseRolloutAction(State state, ActionSet actions, AgentNaiveHeuristic heuristicAgent) {
        // Epsilon-greedy: decide whether to use greedy heuristic prior or uniform random action
        if (agent.rand.nextFloat() < heuristicAgent.rolloutGreedyProbability) {
            return chooseGreedyHeuristicAction(state, actions, 0.017);
        } else {
            return actions.get(agent.rand.nextInt(actions.size()));
        }
    }
}
