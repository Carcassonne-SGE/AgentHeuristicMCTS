package mcts;

import at.ac.tuwien.ifs.sge.agent.GameAgent;
import core.AgentUtil;
import model.collections.ActionSet;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;
import model.state.State;
import model.bits.CarcassonneActionLayoutBit;
import sge.CarcassonneAction;
import sge.CarcassonneGame;

import at.ac.tuwien.ifs.sge.engine.Logger;
import java.util.concurrent.TimeUnit;

/// AgentGreedyHeuristic
///
/// An agent that greedily evaluates and selects game actions based purely on the given
/// heuristic policy configuration(s). Uses the priors to create a pdf using softmax and samples
/// from that with a very low temperature
public class AgentGreedyHeuristic implements GameAgent<CarcassonneGame, CarcassonneAction> {

    private final HeuristicConfiguration[] heuristics;

    /// AgentGreedyHeuristic
    ///
    /// Constructor creating an agent with a logger and the default heuristic.
    ///
    /// @param logger the logger instance
    public AgentGreedyHeuristic(Logger logger) {
        this();
    }

    /// AgentGreedyHeuristic
    ///
    /// Constructor creating an agent with the default heuristic.
    public AgentGreedyHeuristic() {
        heuristics = new HeuristicConfiguration[] { HeuristicManager.createDefaultHeuristic() };
    }

    /// AgentGreedyHeuristic
    ///
    /// Constructor creating an agent with a specific heuristic configuration.
    ///
    /// @param heuristic the heuristic configuration to use
    public AgentGreedyHeuristic(HeuristicConfiguration heuristic) {
        this.heuristics = new HeuristicConfiguration[] { heuristic };
    }

    /// AgentGreedyHeuristic
    ///
    /// Constructor creating an agent with multiple heuristic configurations.
    ///
    /// @param heuristics the array of heuristic configurations to use
    public AgentGreedyHeuristic(HeuristicConfiguration[] heuristics) {
        this.heuristics = heuristics;
    }

    /// computeNextAction
    ///
    /// Computes and returns the next action by evaluating all legal actions using
    /// the
    /// configured heuristics, normalizing the scores, and sampling the best action.
    ///
    /// @param game the Carcassonne game instance
    /// @param computationTime the search budget duration
    /// @param timeUnit the unit of computationTime
    /// @return the selected CarcassonneAction
    @Override
    public CarcassonneAction computeNextAction(CarcassonneGame game, long computationTime, TimeUnit timeUnit) {
        var board = game.getBoard();
        var possibleAction = board.calculatePossibleActionsUnique();

        if (possibleAction.isEmpty()) {
            return null;
        }

        // Aggregate normalized score distributions across all configured heuristics
        float[] result = new float[possibleAction.size()];
        for (int i = 0; i < heuristics.length; i++) {
            var intermediateResult = heuristicsNormalized(heuristics[i], possibleAction, board);
            for (int j = 0; j < intermediateResult.length; j++) {
                result[j] += intermediateResult[j] / heuristics.length;
            }
        }

        // Softmax-sample the best action index with a low temperature for near-greedy
        // choice
        int samp = AgentUtil.sampleWithTemperature(result, 0.018, new java.util.Random());
        return possibleAction.getActionObject(samp);
    }

    /// heuristicsNormalized
    ///
    /// Evaluates prior heuristic scores for all actions in the given state using a
    /// specific
    /// heuristic configuration, and normalizes them in-place.
    ///
    /// @param config the heuristic configuration
    /// @param actions the set of possible actions
    /// @param state the current board state
    /// @return the normalized prior score array
    public float[] heuristicsNormalized(HeuristicConfiguration config, ActionSet actions, State state) {
        float[] result = HeuristicManager.computePriors(state, actions, config);
        // Normalize the resulting prior scores to represent a probability distribution
        AgentUtil.normalizeInPlace(result);
        return result;
    }
}
