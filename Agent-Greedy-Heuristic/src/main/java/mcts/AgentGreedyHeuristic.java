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

import java.util.concurrent.TimeUnit;

public class AgentGreedyHeuristic implements GameAgent<CarcassonneGame, CarcassonneAction> {

    private final HeuristicConfiguration[] heuristics;
    public AgentGreedyHeuristic(){
        heuristics = new HeuristicConfiguration[]{HeuristicManager.createDefaultHeuristic()};
    }

    public AgentGreedyHeuristic(HeuristicConfiguration heuristic){
        this.heuristics = new HeuristicConfiguration[]{heuristic};
    }
    public AgentGreedyHeuristic(HeuristicConfiguration[] heuristics){
        this.heuristics = heuristics;
    }

    @Override
    public CarcassonneAction computeNextAction(CarcassonneGame game, long computationTime, TimeUnit timeUnit) {
        var board = game.getBoard();
        var possibleAction = board.calculatePossibleActionsUnique();

        if(possibleAction.isEmpty()){
            return null;
        }

        float[] result = new float[possibleAction.size()];
        for(int i = 0; i < heuristics.length; i++){
            var intermediateResult = heuristicsNormalized(heuristics[i],possibleAction,board);
            for(int j = 0; j < intermediateResult.length; j++){
                result[j] += intermediateResult[j]/ heuristics.length;
            }
        }

        int samp = AgentUtil.sampleWithTemperature(result,0.018, new java.util.Random());
        return possibleAction.getActionObject(samp);
    }

    public float[] heuristicsNormalized(HeuristicConfiguration config, ActionSet actions, State state){
        float[] result = new float[actions.size()];
        int cachedPositionRotation = Integer.MIN_VALUE;
        float cachedTileScore = 0f;

        for(int i = 0; i < actions.size(); i++){
            var act = actions.get(i);
            int x = CarcassonneActionLayoutBit.getX(act);
            int y = CarcassonneActionLayoutBit.getY(act);
            int rot = CarcassonneActionLayoutBit.getRotation(act);
            int positionRotationKey = (x << 10) ^ (y << 2) ^ rot;

            if (positionRotationKey != cachedPositionRotation) {
                cachedPositionRotation = positionRotationKey;
                cachedTileScore = HeuristicManager.tilePlacementScore(state, x, y, rot, config.positionHeuristik());
            }

            float heuristicValue = HeuristicManager.computePrior(state, act, cachedTileScore, config);
            result[i] = heuristicValue;
        }
        AgentUtil.normalizeInPlace(result);
        return result;
    }
}
