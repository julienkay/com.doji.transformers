namespace Doji.AI.Transformers {
    public class BeamSearchScorer {
        public BeamSearchScorer(
            int batchSize,
            int numBeams,
            float? lengthPenalty = 1.0f,
            StoppingCondition doEarlyStopping = StoppingCondition.False,
            int? numBeamHypsToKeep = 1,
            int? numBeamGroups = 1,
            int? maxLength = null) { }
    }
}