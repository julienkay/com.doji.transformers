using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    public class StoppingCriteriaList : List<StoppingCriteria> {

        public StoppingCriteriaList(FunctionalGraph ops) { }

        public FunctionalTensor Apply(FunctionalTensor inputIds, FunctionalTensor scores) {
            FunctionalTensor isDone = FunctionalUtils.Zeros<int>(new TensorShape(inputIds.shape()[0]));
            foreach (var criteria in this) {
                isDone = isDone | criteria.Apply(inputIds, scores);
            }
            return isDone;
        }
    }
}