using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    public class StoppingCriteriaList : List<StoppingCriteria> {
        public TensorInt Apply(TensorInt inputIds, TensorFloat scores) {
            throw new System.NotImplementedException();
        }
    }
}