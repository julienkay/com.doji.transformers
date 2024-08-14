using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    public class StoppingCriteriaList : List<StoppingCriteria> {

        private Ops _ops;

        public StoppingCriteriaList(Ops ops) {
            _ops = ops;
        }

        public TensorInt Apply(TensorInt inputIds, TensorFloat scores) {
            TensorInt isDone = _ops.Zeros<TensorInt>(new TensorShape(inputIds.shape[0]));
            foreach (var criteria in this) {
                isDone = _ops.Or(isDone, criteria.Apply(inputIds, scores));
            }
            return isDone;
        }

        public new void Add(StoppingCriteria criteria) {
            if (criteria != null) {
                criteria.Ops = _ops;
            }
            base.Add(criteria);
        }
    }
}