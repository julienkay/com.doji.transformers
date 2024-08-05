using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    public class LogitsProcessorList : List<LogitsProcessor> {
        public TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            foreach (var processor in this) {
                scores = processor.Apply(inputIds, scores);
            }
            return scores;
        }
    }
}