using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    public class LogitsProcessorList : List<LogitsProcessor> {
        public Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) {
            foreach (var processor in this) {
                scores = processor.Apply(inputIds, scores);
            }
            return scores;
        }
    }
}