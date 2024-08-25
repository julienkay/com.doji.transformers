using Unity.Sentis;

namespace Doji.AI.Transformers {

    /// <summary>
    /// This class can be used to stop generation whenever specific string sequences are generated.
    /// It preprocesses the strings together with the tokenizer vocab to find positions where tokens
    /// can validly complete the stop strings.
    /// </summary>
    public abstract class StoppingCriteria {
        internal Ops Ops { get; set; }
        public abstract Tensor<int> Apply(Tensor<int> inputIds, Tensor<float> scores);
    }
}