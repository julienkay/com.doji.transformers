using Unity.Sentis;

namespace Doji.AI.Transformers {

    /// <summary>
    /// This class can be used to stop generation whenever the generated number of tokens exceeds <see cref="MaxNewTokens"/>.
    /// Keep in mind for decoder-only type of transformers, this will ** not** include the initial prompted tokens.
    /// This is very close to <see cref="MaxLengthCriteria"/> but ignores the number of initial tokens.
    /// </summary>
    public class MaxNewTokensCriteria : StoppingCriteria {

        /// <summary>
        /// The number of initial tokens.
        /// </summary>
        public int StartLength { get; }

        /// <summary>
        /// The maximum number of tokens to generate.
        /// </summary>
        public int MaxNewTokens { get; }

        public int MaxLength { get; }

        public MaxNewTokensCriteria(int startLength, int maxNewTokens) {
            StartLength = startLength;
            MaxNewTokens = maxNewTokens;
            MaxLength = startLength + maxNewTokens;
        }

        public override FunctionalTensor Apply(FunctionalTensor inputIds, FunctionalTensor scores) {
            bool isDone = inputIds.shape()[-1] >= MaxLength;
            return FunctionalUtils.Full(inputIds.shape()[0], isDone);
        }
    }
}