using Unity.Sentis;

namespace Doji.AI.Transformers {

    /// <summary>
    /// This class can be used to stop generation whenever the full generated number of tokens
    /// exceeds <see cref="MaxLength"/>. Keep in mind for decoder-only type of transformers,
    /// this will include the initial prompted tokens.
    /// </summary>
    public class MaxLengthCriteria : StoppingCriteria {

        /// <summary>
        /// The maximum length that the output sequence can have in number of tokens.
        /// </summary>
        public int MaxLength { get; }

        /// <summary>
        /// The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
        /// </summary>
        public int? MaxPositionEmbeddings { get; }

        public MaxLengthCriteria(int maxLength, int? maxPositionEmbeddings) {
            MaxLength = maxLength;
            MaxPositionEmbeddings = maxPositionEmbeddings;
        }

        public override TensorInt Apply(TensorInt inputIds, TensorFloat scores) {
            int curLen = inputIds.shape[-1];
            bool isDone = curLen >= MaxLength;
            if (MaxPositionEmbeddings != null && !isDone && curLen >= MaxPositionEmbeddings) {
                Log.Warning("This is a friendly reminder - the current text generation call will exceed the model's predefined " +
                    $"maximum length ({MaxPositionEmbeddings}). Depending on the model, you may observe " +
                    "exceptions, performance degradation, or nothing at all."
                );
            }
            return Ops.Full(inputIds.shape[0], isDone);
        }
    }
}