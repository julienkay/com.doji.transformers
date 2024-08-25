using Unity.Sentis;

namespace Doji.AI.Transformers {

    /// <summary>
    /// This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    /// By default, it uses the `model.generation_config.eos_token_id`.
    /// </summary>
    public class EosTokenCriteria : StoppingCriteria {

        /// <summary>
        /// The id(s) of the *end-of-sequence* token.
        /// </summary>
        public int[] EosTokenId { get; }

        public EosTokenCriteria(int[] eosTokenId) {
            EosTokenId = eosTokenId;
        }

        public override Tensor<int> Apply(Tensor<int> inputIds, Tensor<float> scores) {
            Tensor<int> isDone = Ops.Zeros<int>(new TensorShape(inputIds.shape[0]));
            Tensor<int> lastTokenInputs = Ops.Slice(inputIds, .., ^1);
            foreach (int eosToken in EosTokenId) {
                Tensor<int> eos = Ops.NewTensor<int>(eosToken);
                isDone = Ops.Or(isDone, Ops.Equal(lastTokenInputs, eos));
            }
            return isDone;
        }
    }
}