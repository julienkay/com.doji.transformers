using Unity.Sentis;
using static FunctionalUtils;

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

        public override FunctionalTensor Apply(FunctionalTensor inputIds, FunctionalTensor scores) {
            FunctionalTensor isDone = Zeros<int>(new TensorShape(inputIds.shape()[0]));
            FunctionalTensor lastTokenInputs = inputIds[.., ^1];
            //TODO: better torch.isin() implementation
            foreach (int eosToken in EosTokenId) {
                FunctionalTensor eos = Functional.Constant(eosToken);
                isDone = isDone | (lastTokenInputs == eos);
            }
            return isDone;
        }
    }
}