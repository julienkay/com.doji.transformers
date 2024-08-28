using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class StopStringCriteria : StoppingCriteria {
        public PreTrainedTokenizerBase Tokenizer { get; }
        public List<string> StopStrings { get; }
        public StopStringCriteria(PreTrainedTokenizerBase tokenizer, List<string> stopStrings) {
            Tokenizer = tokenizer;
            StopStrings = stopStrings;
        }
        public override FunctionalTensor Apply(FunctionalTensor inputIds, FunctionalTensor scores) {
            throw new System.NotImplementedException();
        }
    }
}