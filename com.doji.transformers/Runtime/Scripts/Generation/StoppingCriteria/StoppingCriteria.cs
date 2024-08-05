using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    /// <summary>
    /// This class can be used to stop generation whenever specific string sequences are generated.
    /// It preprocesses the strings together with the tokenizer vocab to find positions where tokens
    /// can validly complete the stop strings.
    /// </summary>
    public abstract class StoppingCriteria {
        //public abstract bool Apply(TensorInt inputIds, TensorFloat scores);
    }

    public class MaxLengthCriteria : StoppingCriteria {
        public int MaxLength { get; }
        public int? MaxPositionEmbeddings { get; }

        public MaxLengthCriteria(int maxLength, int? maxPositionEmbeddings) {
            MaxLength = maxLength;
            MaxPositionEmbeddings = maxPositionEmbeddings;
        }
    }

    public class MaxNewTokensCriteria : StoppingCriteria {
        public int StartLength { get; }
        public int MaxNewTokens { get; }

        public MaxNewTokensCriteria(int startLength, int maxNewTokens) {
            StartLength = startLength;
            MaxNewTokens = maxNewTokens;
        }
    }


    public class MaxTimeCriteria : StoppingCriteria {
        public double MaxTime { get; }

        public MaxTimeCriteria(double maxTime) {
            MaxTime = maxTime;
        }
    }

    public class StopStringCriteria : StoppingCriteria {
        public PreTrainedTokenizerBase Tokenizer { get; }
        public List<string> StopStrings { get; }

        public StopStringCriteria(PreTrainedTokenizerBase tokenizer, List<string> stopStrings) {
            Tokenizer = tokenizer;
            StopStrings = stopStrings;
        }
    }

    public class EosTokenCriteria : StoppingCriteria {
        public int[] EosTokenId { get; }

        public EosTokenCriteria(int[] eosTokenId) {
            EosTokenId = eosTokenId;
        }
    }
}