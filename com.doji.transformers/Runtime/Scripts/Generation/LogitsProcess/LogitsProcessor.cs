using System.Collections.Generic;
using System;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    /// <summary>
    /// Abstract base class for all logit processors that can be applied during generation.
    /// </summary>
    public abstract class LogitsProcessor {
        public abstract TensorFloat Apply(TensorInt inputIds, TensorFloat scores);
    }

    public class UnbatchedClassifierFreeGuidanceLogitsProcessor : LogitsProcessor {
        public UnbatchedClassifierFreeGuidanceLogitsProcessor(float guidanceScale, object self, object unconditionalIds, object unconditionalAttentionMask, bool useCache) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class SequenceBiasLogitsProcessor : LogitsProcessor {
        public SequenceBiasLogitsProcessor(object sequenceBias) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class HammingDiversityLogitsProcessor : LogitsProcessor {
        public HammingDiversityLogitsProcessor(float diversityPenalty, int numBeams, int numBeamGroups) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class EncoderRepetitionPenaltyLogitsProcessor : LogitsProcessor {
        public EncoderRepetitionPenaltyLogitsProcessor(float penalty, object encoderInputIds) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class RepetitionPenaltyLogitsProcessor : LogitsProcessor {
        public RepetitionPenaltyLogitsProcessor(float penalty) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class NoRepeatNGramLogitsProcessor : LogitsProcessor {
        public NoRepeatNGramLogitsProcessor(int noRepeatNGramSize) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class EncoderNoRepeatNGramLogitsProcessor : LogitsProcessor {
        public EncoderNoRepeatNGramLogitsProcessor(int encoderNoRepeatNGramSize, object encoderInputIds) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class NoBadWordsLogitsProcessor : LogitsProcessor {
        public NoBadWordsLogitsProcessor(List<List<int>> badWordsIds, object eosTokenTensor) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class MinLengthLogitsProcessor : LogitsProcessor {
        public MinLengthLogitsProcessor(int minLength, object eosTokenTensor) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class MinNewTokensLengthLogitsProcessor : LogitsProcessor {
        public MinNewTokensLengthLogitsProcessor(int inputIdsSeqLength, int minNewTokens, object eosTokenTensor) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class PrefixConstrainedLogitsProcessor : LogitsProcessor {
        public PrefixConstrainedLogitsProcessor(Func<int, List<int>> prefixAllowedTokensFn, int numBeams) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class ForcedBOSTokenLogitsProcessor : LogitsProcessor {
        public ForcedBOSTokenLogitsProcessor(int forcedBosTokenId) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class ForcedEOSTokenLogitsProcessor : LogitsProcessor {
        public ForcedEOSTokenLogitsProcessor(int maxLength, int forcedEosTokenId) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class InfNanRemoveLogitsProcessor : LogitsProcessor {
        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class ExponentialDecayLengthPenalty : LogitsProcessor {
        public ExponentialDecayLengthPenalty((int startIndex, float decayFactor)? exponentialDecayLengthPenalty, object eosTokenTensor, int inputIdsSeqLength) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class SuppressTokensLogitsProcessor : LogitsProcessor {
        public SuppressTokensLogitsProcessor(List<int> suppressTokens) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class SuppressTokensAtBeginLogitsProcessor : LogitsProcessor {
        public SuppressTokensAtBeginLogitsProcessor(List<int> beginSuppressTokens, int beginIndex) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class ForceTokensLogitsProcessor : LogitsProcessor {
        public ForceTokensLogitsProcessor(List<List<int>> forcedDecoderIds, bool hasWarned) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class WatermarkLogitsProcessor : LogitsProcessor {
        public WatermarkLogitsProcessor(int vocabSize, float greenlistRatio, float bias, int hashingKey, string seedingScheme, int contextWidth) { }

        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
    public class LogitNormalization : LogitsProcessor {
        public override TensorFloat Apply(TensorInt inputIds, TensorFloat scores) {
            throw new NotImplementedException();
        }
    }
}