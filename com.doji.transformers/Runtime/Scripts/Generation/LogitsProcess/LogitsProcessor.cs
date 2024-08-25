using System.Collections.Generic;
using System;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    /// <summary>
    /// Abstract base class for all logit processors that can be applied during generation.
    /// </summary>
    public abstract class LogitsProcessor {
        public abstract Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores);
    }
    public class MinLengthLogitsProcessor : LogitsProcessor {
        public MinLengthLogitsProcessor(int minLength, object eosTokenTensor) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class MinNewTokensLengthLogitsProcessor : LogitsProcessor {
        public MinNewTokensLengthLogitsProcessor(int inputIdsSeqLength, int minNewTokens, object eosTokenTensor) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class TemperatureLogitsWarper : LogitsProcessor {
        public float Temperature { get; }
        public TemperatureLogitsWarper(float temperature) {
            Temperature = temperature;
        }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class RepetitionPenaltyLogitsProcessor : LogitsProcessor {
        public RepetitionPenaltyLogitsProcessor(float penalty) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class EncoderRepetitionPenaltyLogitsProcessor : LogitsProcessor {
        public EncoderRepetitionPenaltyLogitsProcessor(float penalty, object encoderInputIds) { }

        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class TopPLogitsWarper : LogitsProcessor {
        public float TopP { get; }
        public int MinTokensToKeep { get; }
        public TopPLogitsWarper(float topP, int minTokensToKeep) {
            TopP = topP;
            MinTokensToKeep = minTokensToKeep;
        }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class TopKLogitsWarper : LogitsProcessor {
        public int TopK { get; }
        public int MinTokensToKeep { get; }
        public TopKLogitsWarper(int topK, int minTokensToKeep) {
            TopK = topK;
            MinTokensToKeep = minTokensToKeep;
        }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class MinPLogitsWarper : LogitsProcessor {
        public float MinP { get; }
        public int MinTokensToKeep { get; }
        public MinPLogitsWarper(float minP, int minTokensToKeep) {
            MinP = minP;
            MinTokensToKeep = minTokensToKeep;
        }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class TypicalLogitsWarper : LogitsProcessor {
        public float TypicalP { get; }
        public int MinTokensToKeep { get; }
        public TypicalLogitsWarper(float typicalP, int minTokensToKeep) {
            TypicalP = typicalP;
            MinTokensToKeep = minTokensToKeep;
        }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class EpsilonLogitsWarper : LogitsProcessor {
        public float EpsilonCutoff { get; }
        public int MinTokensToKeep { get; }
        public EpsilonLogitsWarper(float epsilonCutoff, int minTokensToKeep) {
            EpsilonCutoff = epsilonCutoff;
            MinTokensToKeep = minTokensToKeep;
        }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class EtaLogitsWarper : LogitsProcessor {
        public float EtaCutoff { get; }
        public int MinTokensToKeep { get; }
        public EtaLogitsWarper(float etaCutoff, int minTokensToKeep) {
            EtaCutoff = etaCutoff;
            MinTokensToKeep = minTokensToKeep;
        }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class NoRepeatNGramLogitsProcessor : LogitsProcessor {
        public NoRepeatNGramLogitsProcessor(int noRepeatNGramSize) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class EncoderNoRepeatNGramLogitsProcessor : LogitsProcessor {
        public EncoderNoRepeatNGramLogitsProcessor(int encoderNoRepeatNGramSize, object encoderInputIds) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class SequenceBiasLogitsProcessor : LogitsProcessor {
        public SequenceBiasLogitsProcessor(object sequenceBias) { }

        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class NoBadWordsLogitsProcessor : LogitsProcessor {
        public NoBadWordsLogitsProcessor(List<List<int>> badWordsIds, object eosTokenTensor) { }

        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class PrefixConstrainedLogitsProcessor : LogitsProcessor {
        public PrefixConstrainedLogitsProcessor(Func<int, List<int>> prefixAllowedTokensFn, int numBeams) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class HammingDiversityLogitsProcessor : LogitsProcessor {
        public HammingDiversityLogitsProcessor(float diversityPenalty, int numBeams, int numBeamGroups) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class ForcedBOSTokenLogitsProcessor : LogitsProcessor {
        public ForcedBOSTokenLogitsProcessor(int forcedBosTokenId) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class ForcedEOSTokenLogitsProcessor : LogitsProcessor {
        public ForcedEOSTokenLogitsProcessor(int maxLength, int forcedEosTokenId) { }

        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class InfNanRemoveLogitsProcessor : LogitsProcessor {
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class ExponentialDecayLengthPenalty : LogitsProcessor {
        public ExponentialDecayLengthPenalty((int startIndex, float decayFactor)? exponentialDecayLengthPenalty, object eosTokenTensor, int inputIdsSeqLength) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class LogitNormalization : LogitsProcessor {
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class SuppressTokensAtBeginLogitsProcessor : LogitsProcessor {
        public SuppressTokensAtBeginLogitsProcessor(List<int> beginSuppressTokens, int beginIndex) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class SuppressTokensLogitsProcessor : LogitsProcessor {
        public SuppressTokensLogitsProcessor(List<int> suppressTokens) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class ForceTokensLogitsProcessor : LogitsProcessor {
        public ForceTokensLogitsProcessor(List<List<int>> forcedDecoderIds, bool hasWarned) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class WhisperTimeStampLogitsProcessor : LogitsProcessor {
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class WhisperNoSpeechDetection : LogitsProcessor {
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class IfierFreeGuidanceLogitsProcessor : LogitsProcessor {
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class AlternatingCodebooksLogitsProcessor : LogitsProcessor {
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class UnbatchedClassifierFreeGuidanceLogitsProcessor : LogitsProcessor {
        public UnbatchedClassifierFreeGuidanceLogitsProcessor(float guidanceScale, object self, object unconditionalIds, object unconditionalAttentionMask, bool useCache) { }

        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class BarkEosPrioritizerLogitsProcessor : LogitsProcessor {
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }

    public class WatermarkLogitsProcessor : LogitsProcessor {
        public WatermarkLogitsProcessor(int vocabSize, float greenlistRatio, float bias, int hashingKey, string seedingScheme, int contextWidth) { }
        public override Tensor<float> Apply(Tensor<int> inputIds, Tensor<float> scores) { throw new NotImplementedException(); }
    }
}