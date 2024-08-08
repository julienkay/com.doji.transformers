using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class GenerateDecoderOnlyOutput : ModelOutput {
        public TensorInt Sequences {
            get {
                return Get<TensorInt>("sequences");
            }
            internal set {
                this["sequences"] = value;
            }
        }
        public List<TensorFloat> Scores {
            get {
                return Get<List<TensorFloat>>("scores");
            }
            internal set {
                this["scores"] = value;
            }
        }
        public List<TensorFloat> Logits {
            get {
                return Get<List<TensorFloat>>("logits");
            }
            internal set {
                this["logits"] = value;
            }
        }
        public List<Tensor> Attentions {
            get {
                return Get<List<Tensor>>("attentions");
            }
            internal set {
                this["attentions"] = value;
            }
        }
        public List<Tensor> HiddenStates {
            get {
                return Get<List<Tensor>>("hidden_states");
            }
            internal set {
                this["hidden_states"] = value;
            }
        }
        public object PastKeyValues {
            get {
                return Get<object>("past_key_values");
            }
            internal set {
                this["past_key_values"] = value;
            }
        }
    }
}