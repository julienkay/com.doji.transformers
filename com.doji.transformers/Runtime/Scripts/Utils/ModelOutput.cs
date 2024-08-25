using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Base class for all model outputs as dataclass. Has a <see cref="Get"/> method that allows
    /// indexing by strings (like a dictionary) that will ignore the null attributes.
    /// Otherwise behaves like a regular dictionary.
    /// </summary>
    public abstract class ModelOutput : Dictionary<string, object> {
        public ModelOutput() : base() { }
        public T Get<T>(string key, T defaultValue = default) {
            if (TryGetValue(key, out object value)) {
                return (T)value;
            }
            return defaultValue;
        }
        public object Get(string key, object defaultValue = null) {
            return this.GetValueOrDefault(key, defaultValue);
        }
    }
    public class CausalLMOutputWithPast : ModelOutput {
        public Tensor<float> Logits { get; }
        public CausalLMOutputWithPast(Tensor<float> logits) : base() {
            Logits = logits;
            Add("logits", logits);
        }
    }
}