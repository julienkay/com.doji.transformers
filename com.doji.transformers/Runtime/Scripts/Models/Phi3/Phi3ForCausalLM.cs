using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class Phi3ForCausalLM : PretrainedModel {

        protected override bool AcceptsAttentionMask => true;
        protected override bool HasEncoder => false;

        private Dictionary<string, Tensor> _inputs = new Dictionary<string, Tensor>();

        public Phi3ForCausalLM(Model model, PretrainedConfig config, BackendType backend = BackendType.GPUCompute) : base(model, config, backend) { }

        /// <summary>
        /// Instantiate a Phi3 model from a JSON configuration file.
        /// </summary>
        public static Phi3ForCausalLM FromPretrained(string model, BackendType backend = BackendType.GPUCompute) {
            return FromPretrained<Phi3ForCausalLM>(model, backend);
        }

        public TensorFloat Execute(TensorInt inputIds, TensorInt attentionMask, TensorInt positionIds) {
            _inputs["input_ids"] = inputIds;
            _inputs["attention_mask"] = attentionMask;
            _inputs["position_ids"] = positionIds;
            _worker.Execute(_inputs);
            return _worker.PeekOutput("logits") as TensorFloat;
        }
    }
}