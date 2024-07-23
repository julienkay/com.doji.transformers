using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class Phi3ForCausalLM : PretrainedModel {

        public Phi3ForCausalLM(Model model, PretrainedConfig config, BackendType backend = BackendType.GPUCompute) : base(model, config, backend) { }

        /// <summary>
        /// Instantiate a Phi3 model from a JSON configuration file.
        /// </summary>
        public static Phi3ForCausalLM FromPretrained(string model, BackendType backend = BackendType.GPUCompute) {
            return FromPretrained<Phi3ForCausalLM>(model, backend);
        }
    }
}