using System.Linq;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Transformers.Samples {

    // before running this sample, go to https://huggingface.co/julienkay/Phi-3-mini-4k-instruct_no_cache_uint8
    // download all the files and place them in StreamingAssets/julienkay/Phi-3-mini-4k-instruct_no_cache_uint8

    public class Phi3ForCausalLMSample : MonoBehaviour {

        private IBackend _backend;
        private PreTrainedTokenizerBase _tokenizer;
        private Phi3ForCausalLM _model;

        private void Start() {
            string modelId = "julienkay/Phi-3-mini-4k-instruct_no_cache_uint8";
            var prompt = "<|user|>\nCan you provide ways to eat combinations of bananas and dragonfruits?<|end|>\n<|assistant|>\n";
            _backend = WorkerFactory.CreateBackend(BackendType.GPUCompute);
            _tokenizer = AutoTokenizer.FromPretrained(modelId);
            _model = Phi3ForCausalLM.FromPretrained(modelId);

            var encodings = _tokenizer.Encode(prompt);
            var inputIds = encodings.InputIds.ToArray();

            using TensorInt inputTensor = new TensorInt(new TensorShape(1, inputIds.Length), inputIds);
            _model.GenerationConfig.MaxNewTokens = 20;
            var result = _model.Generate(inputTensor);
            var seq = result.Get<TensorInt>("sequences");
            seq = seq.ReadbackAndClone();
            string output = _tokenizer.Decode(seq.ToReadOnlyArray().ToList(), skipSpecialTokens: true, cleanUpTokenizationSpaces: false);
            Debug.Log(output);
        }

        private void OnDestroy() {
            _backend.Dispose();
            _model.Dispose();
        }
    }
}