using Doji.AI.Transformers;
using System.IO;
using System.Linq;
using Unity.Sentis;
using UnityEngine;

public class TestPhi3 : MonoBehaviour {

    private IBackend _backend;
    private PreTrainedTokenizerBase _tokenizer;
    private Phi3ForCausalLM _model;

    private void Start() {
        string modelId = "julienkay/Phi-3-mini-4k-instruct_no_cache_uint8";
        var prompt = "This is a sample script .";

        _backend = WorkerFactory.CreateBackend(BackendType.GPUCompute);
        _tokenizer = AutoTokenizer.FromPretrained(modelId);
        _model = Phi3ForCausalLM.FromPretrained(modelId);

        var encodings = _tokenizer.Encode(prompt);
        var inputIds = encodings.InputIds.ToArray();
        //var attention_mask = encodings.AttentionMask.ToArray();
        //var token_type_ids = encodings.TokenTypeIds.ToArray();
            
        using TensorInt inputTensor = new TensorInt(new TensorShape(1, inputIds.Length), inputIds);
        var generationConfig = GenerationConfig.Deserialize(File.ReadAllText("Assets/StreamingAssets/julienkay/Phi-3-mini-4k-instruct_no_cache_uint8/generation_config.json"));
        generationConfig.MaxNewTokens = 30;
        var result = _model.Generate(inputTensor, generationConfig);
        var seq = result.Get<TensorInt>("sequences");
        seq = seq.ReadbackAndClone();
        string output = _tokenizer.Decode(seq.ToReadOnlyArray().ToList(), skipSpecialTokens: true, cleanUpTokenizationSpaces: false);
        Debug.Log("Predicted IDs: " + string.Join(", ", seq.ToReadOnlyArray()));
        Debug.Log(output);
    }

    private void OnDestroy() {
        _backend.Dispose();
        _model.Dispose();
    }
}