using Doji.AI.Transformers;
using System.Linq;
using Unity.Sentis;
using UnityEngine;

public class TestPhi3 : MonoBehaviour {

    private IBackend _backend;
    PreTrainedTokenizerBase _tokenizer;
    Phi3ForCausalLM _model;

    private void Start() {
        string modelId = "julienkay/Phi-3-mini-4k-instruct_no_cache_uint8";
        var prompt = "This is an example script .";

        _backend = WorkerFactory.CreateBackend(BackendType.GPUCompute);
        _tokenizer = AutoTokenizer.FromPretrained(modelId);
        _model = Phi3ForCausalLM.FromPretrained(modelId);

        var tokens = _tokenizer.Encode(prompt);
        Debug.Log(string.Join(", ", tokens.InputIds));

        int SEQUENCE_LENGTH = 4096;//?
        var sample = prompt;
        for (int i = 0; i < 10; i++) {
            var encodings = _tokenizer.Encode(sample);
            var input_ids = encodings.InputIds.ToArray();
            var attention_mask = encodings.AttentionMask.ToArray();
            //var token_type_ids = encodings.TokenTypeIds.ToArray();
            
            if (input_ids.Length >= SEQUENCE_LENGTH - 1) break;
            
            using TensorInt input_tensor = new TensorInt(new TensorShape(input_ids.Length), input_ids);
            using TensorInt attention_mask_tensor = new TensorInt(new TensorShape(attention_mask.Length), attention_mask);
            using TensorInt position_ids_tensor = new TensorInt(new TensorShape(input_ids.Length), Enumerable.Range(0, input_ids.Length).ToArray());//??
            var predictions = _model.Execute(input_tensor, attention_mask_tensor, position_ids_tensor);
            int[] nextToken = SampleNext(predictions);
            Debug.Log(nextToken);
            //sample += ' ' + _tokenizer.Decode(nextToken.ToList());
            break;
        }
        Debug.Log(sample);
    }

    private int[] SampleNext(TensorFloat predictions) {
        // Greedy approach.
        using TensorFloat probabilities = TensorFloat.AllocNoData(predictions.shape);
        _backend.Softmax(predictions, probabilities, axis: -1);
        using TensorInt next_token = TensorInt.AllocNoData(probabilities.shape.Reduce(-1, false));
        _backend.ArgMax(probabilities, next_token, -1, false);
        using TensorInt pred = next_token.ReadbackAndClone();
        Debug.Log(pred.count);
        return pred.ToReadOnlyArray();
    }

    private void OnDestroy() {
        _backend.Dispose();
        _model.Dispose();
    }
}