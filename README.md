<a href="https://www.doji-tech.com/">
  <img src="https://www.doji-tech.com/assets/favicon.ico" alt="doji logo" title="Doji" align="right" height="70" />
</a>

# Transformers
A Unity package to run pretrained transformer models with Unity Sentis

[OpenUPM] · [Documentation (coming soon)] · [Feedback/Questions]

## About

This is essentially a C# port of Hugging Face’s [transformers] library.

There are two use cases for this package right now:
- It's used by the [com.doji.diffusers] package to run Stable Diffusion models in Unity (most SD models use a ClipTokenizer for prompting, newer pipelines require additional ones like T5Tokenizer)
- To run small LLMs like [Phi-3](https://huggingface.co/julienkay/Phi-3-mini-4k-instruct_no_cache_uint8) in Unity (WIP)

## Roadmap:

Tokenizers
- [x] CLIPTokenizer
- [x] LLamaTokenizer
- [x] GPT2Tokenizer
- [ ] T5Tokenizer

LLMs
- [ ] Phi-3

The intention is to provide a similar API like Hugging Face's transformers library, so usage in Unity will look something like this:

 ```csharp
var tokenizer = AutoTokenizer.FromPretrained("julienkay/Phi-3-mini-4k-instruct_no_cache_uint8");
var model = Phi3ForCausalLM.FromPretrained("julienkay/Phi-3-mini-4k-instruct_no_cache_uint8");

var inputs = tokenizer.Encode("<input-prompt>");
var outputs = model.Generate(inputs);
var predictedText = tokenizer.Decode(outputs);
```

[OpenUPM]: https://openupm.com/packages/com.doji.transformers
[Documentation (coming soon)]: https://github.com/julienkay/com.doji.transformers
[Feedback/Questions]: https://discussions.unity.com/t/stable-diffusion-diffusers-transformers-package/332701
[transformers]: https://github.com/huggingface/transformers
[com.doji.diffusers]: https://github.com/julienkay/com.doji.diffusers
