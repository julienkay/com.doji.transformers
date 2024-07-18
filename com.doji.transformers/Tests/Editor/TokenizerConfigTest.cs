using NUnit.Framework;

namespace Doji.AI.Transformers.Editor.Tests {

    public class TokenizerConfigTest {

        [Test]
        public void TestDeserialize() {
            // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/tokenizer/tokenizer_config.json
            string json = @"{""add_prefix_space"":false,""bos_token"":{""__type"":""AddedToken"",""content"":""<|startoftext|>"",""lstrip"":false,""normalized"":true,""rstrip"":false,""single_word"":false},""do_lower_case"":true,""eos_token"":{""__type"":""AddedToken"",""content"":""<|endoftext|>"",""lstrip"":false,""normalized"":true,""rstrip"":false,""single_word"":false},""errors"":""replace"",""model_max_length"":77,""name_or_path"":""openai/clip-vit-large-patch14"",""pad_token"":""<|endoftext|>"",""special_tokens_map_file"":""./special_tokens_map.json"",""tokenizer_class"":""CLIPTokenizer"",""unk_token"":{""__type"":""AddedToken"",""content"":""<|endoftext|>"",""lstrip"":false,""normalized"":true,""rstrip"":false,""single_word"":false}}";
            TokenizerConfig config = TokenizerConfig.Deserialize(json);
        
            Assert.That(config, Is.Not.Null);
            Assert.That(config.BosToken.Content, Is.EqualTo("<|startoftext|>"));
            Assert.That(config.BosToken, Is.TypeOf(typeof(AddedToken)));
            AddedToken bosToken = config.BosToken as AddedToken;
            Assert.That(bosToken.Lstrip == false);
            Assert.That(bosToken.Normalized == true);
            Assert.That(bosToken.Rstrip == false);
            Assert.That(bosToken.SingleWord == false);
            Assert.That(config.DoLowerCase == true);
            Assert.That(config.PadToken, Is.TypeOf(typeof(TokenString)));
            Assert.That(config.ModelMaxLength, Is.EqualTo(77));
            Assert.That(config.SepToken, Is.Null);
            Assert.That(config.ClsToken, Is.Null);
            Assert.That(config.MaskToken, Is.Null);
        }

        [Test]
        public void TestDeserializeAddedTokens() {
            // https://huggingface.co/stabilityai/sdxl-turbo/blob/main/tokenizer/tokenizer_config.json
            string json = @"{""add_prefix_space"":false,""added_tokens_decoder"":{""49406"":{""content"":""<|startoftext|>"",""lstrip"":false,""normalized"":true,""rstrip"":false,""single_word"":false,""special"":true},""49407"":{""content"":""<|endoftext|>"",""lstrip"":false,""normalized"":true,""rstrip"":false,""single_word"":false,""special"":true}},""bos_token"":""<|startoftext|>"",""clean_up_tokenization_spaces"":true,""do_lower_case"":true,""eos_token"":""<|endoftext|>"",""errors"":""replace"",""model_max_length"":77,""pad_token"":""<|endoftext|>"",""tokenizer_class"":""CLIPTokenizer"",""unk_token"":""<|endoftext|>""}";
            TokenizerConfig config = TokenizerConfig.Deserialize(json);
            Assert.That(config, Is.Not.Null);
            Assert.That(config.BosToken.Content, Is.EqualTo("<|startoftext|>"));
            Assert.That(config.BosToken, Is.TypeOf(typeof(TokenString)));
            TokenString bosToken = config.BosToken as TokenString;
            Assert.That(config.DoLowerCase == true);
            Assert.That(config.PadToken, Is.TypeOf(typeof(TokenString)));
            Assert.That(config.ModelMaxLength, Is.EqualTo(77));
            Assert.That(config.SepToken, Is.Null);
            Assert.That(config.ClsToken, Is.Null);
            Assert.That(config.MaskToken, Is.Null);
            Assert.That(config.AddedTokensDecoder, Is.Not.Null);
            foreach(var token in config.AddedTokensDecoder) {
                UnityEngine.Debug.Log(token.Value.ToString());
            }
        }
    }
}