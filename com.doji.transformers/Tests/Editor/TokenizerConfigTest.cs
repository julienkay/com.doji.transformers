using NUnit.Framework;

namespace Doji.AI.Transformers.Editor.Tests {

    public class TokenizerConfigTest {

        [Test]
        public void TestDeserialize() {
            string json = "{\r\n  \"add_prefix_space\": false,\r\n  \"bos_token\": {\r\n    \"__type\": \"AddedToken\",\r\n    \"content\": \"<|startoftext|>\",\r\n    \"lstrip\": false,\r\n    \"normalized\": true,\r\n    \"rstrip\": false,\r\n    \"single_word\": false\r\n  },\r\n  \"do_lower_case\": true,\r\n  \"eos_token\": {\r\n    \"__type\": \"AddedToken\",\r\n    \"content\": \"<|endoftext|>\",\r\n    \"lstrip\": false,\r\n    \"normalized\": true,\r\n    \"rstrip\": false,\r\n    \"single_word\": false\r\n  },\r\n  \"errors\": \"replace\",\r\n  \"model_max_length\": 77,\r\n  \"name_or_path\": \"/home/anton_huggingface_co/.cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/245f52e962f4c0733f56daa14d2c85d3d2210e13/tokenizer\",\r\n  \"pad_token\": \"<|endoftext|>\",\r\n  \"special_tokens_map_file\": \"./special_tokens_map.json\",\r\n  \"tokenizer_class\": \"CLIPTokenizer\",\r\n  \"unk_token\": {\r\n    \"__type\": \"AddedToken\",\r\n    \"content\": \"<|endoftext|>\",\r\n    \"lstrip\": false,\r\n    \"normalized\": true,\r\n    \"rstrip\": false,\r\n    \"single_word\": false\r\n  }\r\n}";
            TokenizerConfig config = TokenizerConfig.Deserialize(json);
        
            Assert.That(config, Is.Not.Null);
            Assert.That(config.BosToken == "<|startoftext|>");
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
    }
}