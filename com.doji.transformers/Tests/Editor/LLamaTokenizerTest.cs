using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers.Editor.Tests {

    /// <summary>
    /// LLamaTokenizerTest test with a reduced vocabulary.
    /// </summary>
    public class LLamaTokenizerTest {

        public static IEnumerable TokenizeTestData {
            get {
                yield return new TestCaseData("The quick brown fox jumps over the lazy dog.").Returns(new List<string>() { "▁The", "▁quick", "▁brown", "▁fo", "x", "▁j", "umps", "▁over", "▁the", "▁lazy", "▁dog", "." });
            }
        }

        public static IEnumerable EncodeTestData {
            get {
                yield return new TestCaseData("The quick brown fox jumps over the lazy dog.").Returns(new List<int>() { 450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889 });
            }
        }

        private static List<string> RoundtripInput = new List<string>() { "The quick brown fox jumps over the lazy dog." };

        [Test]
        [TestCaseSource(nameof(TokenizeTestData))]
        public List<string> TestTokenize(string text) {
            LlamaTokenizer t = CreateTokenizer();
            List<string> tokens = t.Tokenize(text);
            return tokens;
        }

        [Test]
        public void TestEncodeType() {
            LlamaTokenizer t = CreateTokenizer();
            var encoding = t.Encode("lower newer");

            Assert.IsTrue(encoding.ContainsKey("input_ids"), "Encoded ids not found in 'input_ids'.");
            object encodedIds = encoding["input_ids"];
            Assert.IsTrue(encodedIds is ICollection, "Unexpected type for encoded text.");
            Assert.IsTrue(encoding is InputEncoding, "Unexpected type for encoding.");
        }

        [Test]
        [TestCaseSource(nameof(EncodeTestData))]
        public IEnumerable<int> TestEncode(string text) {
            LlamaTokenizer t = CreateTokenizer();
            InputEncoding encoding = t.Encode(text) as InputEncoding;
            var encodedIds = encoding.InputIds;
            return encodedIds;
        }

        [Test]
        public void TestRoundtrip([ValueSource(nameof(RoundtripInput))] string prompt) {
            LlamaTokenizer t = CreateTokenizer();
            var result = t.Decode(t.Encode(prompt).InputIds.ToList(), skipSpecialTokens: true);
            Assert.That(result, Is.EqualTo(prompt));
        }

        /// <summary>
        /// Creates a LlamaTokenizer.
        /// </summary>
        private LlamaTokenizer CreateTokenizer() {
            // create LlamaTokenizer with microsoft/Phi-3-mini-4k-instruct settings
            string tokenizerConfig = "{\"add_bos_token\":false,\"add_eos_token\":false,\"added_tokens_decoder\":{\"0\":{\"content\":\"<unk>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":false,\"single_word\":false,\"special\":true},\"1\":{\"content\":\"<s>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":false,\"single_word\":false,\"special\":true},\"2\":{\"content\":\"</s>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":false},\"32000\":{\"content\":\"<|endoftext|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":false,\"single_word\":false,\"special\":true},\"32001\":{\"content\":\"<|assistant|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true},\"32002\":{\"content\":\"<|placeholder1|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true},\"32003\":{\"content\":\"<|placeholder2|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true},\"32004\":{\"content\":\"<|placeholder3|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true},\"32005\":{\"content\":\"<|placeholder4|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true},\"32006\":{\"content\":\"<|system|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true},\"32007\":{\"content\":\"<|end|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true},\"32008\":{\"content\":\"<|placeholder5|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true},\"32009\":{\"content\":\"<|placeholder6|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true},\"32010\":{\"content\":\"<|user|>\",\"lstrip\":false,\"normalized\":false,\"rstrip\":true,\"single_word\":false,\"special\":true}},\"bos_token\":\"<s>\",\"chat_template\":\"{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\\n' + message['content'] + '<|end|>\\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\\n' + message['content'] + '<|end|>\\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\\n' + message['content'] + '<|end|>\\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% else %}{{ eos_token }}{% endif %}\",\"clean_up_tokenization_spaces\":false,\"eos_token\":\"<|endoftext|>\",\"legacy\":false,\"model_max_length\":4096,\"pad_token\":\"<|endoftext|>\",\"padding_side\":\"left\",\"sp_model_kwargs\":{},\"tokenizer_class\":\"LlamaTokenizer\",\"unk_token\":\"<unk>\",\"use_default_system_prompt\":false}";
            var config = TokenizerConfig.Deserialize(tokenizerConfig);
            string modelPath = "Packages/com.doji.transformers/Tests/Editor/Resources/Phi-3-mini-4k-instruct/tokenizer.model";
            return new LlamaTokenizer(modelPath, config);
        }
    }
}