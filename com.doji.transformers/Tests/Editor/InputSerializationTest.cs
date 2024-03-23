using NUnit.Framework;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace Doji.AI.Transformers.Editor.Tests {

    public class InputSerializationTest {

        class InputContainer {
            public Input Input { get; set; }
        }

        [Test]
        public void SingleInput() {
            string text = "test";
            Input input = text;

            string s = JsonConvert.SerializeObject(input);
            var expectedJson = "{\"Type\":\"SingleInput\",\"Value\":\"test\"}";
            Assert.That(s, Is.EqualTo(expectedJson));

            Input deserialized = JsonConvert.DeserializeObject<Input>(s);
            Assert.IsInstanceOf<SingleInput>(deserialized);
            Assert.AreEqual(text, (deserialized as SingleInput).Text);
        }

        [Test]
        public void BatchInput() {
            var sequence = new List<string>() { "test1", "test2", "test3" };
            Input input = (BatchInput)sequence;

            string s = JsonConvert.SerializeObject(input);
            var expectedJson = "{\"Type\":\"BatchInput\",\"Value\":[\"test1\",\"test2\",\"test3\"]}";
            Assert.That(s, Is.EqualTo(expectedJson));

            Input deserialized = JsonConvert.DeserializeObject<Input>(s);
            Assert.IsInstanceOf<BatchInput>(deserialized);
            CollectionAssert.AreEqual(sequence, (deserialized as BatchInput).Sequence);
        }

        [Test]
        public void PretokenizedSingleInput() {
            var pretokenizedText = new List<string>() { "test1", "test2", "test3" };
            Input input = (PretokenizedSingleInput)pretokenizedText;
            string s = JsonConvert.SerializeObject(input);
            
            var expectedJson = "{\"Type\":\"PretokenizedSingleInput\",\"Value\":[\"test1\",\"test2\",\"test3\"]}";
            Assert.That(s, Is.EqualTo(expectedJson));

            Input deserialized = JsonConvert.DeserializeObject<Input>(s) as PretokenizedSingleInput;
            Assert.IsInstanceOf<PretokenizedSingleInput>(deserialized);
            CollectionAssert.AreEqual(pretokenizedText, (deserialized as PretokenizedSingleInput).PretokenizedText);
        }

        [Test]
        public void PretokenizedBatchInput() {
            var sequence = new List<List<string>>() {
                new List<string>() { "batch1test1", "batch1test2", "batch1test3" },
                new List<string>() { "batch2test1", "batch2test2", "batch2test3" },
                new List<string>() { "batch3test1", "batch3test2", "batch3test3" },
            };
            Input input = (PretokenizedBatchInput)sequence;
            string s = JsonConvert.SerializeObject(input);

            var expectedJson = "{\"Type\":\"PretokenizedBatchInput\",\"Value\":[[\"batch1test1\",\"batch1test2\",\"batch1test3\"],[\"batch2test1\",\"batch2test2\",\"batch2test3\"],[\"batch3test1\",\"batch3test2\",\"batch3test3\"]]}";
            Assert.That(s, Is.EqualTo(expectedJson));

            Input deserialized = JsonConvert.DeserializeObject<Input>(s);
            Assert.IsInstanceOf<PretokenizedBatchInput>(deserialized);
            CollectionAssert.AreEqual(sequence, (deserialized as PretokenizedBatchInput).Sequence);
        }
    }
}