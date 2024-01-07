using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;

namespace Doji.AI.Transformers.Editor.Tests {

    public class ClipTokenizerTest {

        [Test]
        public void TestTokenize() {
            ClipTokenizer t = CreateTokenizer();
            List<string> tokens = t.Tokenize("lower newer");

            List<string> exptected = new List<string>() { "lo", "w", "er</w>", "n", "e", "w", "er</w>" };
            CollectionAssert.AreEqual(tokens, exptected);
        }

        private ClipTokenizer CreateTokenizer() {
            string[] vocabList = { "l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "lo", "l</w>", "w</w>", "r</w>", "t</w>", "low</w>", "er</w>", "lowest</w>", "newer</w>", "wider", "<unk>", "", "" };
            string[] mergesFile = { "#version: 0.2", "l o", "lo w</w>", "e r</w>" };
            Dictionary<string, int> vocabTokens = new Dictionary<string, int>();
            for (int i = 0; i < vocabList.Length; i++) {
                vocabTokens[vocabList[i]] = i;
            }

            Vocab vocab = new Vocab(vocabTokens);
            ClipTokenizer tokenizer = new ClipTokenizer(vocab, mergesFile);
            return tokenizer;
        }
    }
}