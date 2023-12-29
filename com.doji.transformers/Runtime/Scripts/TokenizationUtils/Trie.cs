using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Trie in Python. Creates a Trie out of a list of words.
    /// The trie is used to split on `added_tokens` in one pass
    /// Loose reference https://en.wikipedia.org/wiki/Trie
    /// </summary>
    internal class Trie {

        private readonly HashSet<string> _tokens = new HashSet<string>();
        public SortedDictionary<char, object> data = new SortedDictionary<char, object>();

        public void Add(string word) {
            if (string.IsNullOrEmpty(word)) {
                // Prevent empty string
                return;
            }

            _tokens.Add(word);
            var node = data;

            foreach (var ch in word) {
                if (!node.ContainsKey(ch)) {
                    node[ch] = new SortedDictionary<char, object>();
                }
                node = (SortedDictionary<char, object>)node[ch];
            }

            // Use '\0' to represent the empty string
            node['\0'] = 1;
        }

        public override string ToString() {
            StringBuilder s = new StringBuilder();
            ToString(data, s);
            return s.ToString();
        }
        
        private void ToString(SortedDictionary<char, object> data, StringBuilder s) {
            int i = 0;
            foreach (var node in data) {
                bool isFirst = i == 0;
                bool isLast = i == data.Count - 1;
                if (isFirst) {
                    s.Append("{");
                } else {
                    s.Append(",");
                }
                string key = node.Key == '\0' ? "" : node.Key.ToString();

                s.Append($"\"{key}\":");

                if (node.Key == '\0') {
                    s.Append($"{(int)node.Value}");
                } else {
                    ToString(node.Value as SortedDictionary<char, object>, s);
                }

                if (isLast) {
                    s.Append("}");
                }

                i++;
            }
        }
    }
}