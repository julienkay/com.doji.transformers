using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Trie in Python. Creates a Trie out of a list of words.
    /// The trie is used to split on `added_tokens` in one pass
    /// Loose reference https://en.wikipedia.org/wiki/Trie
    /// </summary>
    internal class Trie {

        public readonly HashSet<string> Tokens = new HashSet<string>();
        public Dictionary<char, object> Data = new Dictionary<char, object>();

        public void Add(string word) {
            if (string.IsNullOrEmpty(word)) {
                // Prevent empty string
                return;
            }

            Tokens.Add(word);
            var node = Data;

            foreach (var ch in word) {
                if (!node.ContainsKey(ch)) {
                    node[ch] = new Dictionary<char, object>();
                }
                node = (Dictionary<char, object>)node[ch];
            }

            // Use '\0' to represent the empty string
            node['\0'] = 1;
        }

        public List<string> Split(string text) {
            var states = new Dictionary<int, object>();
            var offsets = new List<int> { 0 };
            var toRemove = new List<int>();
            int skip = 0;
            int end = text.Length;

            for (int current = 0; current < text.Length; current++) {
                if (skip > 0 && current < skip) {
                    continue;
                }
                char currentChar = text[current];

                toRemove.Clear();
                var reset = false;

                var statesCopy = states.ToArray();
                foreach (var entry in statesCopy) {
                    int start = entry.Key;
                    var triePointer = (Dictionary<char, object>)entry.Value;

                    if (triePointer.ContainsKey('\0')) {

                        var statesCopy2 = states.ToArray();
                        foreach (var lookEntry in statesCopy2) {
                            int lookStart = lookEntry.Key;
                            var lookTriePointer = (Dictionary<char, object>)lookEntry.Value;

                            int lookaheadIndex;

                            if (lookStart == start) {
                                lookaheadIndex = current;
                                end = current;
                            } else {
                                lookaheadIndex = current + 1;
                                end = current + 1;
                            }

                            char nextChar = lookaheadIndex < text.Length ? text[lookaheadIndex] : '\0';

                            if (lookTriePointer.ContainsKey('\0')) {
                                start = lookStart;
                                end = lookaheadIndex;
                                skip = lookaheadIndex;
                            }

                            while (lookTriePointer.ContainsKey(nextChar)) {
                                lookTriePointer = (Dictionary<char, object>)lookTriePointer[nextChar];
                                lookaheadIndex++;

                                if (lookTriePointer.ContainsKey('\0')) {
                                    start = lookStart;
                                    end = lookaheadIndex;
                                    skip = lookaheadIndex;
                                }

                                if (lookaheadIndex == text.Length) {
                                    break;
                                }

                                nextChar = text[lookaheadIndex];
                            }
                        }

                        offsets.Add(start);
                        offsets.Add(end);
                        reset = true;
                        break;
                    } else if (triePointer.ContainsKey(currentChar)) {
                        triePointer = (Dictionary<char, object>)triePointer[currentChar];
                        states[start] = triePointer;
                    } else {
                        toRemove.Add(start);
                    }
                }

                if (reset) {
                    states.Clear();
                } else {
                    foreach (var start in toRemove) {
                        states.Remove(start);
                    }
                }

                if (current >= skip && Data.ContainsKey(currentChar)) {
                    states[current] = Data[currentChar];
                }
            }

            foreach (var entry in states) {
                int start = entry.Key;
                var triePointer = (Dictionary<char, object>)entry.Value;

                if (triePointer.ContainsKey('\0')) {
                    end = text.Length;
                    offsets.Add(start);
                    offsets.Add(end);
                    break;
                }
            }

            return CutText(text, offsets);
        }
        private List<string> CutText(string text, List<int> offsets) {
            // We have all the offsets now, we just need to do the actual splitting.
            // We need to eventually add the first part of the string and the eventual
            // last part.
            offsets.Add(text.Length);
            List<string> tokens = new List<string>();
            int start = 0;

            foreach (int end in offsets) {
                if (start > end) {
                    Log.Error("There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it anyway.");
                    continue;
                } else if (start == end) {
                    // This might happen if there's a match at index 0
                    // we're also preventing zero-width cuts in case of two
                    // consecutive matches
                    continue;
                }

                tokens.Add(text.Substring(start, end - start));
                start = end;
            }

            return tokens;
        }

        public override string ToString() {
            StringBuilder s = new StringBuilder();
            ToString(Data, s);
            return s.ToString();
        }

        private void ToString(Dictionary<char, object> data, StringBuilder s) {
            int i = 0;
            foreach (var node in data) {
                bool isFirst = i == 0;
                bool isLast = i == data.Count - 1;
                if (isFirst) {
                    s.Append("{");
                } else {
                    s.Append(", ");
                }
                string key = node.Key == '\0' ? "" : node.Key.ToString();

                s.Append($"\"{key}\": ");

                if (node.Key == '\0') {
                    s.Append($"{(int)node.Value}");
                } else {
                    ToString(node.Value as Dictionary<char, object>, s);
                }

                if (isLast) {
                    s.Append("}");
                }

                i++;
            }
        }
    }
}