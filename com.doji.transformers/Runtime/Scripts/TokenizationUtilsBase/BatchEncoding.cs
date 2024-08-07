using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Represents the encoded output for a batch of text inputs.
    /// </summary>
    public class BatchEncoding : Encoding {

        public BatchEncoding() : base() { }
        public BatchEncoding(Dictionary<string, object> dict) : base(dict) { }

        public override IEnumerable<int> InputIds {
            get {
                if (!TryGetValue("input_ids", out var inputIds)) {
                    return null;
                }
                List<int> flattenedList = new List<int>();
                foreach (var innerList in inputIds as List<List<int>>) {
                    flattenedList.AddRange(innerList);
                }
                return flattenedList;
            }
        }

        public override IEnumerable<int> AttentionMask {
            get {
                if (!TryGetValue("attention_mask", out var inputIds)) {
                    return null;
                }
                List<int> flattenedList = new List<int>();
                foreach (var innerList in inputIds as List<List<int>>) {
                    flattenedList.AddRange(innerList);
                }
                return flattenedList;
            }
        }

        public override IEnumerable<int> TokenTypeIds {
            get {
                if (!TryGetValue("token_type_ids", out var inputIds)) {
                    return null;
                }
                List<int> flattenedList = new List<int>();
                foreach (var innerList in inputIds as List<List<int>>) {
                    flattenedList.AddRange(innerList);
                }
                return flattenedList;
            }
        }

        /// <summary>
        /// Appends all values from <paramref name="dict"/> to this dictionary
        /// which turns this 
        /// </summary>
        /// <param name="dict"></param>
        public void Append(Encoding dict) {
            foreach (var kvp in dict) {
                if (!ContainsKey(kvp.Key)) {
                    this[kvp.Key] = new List<List<int>>();
                }
                (this[kvp.Key] as List<List<int>>).Add(kvp.Value as List<int>);
            }
        }
    }
}