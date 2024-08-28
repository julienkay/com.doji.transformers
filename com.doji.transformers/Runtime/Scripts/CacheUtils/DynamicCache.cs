using System;
using System.Collections.Generic;
using Unity.Sentis;
using static FunctionalUtils;

namespace Doji.AI.Transformers {
    
    public class DynamicCache : Cache {

        public List<FunctionalTensor> KeyCache { get; set; }
        public List<FunctionalTensor> ValueCache { get; set; }

        private int _seenTokens;

        public DynamicCache() : base() {
            KeyCache = new List<FunctionalTensor>();
            ValueCache = new List<FunctionalTensor>();
        }

        public override (FunctionalTensor Key, FunctionalTensor Value) this[int index] {
            get {
                if (index < 0 || index >= Math.Min(KeyCache.Count, ValueCache.Count)) {
                    throw new IndexOutOfRangeException($"Index ({index} is outside the bounds of the cached values (0-{KeyCache.Count - 1}).");
                }
                return (KeyCache[index], ValueCache[index]);
            }
        }

        public override IEnumerator<(FunctionalTensor Key, FunctionalTensor Value)> GetEnumerator() {
            // Ensure both lists have the same number of elements
            int count = Math.Min(KeyCache.Count, ValueCache.Count);

            for (int i = 0; i < count; i++) {
                yield return (KeyCache[i], ValueCache[i]);
            }
        }

        public override int GetSeqLength(int? layerIdx = 0) {
            if (KeyCache.Count <= layerIdx) {
                return 0;
            }
            return KeyCache[layerIdx.Value].shape()[-2];
        }

        public override void Update(FunctionalTensor keyStates, FunctionalTensor valueStates, int layerIdx) {
            // Update the number of seen tokens
            if (layerIdx == 0) {
                _seenTokens += keyStates.shape()[-2];
            }

            // Update the cache
            if (KeyCache.Count <= layerIdx) {
                KeyCache.Add(keyStates);
                ValueCache.Add(valueStates);
            } else {
                KeyCache[layerIdx] = Concat(KeyCache[layerIdx], keyStates, dim: -2);
                ValueCache[layerIdx] = Concat(ValueCache[layerIdx], valueStates, dim: -2);
            }
        }
    }
}