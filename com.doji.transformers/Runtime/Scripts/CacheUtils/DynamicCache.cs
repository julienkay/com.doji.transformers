using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    
    public class DynamicCache : Cache {

        public List<Tensor> KeyCache { get; set; }
        public List<Tensor> ValueCache { get; set; }

        private int _seenTokens;

        public DynamicCache() : base() {
            KeyCache = new List<Tensor>();
            ValueCache = new List<Tensor>();
        }

        public override (Tensor Key, Tensor Value) this[int index] {
            get {
                if (index < 0 || index >= Math.Min(KeyCache.Count, ValueCache.Count)) {
                    throw new IndexOutOfRangeException($"Index ({index} is outside the bounds of the cached values (0-{KeyCache.Count - 1}).");
                }
                return (KeyCache[index], ValueCache[index]);
            }
        }

        public override IEnumerator<(Tensor Key, Tensor Value)> GetEnumerator() {
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
            return KeyCache[layerIdx.Value].shape[-2];
        }

        public override void Update(Tensor keyStates, Tensor valueStates, int layerIdx) {
            // Update the number of seen tokens
            if (layerIdx == 0) {
                _seenTokens += keyStates.shape[-2];
            }

            // Update the cache
            if (KeyCache.Count <= layerIdx) {
                KeyCache.Add(keyStates);
                ValueCache.Add(valueStates);
            } else {
                KeyCache[layerIdx] = Ops.Cat(KeyCache[layerIdx], keyStates, axis: -2);
                ValueCache[layerIdx] = Ops.Cat(ValueCache[layerIdx], valueStates, axis: -2);
            }
        }
    }
}