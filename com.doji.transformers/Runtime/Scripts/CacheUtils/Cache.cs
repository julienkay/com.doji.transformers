using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public abstract class Cache : IEnumerable<(FunctionalTensor Key, FunctionalTensor Value)> {

        internal FunctionalGraph Ops { get; set; }
        public int? MaxBatchSize { get; private set; }
        public int? MaxCacheLen { get; private set; }

        public virtual void Reset() { }

        /// <summary>
        /// Returns the sequence length of the cached states."
        /// </summary>
        public virtual int GetSeqLength(int? layerIdx = 0) {
            throw new NotImplementedException($"Make sure to implement {nameof(GetSeqLength)} in subclass '{GetType()}'.");
        }

        /// <summary>
        /// Updates the cache with the new <paramref name="keyStates"/> and <paramref name="valueStates"/> for the layer <paramref name="layerIdx"/>.
        /// </summary>
        public abstract void Update(FunctionalTensor keyStates, FunctionalTensor valueStates, int layerIdx);

        public abstract (FunctionalTensor Key, FunctionalTensor Value) this[int index] { get; }

        public abstract IEnumerator<(FunctionalTensor Key, FunctionalTensor Value)> GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() {
            return GetEnumerator();
        }
    }
}