using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class EncoderDecoderCache : Cache {

        public Cache SelfAttentionCache { get; private set; }
        public Cache CrossAttentionCache { get; private set; }

        public override (FunctionalTensor Key, FunctionalTensor Value) this[int index] => throw new System.NotImplementedException();

        public EncoderDecoderCache(Cache selfAttentionCache, Cache crossAttentionCache) {
            SelfAttentionCache = selfAttentionCache;
            CrossAttentionCache = crossAttentionCache;
        }
        
        public override void Reset() {
            throw new System.NotImplementedException();
        }

        public override void Update(FunctionalTensor keyStates, FunctionalTensor valueStates, int layerIdx) {
            throw new System.NotImplementedException();
        }

        public override IEnumerator<(FunctionalTensor Key, FunctionalTensor Value)> GetEnumerator() {
            throw new System.NotImplementedException();
        }
    }
}