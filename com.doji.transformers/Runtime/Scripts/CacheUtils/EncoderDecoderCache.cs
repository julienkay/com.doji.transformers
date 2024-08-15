using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class EncoderDecoderCache : Cache {

        public Cache SelfAttentionCache { get; private set; }
        public Cache CrossAttentionCache { get; private set; }

        public override (Tensor Key, Tensor Value) this[int index] => throw new System.NotImplementedException();

        public EncoderDecoderCache(Cache selfAttentionCache, Cache crossAttentionCache) {
            SelfAttentionCache = selfAttentionCache;
            CrossAttentionCache = crossAttentionCache;
        }
        
        public override void Reset() {
            throw new System.NotImplementedException();
        }

        public override void Update(Tensor keyStates, Tensor valueStates, int layerIdx) {
            throw new System.NotImplementedException();
        }

        public override IEnumerator<(Tensor Key, Tensor Value)> GetEnumerator() {
            throw new System.NotImplementedException();
        }
    }
}