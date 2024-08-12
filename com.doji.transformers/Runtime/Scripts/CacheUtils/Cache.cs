using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {
    public abstract class Cache {
        public int? MaxBatchSize { get; private set; }
        public int? MaxCacheLen { get; private set; }

        public virtual void Reset() { }
        /// <summary>
        /// Returns the sequence length of the cached states."
        /// </summary>
        public virtual int GetSeqLength(int? layerIdx = 0) {
            throw new NotImplementedException($"Make sure to implement {nameof(GetSeqLength)} in subclass '{GetType()}'.");
        }
    }
    /// <summary>
    /// Base class for cache configs
    /// </summary>
    public abstract class CacheConfig : Dictionary<string, object> { }
    /// <summary>
    /// Configuration class for quantized cache settings.
    /// </summary>
    public class QuantizedCacheConfig : CacheConfig {

        [JsonProperty("backend")]
        public string Backend { get; set; }
        
        [JsonProperty("nBits")]
        public int? NBits { get; set; }

        [JsonProperty("axisKey")]
        public int? AxisKey { get; set; }

        [JsonProperty("axisValue")]
        public int? AxisValue { get; set; }

        [JsonProperty("qGroupSize")]
        public int? QGroupSize { get; set; }

        [JsonProperty("residualLength")]
        public int? ResidualLength { get; set; }

        public QuantizedCacheConfig(
            string backend = "quanto",
            int? nBits = 4,
            int? axisKey = 0,
            int? axisValue = 0,
            int? qGroupSize = 64,
            int? residualLength = 128)
        {
            Backend = backend;
            NBits = nBits;
            AxisKey = axisKey;
            AxisValue = axisValue;
            QGroupSize = qGroupSize;
            ResidualLength = residualLength;
        }
    }
    public class DynamicCache : Cache {

        public List<Tensor> KeyCache { get; set; }

        public DynamicCache() : base() {
            KeyCache = new List<Tensor>();
        }

        public override int GetSeqLength(int? layerIdx = 0) {
            if (KeyCache.Count <= layerIdx) {
                return 0;
            }
            return KeyCache[layerIdx.Value].shape[-2];
        }
    }
    public class OffloadedCache : DynamicCache { }
    public class QuantizedCache : DynamicCache { }
    public class QuantoQuantizedCache : QuantizedCache { }
    public class HQQQuantizedCache : QuantizedCache { }
    public class SinkCache : Cache { }
    public class StaticCache : Cache {
        public override void Reset() {
            throw new NotImplementedException();
        }
    }
    public class SlidingWindowCache : StaticCache {
        public override void Reset() {
            throw new NotImplementedException();
        }
    }
    public class EncoderDecoderCache : Cache {
        public Cache SelfAttentionCache { get; private set; }
        public Cache CrossAttentionCache { get; private set; }
        public EncoderDecoderCache(Cache selfAttentionCache, Cache crossAttentionCache) {
            SelfAttentionCache = selfAttentionCache;
            CrossAttentionCache = crossAttentionCache;
        }
        public override void Reset() {
            throw new NotImplementedException();
        }
    }
    public class HybridCache : Cache {
        public override void Reset() {
            throw new NotImplementedException();
        }
    }
    public class MambaCache : Cache {
        public override void Reset() {
            throw new NotImplementedException();
        }
    }
}