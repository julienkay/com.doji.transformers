using Newtonsoft.Json;
using System.Collections.Generic;

namespace Doji.AI.Transformers {

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
}