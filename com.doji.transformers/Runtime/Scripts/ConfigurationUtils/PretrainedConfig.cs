using Newtonsoft.Json;
using System.Collections.Generic;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Base class for all configuration classes. Handles a few parameters common to all models' configurations
    /// Methods for loading configurations.
    /// </summary>
    public class PretrainedConfig {

        [JsonProperty("_name_or_path")]
        public string NameOrPath { get; set; }

        [JsonProperty("architectures")]
        public List<string> Architectures { get; set; }
    }
}