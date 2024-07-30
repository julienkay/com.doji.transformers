using System.Collections;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI.Transformers {
    public abstract partial class PretrainedModel {

        /// <summary>
        /// Generates sequences of token ids for models with a language modeling head.
        /// </summary>
        public void Generate(
            TensorInt inputs,
            GenerationConfig config)
        {

        }
    }
}