using UnityEngine;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A custom asset type that contains the raw data from tokenizer.model files.
    /// This is needed to allow to have tokenizer models in Resources folders
    /// (not actually deserializing the protocol buffer) and get them to be included in a build
    /// for passing to the tokenizer implementation via path.
    /// </summary>
    public class TokenizerModelAsset : ScriptableObject {
        [SerializeField, HideInInspector]
        private byte[] modelData;

        public byte[] ModelData => modelData;

        public void SetBytes(byte[] data) {
            modelData = data;
        }
    }
}
