using UnityEditor.AssetImporters;
using UnityEngine;
using System.IO;

namespace Doji.AI.Transformers.Editor {

    [ScriptedImporter(version: 1, ext: "model")]
    public class TokenizerModelImporter : ScriptedImporter {
        public override void OnImportAsset(AssetImportContext ctx) {
            byte[] bytes = File.ReadAllBytes(ctx.assetPath);
            var asset = ScriptableObject.CreateInstance<TokenizerModelAsset>();
            asset.SetBytes(bytes);
            ctx.AddObjectToAsset("TokenizerModel", asset);
            ctx.SetMainObject(asset);
        }
    }
}
