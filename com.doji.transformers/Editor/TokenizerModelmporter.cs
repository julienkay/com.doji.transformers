using UnityEditor.AssetImporters;
using UnityEngine;
using System.IO;

namespace Doji.AI.Transformers.Editor {

    [ScriptedImporter(version: 1, ext: "model")]
    public class TokenizerModelImporter : ScriptedImporter {
        public override void OnImportAsset(AssetImportContext ctx) {
            byte[] fileData = File.ReadAllBytes(ctx.assetPath);
            TextAsset textAsset = new TextAsset(fileData);
            Debug.Log(textAsset.text);
            /*var imported = textAsset.bytes;
            for (int i = 0; i < fileData.Length; i++) {
                if (fileData[i] != imported[i]) {
                    Debug.LogError(i);
                }
            }*/
            textAsset.name = Path.GetFileNameWithoutExtension(ctx.assetPath);
            ctx.AddObjectToAsset("ModelTextAsset", textAsset);
            ctx.SetMainObject(textAsset);
        }
    }
}
