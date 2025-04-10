using UnityEditor.AssetImporters;
using UnityEditor;
using UnityEngine;
using System.IO;
using System.Text;
using System.Collections.Generic;

namespace Doji.AI.Transformers.Editor {

    [CustomEditor(typeof(TokenizerModelImporter))]
    public class TokenizerModelImporterEditor : ScriptedImporterEditor {

        private byte[] data;
        private List<string> parsedStrings;
        private Vector2 scroll;

        public override void OnEnable() {
            base.OnEnable();
            var importer = serializedObject.targetObject as TokenizerModelImporter;
            data = File.ReadAllBytes(importer.assetPath);
            parsedStrings = ExtractAsciiStrings(data);
        }

        public override void OnInspectorGUI() {
            var importer = serializedObject.targetObject as TokenizerModelImporter;

            EditorGUILayout.LabelField("Tokenizer Model Content:", EditorStyles.boldLabel);

            if (parsedStrings != null && parsedStrings.Count > 0) {
                scroll = EditorGUILayout.BeginScrollView(scroll, GUILayout.Height(200));
                foreach (var str in parsedStrings) {
                    EditorGUILayout.LabelField(str);
                }
                EditorGUILayout.EndScrollView();
            } else {
                EditorGUILayout.HelpBox("No ASCII strings found in the model file.", MessageType.Info);
            }

            base.ApplyRevertGUI();
        }

        // Extract readable ASCII strings (length >= 2) from byte stream
        // (technically tokenizer.model files are protobuf serialized?)
        private List<string> ExtractAsciiStrings(byte[] bytes, int minLength = 2) {
            List<string> strings = new List<string>();
            StringBuilder current = new StringBuilder();

            foreach (byte b in bytes) {
                // Accept printable ASCII range (32–126) plus newline
                if (b >= 32 && b <= 126) {
                    current.Append((char)b);
                } else {
                    if (current.Length >= minLength) {
                        strings.Add(current.ToString());
                    }
                    current.Clear();
                }
            }

            // Catch anything left over
            if (current.Length >= minLength) {
                strings.Add(current.ToString());
            }

            return strings;
        }
    }
}
