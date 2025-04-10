using UnityEditor;
using UnityEngine;
using System.Text;
using System.Collections.Generic;
using UnityEngine.UIElements;
using System;

namespace Doji.AI.Transformers.Editor {

    [CustomEditor(typeof(TokenizerModelAsset))]
    public class TokenizerModelImporterEditor : UnityEditor.Editor {

        private byte[] data;
        private List<string> parsedStrings;
        private Vector2 scroll;

        public void OnEnable() {
            var asset = serializedObject.targetObject as TokenizerModelAsset;
            parsedStrings = ExtractAsciiStrings(asset.ModelData);
        }

        public override VisualElement CreateInspectorGUI() {
            var rootInspector = new VisualElement();

            var asset = target as TokenizerModelAsset;
            if (asset == null)
                return rootInspector;
            if (asset.ModelData == null)
                return rootInspector;

            rootInspector.Add(new Label($"Tokenizer Model Content:"));

            if (parsedStrings != null && parsedStrings.Count > 0) {
                var inputMenu = CreateFoldoutListView(parsedStrings, $"<b>Elements ({parsedStrings.Count})</b>");
                rootInspector.Add(inputMenu);
            } else {
                rootInspector.Add(new HelpBox("No ASCII strings found in the model file.", HelpBoxMessageType.Info));
            }

            return rootInspector;
        }

        Foldout CreateFoldoutListView(List<string> items, string name) {
            Func<VisualElement> makeItem = () => new Label();
            Action<VisualElement, int> bindItem = (e, i) => (e as Label).text = items[i];

            var listView = new ListView(items, 16, makeItem, bindItem);
            listView.showAlternatingRowBackgrounds = AlternatingRowBackground.All;
            listView.showBorder = true;
            listView.selectionType = SelectionType.Multiple;
            listView.style.flexGrow = 1;
            listView.horizontalScrollingEnabled = true;

            var inputMenu = new Foldout();
            inputMenu.text = name;
            inputMenu.style.maxHeight = 400;
            inputMenu.Add(listView);

            return inputMenu;
        }
        /*
        public override void OnInspectorGUI() {

            using var _ = new EditorGUI.DisabledScope(false);

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
        }*/

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
