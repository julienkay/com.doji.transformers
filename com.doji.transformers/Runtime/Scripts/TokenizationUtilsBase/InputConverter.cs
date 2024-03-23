using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;

namespace Doji.AI.Transformers {

    public class InputConverter : JsonConverter {
        public override bool CanConvert(Type objectType) {
            return typeof(Input).IsAssignableFrom(objectType);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer) {
            JObject jsonObject = JObject.Load(reader);
            if (jsonObject["Type"] == null) {
                string legacyInput = jsonObject.Value<string>();
                return new SingleInput(legacyInput);
            } 
            string type = jsonObject["Type"].Value<string>();

            switch (type) {
                case nameof(SingleInput):
                    return new SingleInput(jsonObject["Value"].Value<string>());
                case nameof(BatchInput):
                    var batch = jsonObject["Value"];
                    return new BatchInput(batch.ToObject<List<string>>(serializer));
                case nameof(PretokenizedSingleInput):
                    var pretokenized = jsonObject["Value"];
                    return new PretokenizedSingleInput(pretokenized.ToObject<List<string>>(serializer));
                case nameof(PretokenizedBatchInput):
                    var pretokenizedBatch = jsonObject["Value"];
                    return new PretokenizedBatchInput(pretokenizedBatch.ToObject<List<List<string>>>(serializer));
                default:
                    throw new InvalidOperationException("Unknown Input type");
            }
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer) {
            JObject obj = new JObject();
            obj["Type"] = value.GetType().Name;
            if (value is SingleInput singleInput) {
                obj["Value"] = JToken.FromObject(singleInput.Text, serializer);
            } else if (value is BatchInput batchInput) {
                obj["Value"] = JToken.FromObject(batchInput.Sequence, serializer);
            } else if (value is PretokenizedSingleInput pretokenizedSingleInput) {
                obj["Value"] = JToken.FromObject(pretokenizedSingleInput.PretokenizedText, serializer);
            } else if (value is PretokenizedBatchInput pretokenizedBatchInput) {
                obj["Value"] = JToken.FromObject(pretokenizedBatchInput.Sequence, serializer);
            }
            obj.WriteTo(writer);
        }
        /*private InputType _currentObjectType;

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer) {
            var jobj = JObject.ReadFrom(reader);
            _currentObjectType = jobj["Type"].ToObject<InputType>();
            return base.ReadJson(jobj.CreateReader(), objectType, existingValue, serializer);
        }

        public override Input Create(Type objectType) {
            switch (_currentObjectType) {
                case InputType.SingleInput:
                    return new SingleInput();
                case InputType.ChildClass2:
                    return new Child2();
                default:
                    throw new NotImplementedException();
            }
        }*/

        /*public override void WriteJson(JsonWriter writer, Input value, JsonSerializer serializer) {
            if (value is SingleInput singleInput) {
                serializer.Serialize(writer, singleInput.Text);
            } else if (value is BatchInput batchInput) {
                serializer.Serialize(writer, batchInput.Sequence);
            } else if (value is PretokenizedSingleInput pretokenizedSingleInput) {
                serializer.Serialize(writer, pretokenizedSingleInput.PretokenizedText);
            } else if (value is PretokenizedBatchInput pretokenizedBatchInput) {
                serializer.Serialize(writer, pretokenizedBatchInput.Sequence);
            }
        }

        public override Input ReadJson(JsonReader reader, Type objectType, Input existingValue, bool hasExistingValue, JsonSerializer serializer) {
            JObject jsonObject = JObject.Load(reader);

            // Determine the type of Input to deserialize based on the properties present in the JSON object
            if (jsonObject.ContainsKey("Text")) {
                return new SingleInput(jsonObject.Value<string>("Text"));
            } else if (jsonObject.ContainsKey("PretokenizedText")) {
                return new PretokenizedSingleInput(jsonObject.Value<List<string>>("PretokenizedText"));
            } else if (jsonObject.ContainsKey("Sequence")) {

            }

            throw new JsonSerializationException($"Unable to deserialize {nameof(Input)}.");
        }*/
    }
}