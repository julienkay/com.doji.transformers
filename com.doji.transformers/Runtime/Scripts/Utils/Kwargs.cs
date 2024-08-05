using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Provides similar functionality & method names like python kwargs to simplify porting code from Python -> C#
    /// </summary>
    public class Kwargs : Dictionary<string, object>, ICollection, IDictionary {

        public Kwargs(IDictionary<string, object> dictionary) : base(dictionary) { }
        public Kwargs() : base() { }

        public object Get(string key, object defaultValue = null) {
            if (TryGetValue(key, out object value)) {
                return value;
            }
            return defaultValue;
        }

        public T Get<T>(string key, T defaultValue = default) {
            if (TryGetValue(key, out object value)) {
                return (T)value;
            }
            return (T)defaultValue;
        }

        public object Pop(string key, object defaultVal = default) {
            return Pop(key, defaultVal);
        }

        public Kwargs Where(Func<KeyValuePair<string, object>, bool> predicate) {
            var filtered = Where(predicate);
            return new Kwargs(filtered.ToDictionary(kvp => kvp.Key, kvp => kvp.Value));
        }
    }
}