using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Provides similar functionality & method names like python kwargs to simplify porting code from Python -> C#
    /// </summary>
    public class Kwargs : Dictionary<string, object>, ICollection, IDictionary {

        public Kwargs(IEnumerable<KeyValuePair<string, object>> collection) : base(collection) { }
        public Kwargs() : base() { }

        public object Get(string key, object defaultValue = null) {
            return this.GetValueOrDefault(key, defaultValue);
        }

        public T Get<T>(string key, T defaultValue = default) {
            if (TryGetValue(key, out object value)) {
                return (T)value;
            }
            return defaultValue;
        }

        public object Pop(string key, object defaultVal = default) {
            if (TryGetValue(key, out object val)) {
                Remove(key);
                return val;
            } else {
                return defaultVal;
            }
        }

        public Kwargs Where(Func<KeyValuePair<string, object>, bool> predicate) {
            return new Kwargs(this.AsEnumerable().Where(predicate));
        }
     }
}