using System.Collections.Generic;
using System.Collections.Specialized;
using System.Collections;
using System.Linq;

namespace Doji.AI.Transformers {

    /// <summary>
    /// A generic wrapper around <see cref="OrderedDictionary"/>.
    /// </summary>
    /// <typeparam name="TKey"></typeparam>
    /// <typeparam name="TValue"></typeparam>
    internal class OrderedDictionary<TKey, TValue> : ICollection<KeyValuePair<TKey, TValue>>, IEnumerable<KeyValuePair<TKey, TValue>>, IEnumerable, IDictionary<TKey, TValue>, IReadOnlyCollection<KeyValuePair<TKey, TValue>> {
        private readonly OrderedDictionary _dictionary = new OrderedDictionary();

        public ICollection<TKey> Keys => _dictionary.Keys.Cast<TKey>().ToList();

        public ICollection<TValue> Values => _dictionary.Values.Cast<TValue>().ToList();

        public int Count => _dictionary.Count;

        public bool IsReadOnly => false;

        public TValue this[TKey key] {
            get => (TValue)_dictionary[key];
            set => _dictionary[key] = value;
        }

        public void Add(TKey key, TValue value) {
            _dictionary.Add(key, value);
        }

        public bool ContainsKey(TKey key) {
            return _dictionary.Contains(key);
        }

        public bool Remove(TKey key) {
            if (!ContainsKey(key)) return false;

            _dictionary.Remove(key);
            return true;
        }

        public bool TryGetValue(TKey key, out TValue value) {
            if (ContainsKey(key)) {
                value = this[key];
                return true;
            }

            value = default;
            return false;
        }

        public void Add(KeyValuePair<TKey, TValue> item) {
            _dictionary.Add(item.Key, item.Value);
        }

        public void Clear() {
            _dictionary.Clear();
        }

        public bool Contains(KeyValuePair<TKey, TValue> item) {
            return _dictionary.Contains(item.Key) && EqualityComparer<TValue>.Default.Equals(this[item.Key], item.Value);
        }

        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex) {
            foreach (DictionaryEntry entry in _dictionary) {
                array[arrayIndex++] = new KeyValuePair<TKey, TValue>((TKey)entry.Key, (TValue)entry.Value);
            }
        }

        public bool Remove(KeyValuePair<TKey, TValue> item) {
            if (!Contains(item)) return false;

            _dictionary.Remove(item.Key);
            return true;
        }

        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator() {
            foreach (DictionaryEntry entry in _dictionary) {
                yield return new KeyValuePair<TKey, TValue>((TKey)entry.Key, (TValue)entry.Value);
            }
        }

        IEnumerator IEnumerable.GetEnumerator() {
            return GetEnumerator();
        }
    }
}