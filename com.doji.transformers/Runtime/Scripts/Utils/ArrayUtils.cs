using System;

namespace Doji.AI.Transformers {

    public static class ArrayUtils {

        public static T[] Full<T>(int n, T x) {
            if (n < 0) {
                throw new ArgumentException("Value of n must be non-negative.");
            }

            T[] array = new T[n];

            for (int i = 0; i < n; i++) {
                array[i] = x;
            }

            return array;
        }
    }
}