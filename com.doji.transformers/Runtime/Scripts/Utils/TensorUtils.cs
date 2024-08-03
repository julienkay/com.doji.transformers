using System;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public static class TensorUtils {
        public static Tensor Ones(TensorShape shape, DataType type) {
            switch (type) {
                case DataType.Float:
                    return new TensorFloat(shape, OnesF(shape.length));
                case DataType.Int:
                    return new TensorInt(shape, OnesI(shape.length));
                case DataType.Short:
                    throw new NotImplementedException();
                case DataType.Byte:
                    throw new NotImplementedException();
                default:
                    throw new ArgumentException($"Invalid data type '{type}'");
            }
        }

        public static T Ones<T>(TensorShape shape) where T : Tensor {
            switch (typeof(T)) {
                case Type floatType when floatType == typeof(TensorFloat):
                    return Ones(shape, DataType.Float) as T;
                case Type intType when intType == typeof(TensorInt):
                    return Ones(shape, DataType.Int) as T;
                case Type shortType when shortType == typeof(TensorShort):
                case Type byteType when byteType == typeof(TensorByte):
                    throw new NotImplementedException();
                default:
                    throw new ArgumentException($"Invalid data type '{typeof(T)}'");
            }
        }

        private static int[] OnesI(int num) {
            int[] result = new int[num];
            for (int i = 0; i < num; i++) {
                result[i] = 1;
            }
            return result;
        }

        private static float[] OnesF(int num) {
            float[] result = new float[num];
            for (int i = 0; i < num; i++) {
                result[i] = 1;
            }
            return result;
        }
    }
}