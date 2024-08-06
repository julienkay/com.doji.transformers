using System;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Extends Ops class with not-yet implemented operators
    /// and some overloads for more convenience.
    /// </summary>
    public static class SentisUtils {

        /// <summary>
        /// Similar to torch.repeat() or numpy.tile()
        /// </summary>
        public static TensorFloat Repeat(this Ops ops, TensorFloat tensor, int repeats, int axis) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }

            if (repeats == 1) {
                return tensor;
            }

            int[] r = ArrayUtils.Full(tensor.shape.rank, 1);
            r[axis] = repeats;
            return ops.Tile(tensor, r);
        }

        /// <summary>
        /// Similar to torch.repeat() or numpy.tile()
        /// </summary>
        public static TensorInt Repeat(this Ops ops, TensorInt tensor, int repeats, int axis) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }

            if (repeats == 1) {
                return tensor;
            }

            int[] r = ArrayUtils.Full(tensor.shape.rank, 1);
            r[axis] = repeats;
            return ops.Tile(tensor, r);
        }


        /// <summary>
        /// Similar to torch.repeat_interleave() or numpy.repeat()
        /// </summary>
        public static TensorFloat RepeatInterleave(this Ops ops, TensorFloat tensor, int repeats, int dim) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }
            if (tensor.shape.rank > 1) {
                throw new ArgumentException($"RepeatInterleave not supported yet for tensors with rank > 1");
            }

            // implement repeat_interleave using repeat, reshape & transpose ops
            var repeat = ops.Repeat(tensor, repeats, dim);
            var flatShape = new TensorShape(repeat.shape.length);
            repeat.Reshape(flatShape);
            repeat.Reshape(new TensorShape(repeats, flatShape.length / repeats));
            var transpose = ops.Transpose(repeat, new int[] { 1, 0 });
            transpose.Reshape(flatShape);
            return transpose;
        }

        /// <summary>
        /// Similar to torch.repeat_interleave() or numpy.repeat()
        /// </summary>
        public static TensorInt RepeatInterleave(this Ops ops, TensorInt tensor, int repeats, int dim) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }
            if (tensor.shape.rank > 1) {
                throw new ArgumentException($"RepeatInterleave not supported yet for tensors with rank > 1");
            }

            // implement repeat_interleave using repeat, reshape & transpose ops
            var repeat = ops.Repeat(tensor, repeats, dim);
            var flatShape = new TensorShape(repeat.shape.length);
            repeat.Reshape(flatShape);
            repeat.Reshape(new TensorShape(repeats, flatShape.length / repeats));
            var transpose = ops.Transpose(repeat, new int[] { 1, 0 });
            transpose.Reshape(flatShape);
            return transpose;
        }
    }
}