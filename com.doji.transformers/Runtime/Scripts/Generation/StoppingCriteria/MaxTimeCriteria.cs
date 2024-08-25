using System;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    /// <summary>
    /// This class can be used to stop generation whenever the full generation exceeds some amount of time.
    /// By default, the time will start being counted when you initialize this function.
    /// You can override this by passing an <see cref="InitialTimestamp"/>.
    /// </summary>
    public class MaxTimeCriteria : StoppingCriteria {

        /// <summary>
        /// The maximum allowed time in seconds for the generation.
        /// </summary>
        public float MaxTime { get; }

        /// <summary>
        /// The start of the generation allowed time.
        /// </summary>
        public DateTime InitialTimestamp { get; }

        public MaxTimeCriteria(float maxTime, DateTime? initialTimestamp = null) {
            MaxTime = maxTime;
            InitialTimestamp = initialTimestamp ?? DateTime.UtcNow;
        }

        public override Tensor<int> Apply(Tensor<int> inputIds, Tensor<float> scores) {
            bool isDone = (DateTime.UtcNow - InitialTimestamp).TotalSeconds > MaxTime;
            return Ops.Full(inputIds.shape[0], isDone);
        }
    }
}