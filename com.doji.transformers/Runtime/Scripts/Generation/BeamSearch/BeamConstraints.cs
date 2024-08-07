using System.Collections.Generic;

namespace Doji.AI.Transformers {
    /// <summary>
    /// Abstract base class for all constraints that can be applied during generation.
    /// </summary>
    public abstract class Constraint { }
    public class DisjunctiveConstraint : Constraint {
        public DisjunctiveConstraint(List<List<int>> wordIds) { }
    }

    public class PhrasalConstraint : Constraint {
        public PhrasalConstraint(List<int> wordIds) { }
    }
}