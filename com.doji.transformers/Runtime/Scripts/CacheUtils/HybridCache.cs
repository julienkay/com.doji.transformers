using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class HybridCache : Cache {
        public override (Tensor Key, Tensor Value) this[int index] => throw new System.NotImplementedException();

        public override IEnumerator<(Tensor Key, Tensor Value)> GetEnumerator() {
            throw new System.NotImplementedException();
        }

        public override void Reset() {
            throw new System.NotImplementedException();
        }

        public override void Update(Tensor keyStates, Tensor valueStates, int layerIdx) {
            throw new System.NotImplementedException();
        }
    }
}