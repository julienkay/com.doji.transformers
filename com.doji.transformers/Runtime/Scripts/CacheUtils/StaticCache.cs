using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI.Transformers {

    public class StaticCache : Cache {
        public override (FunctionalTensor Key, FunctionalTensor Value) this[int index] => throw new System.NotImplementedException();

        public override IEnumerator<(FunctionalTensor Key, FunctionalTensor Value)> GetEnumerator() {
            throw new System.NotImplementedException();
        }

        public override void Reset() {
            throw new System.NotImplementedException();
        }

        public override void Update(FunctionalTensor keyStates, FunctionalTensor valueStates, int layerIdx) {
            throw new System.NotImplementedException();
        }
    }
}