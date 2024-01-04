using System;
using System.Collections.Generic;

namespace Doji.AI.Transformers {
    public class BatchEncoding : Dictionary<string, object> {

        public static implicit operator BatchEncoding(List<int> list) {
            throw new NotImplementedException();
        }
    }
}