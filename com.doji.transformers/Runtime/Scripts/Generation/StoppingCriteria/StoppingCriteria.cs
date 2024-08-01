namespace Doji.AI.Transformers {
    /// <summary>
    /// This class can be used to stop generation whenever specific string sequences are generated.
    /// It preprocesses the strings together with the tokenizer vocab to find positions where tokens
    /// can validly complete the stop strings.
    /// </summary>
    public abstract class StoppingCriteria { }
}