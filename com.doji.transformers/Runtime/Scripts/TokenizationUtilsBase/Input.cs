using Newtonsoft.Json;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Doji.AI.Transformers {

    /// <summary>
    /// Represents input for a tokenizer with explicit types.
    /// Inputs can either be a single text, a batch/sequence of text,
    /// pretokenized text, or a sequence of pretokenized texts.
    /// </summary>
    /// <remarks>
    /// string and List[string[string]] have implicit conversions in the
    /// base class because they are not ambiguous. For the others, when
    /// calling methods like <see cref="PreTrainedTokenizerBase.Encode{T}"/>
    /// you can use disambiguate between sequences of text and pretokenized text
    /// by using the generic version and specifying the type like so;
    /// <code>
    /// tokenizer.Encode<BatchInput>(myList);
    /// tokenizer.Encode<PretokenizedInput>(myList);
    /// </code>
    /// </remarks>
    [JsonConverter(typeof(InputConverter))]
    public abstract class Input {

        public static implicit operator Input(string text) {
            if (text == null) { return null; }
            return new SingleInput(text);
        }

        public static implicit operator Input(List<List<string>> pretokenizedSequences) {
            if (pretokenizedSequences == null) { return null; }
            return new PretokenizedBatchInput(pretokenizedSequences);
        }

        /// <summary>
        /// Does the input represent a sequence/batch?
        /// </summary>
        public bool IsBatch() {
            return this is BatchInput || this is PretokenizedBatchInput;
        }

        /// <summary>
        /// Is the input already pretokenized?
        /// </summary>
        public bool IsPretokenized() {
            return this is BatchInput || this is PretokenizedBatchInput;
        }

        public abstract override string ToString();
    }

    public interface IBatchInput {
        public IList Sequence { get; set; }
        public int BatchSize {
            get {
                return this.Sequence.Count;
            }
        }
    }

    /// <summary>
    /// Base class for tokenizer inputs that represent text or sequences/batches of text.
    /// </summary>
    public abstract class TextInput : Input {

        public static implicit operator TextInput(List<string> sequence) {
            if (sequence == null) { return null; }
            return new BatchInput(sequence);
        }

        public static explicit operator string(TextInput textInput) {
            if (textInput is SingleInput input) {
                return (string)input;
            } else {
                throw new System.InvalidCastException($"The specified cast from {textInput.GetType()} to string is not valid.");
            }
        }
    }

    /// <summary>
    /// Represents a single text input for the tokenizer.
    /// </summary>
    public class SingleInput : TextInput {

        /// <summary>
        /// The text.
        /// </summary>
        public string Text { get; set; }

        public static explicit operator string(SingleInput input) => input.Text;

        public SingleInput(string text) {
            Text = text;
        }

        public override string ToString() {
            return Text;
        }
    }

    /// <summary>
    /// Represents a sequence/batch of inputs for the tokenizer.
    /// </summary>
    public class BatchInput : TextInput, IBatchInput {

        /// <summary>
        /// The sequence of text.
        /// </summary>
        public IList Sequence { get; set; }

        public static implicit operator BatchInput(List<string> sequence) => new BatchInput(sequence);
        public static explicit operator List<string>(BatchInput input) => (List<string>)input.Sequence;

        public BatchInput(List<string> sequence) {
            Sequence = sequence;
        }

        public override string ToString() {
            StringBuilder sb = new StringBuilder();
            foreach(string s in Sequence) {
                sb.AppendLine(s);
            }
            return sb.ToString();
        }
    }

    /// <summary>
    /// Base class for inputs that represent pretokenized input for a tokenizer.
    /// </summary>
    public abstract class PretokenizedInput : Input {

        public static implicit operator PretokenizedInput(List<string> pretokenizedText) {
            if (pretokenizedText == null) { return null; }
            return new PretokenizedSingleInput(pretokenizedText);
        }
    }

    public class PretokenizedSingleInput : PretokenizedInput {

        /// <summary>
        /// The tokens.
        /// </summary>
        public List<string> PretokenizedText { get; set; }

        public static explicit operator List<string>(PretokenizedSingleInput input) => input.PretokenizedText;

        public PretokenizedSingleInput(List<string> pretokenizedText) {
            PretokenizedText = pretokenizedText;
        }

        public override string ToString() {
            StringBuilder sb = new StringBuilder();
            foreach (string s in PretokenizedText) {
                sb.AppendLine(s);
            }
            return sb.ToString();
        }
    }

    public class PretokenizedBatchInput : PretokenizedInput, IBatchInput {

        /// <summary>
        /// The sequence of tokens for each input.
        /// </summary>
        public IList Sequence { get; set; }

        public static explicit operator List<List<string>>(PretokenizedBatchInput input) => (List<List<string>>)input.Sequence;

        public PretokenizedBatchInput(List<List<string>> pretokenizedSequences) {
            Sequence = pretokenizedSequences;
        }

        public override string ToString() {
            StringBuilder sb = new StringBuilder();
            foreach (List<string> batch in Sequence) {
                foreach (string token in batch) {
                    sb.AppendLine(token);
                }
            }
            return sb.ToString();
        }
    }
}