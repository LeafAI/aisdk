use crate::error::Error;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};

/// Configuration options for OpenAI API requests.
#[derive(Debug, Default, Clone, Serialize, Deserialize, Builder)]
#[builder(setter(into), build_fn(error = "Error"))]
pub(crate) struct OpenAILanguageModelOptions {
    pub(crate) model: String,
    #[builder(default)]
    pub(crate) input: Option<Input>, // open ai requires input to be set
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub(crate) text: Option<TextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub(crate) reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub(crate) max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub(crate) stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub(crate) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub(crate) tools: Option<Vec<ToolParams>>,
    #[serde(skip)]
    #[builder(default)]
    pub(crate) extra_body: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(skip)]
    #[builder(default)]
    pub(crate) extra_headers: Option<std::collections::HashMap<String, String>>,
}

/// Response structure from the OpenAI API.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIResponse {
    /// Conversation parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationParam>,
    /// Timestamp of creation.
    pub created_at: Option<f64>,
    /// Error information if present.
    pub error: Option<OpenAIErrorByCode>,
    /// Unique identifier.
    pub id: Option<String>,
    /// Details for incomplete responses.
    pub incomplete_details: Option<IncompleteDetails>,
    /// Maximum output tokens.
    pub max_output_tokens: Option<u32>,
    /// Maximum tool calls.
    pub max_tool_calls: Option<u32>,
    /// Model used.
    pub model: Option<String>,
    /// Output messages.
    pub output: Option<Vec<MessageItem>>,
    /// Whether parallel tool calls are enabled.
    pub parallel_tool_calls: Option<bool>,
    /// Previous response ID.
    pub previous_response_id: Option<String>,
    /// Reasoning configuration.
    pub reasoning: Option<ReasoningConfig>,
    /// Text configuration.
    pub text: Option<TextConfig>,
    /// Usage statistics.
    pub usage: Option<ResponseUsage>,
}

impl OpenAILanguageModelOptions {
    pub(crate) fn builder() -> OpenAILanguageModelOptionsBuilder {
        OpenAILanguageModelOptionsBuilder::default()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
#[serde(tag = "type")]
/// Events emitted during streaming from OpenAI.
pub(crate) enum OpenAiStreamEvent {
    /// Emitted when the model response is complete.
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        #[serde(default)]
        sequence_number: Option<u64>,
        response: OpenAIResponse,
    },
    /// Emitted when an output item is added to the response.
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        #[serde(default)]
        sequence_number: Option<u64>,
        output_index: u32,
        item: MessageItem,
    },
    /// Emitted when an output item is finalized.
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        #[serde(default)]
        sequence_number: Option<u64>,
        output_index: u32,
        item: MessageItem,
    },
    /// Emitted when function-call arguments stream a delta.
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        #[serde(default)]
        sequence_number: Option<u64>,
        item_id: String,
        output_index: u32,
        delta: String,
    },
    /// Emitted when function-call arguments streaming is complete.
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        #[serde(default)]
        sequence_number: Option<u64>,
        item_id: String,
        output_index: u32,
        arguments: String,
    },
    /// An event that is emitted when a response finishes as incomplete.
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        #[serde(default)]
        sequence_number: Option<u64>,
        response: OpenAIResponse,
    },
    /// Emitted when there is an additional text delta.
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta {
        #[serde(default)]
        sequence_number: Option<u64>,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
        logprobs: Option<Vec<LogProbs>>,
    },

    /// Emitted when a text delta is done.
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone {
        #[serde(default)]
        sequence_number: Option<u64>,
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
        logprobs: Option<Vec<LogProbs>>,
    },

    /// Emitted when a delta is added to a reasoning summary text.
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ResponseReasoningSummaryTextDelta {
        #[serde(default)]
        sequence_number: Option<u64>,
        item_id: String,
        output_index: u32,
        summary_index: u32,
        delta: String,
    },

    /// Emitted when a reasoning summary text is done.
    #[serde(rename = "response.reasoning_summary_text.done")]
    ResponseReasoningSummaryTextDone {
        #[serde(default)]
        sequence_number: Option<u64>,
        item_id: String,
        output_index: u32,
        summary_index: u32,
        text: String,
    },

    /// Emitted when an error occurs.
    #[serde(rename = "error")]
    ResponseError {
        #[serde(default)]
        sequence_number: Option<u64>,
        code: Option<String>,
        message: String,
        param: Option<String>,
    },
    NotSupported(String),
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
/// Token usage statistics from OpenAI response.
pub(crate) struct ResponseUsage {
    /// Number of input tokens.
    pub input_tokens: u32,
    /// Details of input tokens.
    pub input_tokens_details: InputTokenDetails,
    /// Number of output tokens.
    pub output_tokens: u32,
    /// Details of output tokens.
    pub output_tokens_details: OutputTokenDetails,
    /// Total tokens used.
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct EmbeddingUsage {
    pub total_tokens: u32,
    pub prompt_tokens: u32,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct InputTokenDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OutputTokenDetails {
    pub reasoning_tokens: u32,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ConversationParam {
    pub id: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct IncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReasoningConfig {
    pub effort: Option<ReasoningEffort>,
    pub summary: Option<SummaryType>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReasoningEffort {
    #[default]
    None,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub(crate) enum ToolParams {
    Function {
        name: String,
        parameters: serde_json::Value,
        strict: bool,
        description: Option<String>,
    },
}

// auto, concise, or detailed
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SummaryType {
    #[default]
    Auto,
    Concise,
    Detailed,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct TextConfig {
    pub format: Option<TextResponseFormat>,
    pub verbosity: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum TextResponseFormat {
    Text,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        description: Option<String>,
        strict: Option<bool>,
    },
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIErrorByCode {
    pub code: String,
    pub message: String,
}

/// See <https://platform.openai.com/docs/api-reference/responses/create#responses_create-input>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum Input {
    #[serde(untagged)]
    TextInput(String),
    #[serde(untagged)]
    InputItemList(Vec<InputItem>),
}

/// See <https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum InputItem {
    /// See <https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-input_message>
    InputMessage {
        content: InputItemContent,
        role: Role,
    },
    /// See <https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-item>
    Item(MessageItem),
    /// See <https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-item_reference>
    ItemReference { id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum InputItemContent {
    Text(String),
    InputItemContentList(Vec<ContentType>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Role {
    User,
    Assistant,
    System,
    Developer,
}

/// See <https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-item>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
#[serde(untagged)]
pub(crate) enum MessageItem {
    InputMessage {
        content: Vec<ContentType>,
        role: Role,
        #[serde(rename = "type")]
        type_: String, // always "message"
    },
    #[serde(rename = "output")]
    OutputMessage {
        content: Vec<OutputContent>,
        id: Option<String>,
        role: Role,
        status: Option<String>,
        #[serde(rename = "type")]
        type_: String, // always "message"
    },
    FunctionCall {
        arguments: String,
        call_id: String,
        name: String,
        #[serde(rename = "type")]
        type_: String, // always "function_call"
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    FunctionCallOutput {
        call_id: String,
        output: FunctionCallOutput,
        #[serde(rename = "type")]
        type_: String, // always "function_call_output"
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    Reasoning {
        id: Option<String>,
        summary: Vec<ReasoningSummary>,
        #[serde(rename = "type")]
        type_: String, // always "reasoning"
        content: Option<Vec<ReasoningTextContent>>,
        encrypted_content: Option<String>,
        status: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum FunctionCallOutput {
    Text(String),
    Other(ContentType),
    /// A list of content items -- used when a tool result carries an image
    /// alongside its text output (the Responses API accepts an array of
    /// `input_text`/`input_image` items here, mirroring a user message's
    /// `content` shape).
    List(Vec<ContentType>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReasoningSummary {
    #[serde(rename = "type")]
    pub type_: String, // always "summary_text"
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReasoningTextContent {
    pub type_: String, // always "reasoning_text"
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)]
pub(crate) enum ContentType {
    InputText {
        text: String,
    },
    InputImage {
        detail: ImageDetail,
        file_id: Option<String>,
        image_url: Option<String>,
    },
    InputFile {
        file_data: Option<String>,
        filename: Option<String>,
        file_url: Option<String>,
        file_id: Option<String>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub(crate) enum ImageDetail {
    #[default]
    Auto,
    High,
    Low,
}

/// See <https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-item-output_message-content>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OutputContent {
    OutputText {
        annotations: Vec<OutputTextAnnotation>,
        logprobs: Vec<LogProbs>,
        text: String,
    },
    Refusal {
        refusal: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OutputTextAnnotation {
    FileCitation {
        file_id: String,
        filename: String,
        index: usize,
    },
    UrlCitation {
        end_index: usize,
        start_index: usize,
        url: String,
        title: String,
    },
    ContainerFileCitation {
        file_id: String,
        filename: String,
        start_index: usize,
    },
    FilePath {
        file_id: String,
        index: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct LogProbs {
    pub bytes: Vec<u8>,
    pub logprob: f64,
    pub token: String,
    pub top_logprobs: Vec<TopLogProbs>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct TopLogProbs {
    pub bytes: Vec<u8>,
    pub logprob: f64,
    pub token: String,
}

/// See [OpenAI Embedding API](https://platform.openai.com/docs/api-reference/embeddings/object)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Embedding {
    pub embedding: Vec<f32>,
    pub index: usize,
    pub object: String, // always "embedding"
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct EmbeddingResponse {
    pub object: Option<String>, // always "list"
    pub data: Vec<Embedding>,
    pub model: Option<String>,
    pub usage: Option<EmbeddingUsage>,
}

#[derive(Builder, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(dead_code)]
pub(crate) struct OpenAIEmbeddingOptions {
    // TODO: The input must not exceed the max input tokens for the model
    // (8192 tokens for all embedding models), cannot be an empty string, and
    // any array must be 2048 dimensions or less.
    pub input: Vec<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip)]
    #[builder(default)]
    pub(crate) extra_body: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(skip)]
    #[builder(default)]
    pub(crate) extra_headers: Option<std::collections::HashMap<String, String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: some OpenAI-Responses-API-compatible backends (e.g.
    /// agnes, `https://api.agnes-ai.cn/v1`, observed during
    /// `agent-injector`'s `crates/core/tests/agnes_live.rs`) omit the
    /// `sequence_number` field on SSE stream events for some routed model
    /// variants, even though it is present in OpenAI's own documented
    /// schema. Before this field was made `Option<u64>` with
    /// `#[serde(default)]`, such an event would fail to deserialize into any
    /// `OpenAiStreamEvent` variant (a required-but-missing `u64` field is a
    /// hard deserialization error, not a soft one) and silently fall back to
    /// `OpenAiStreamEvent::NotSupported(raw_json)` at the call site
    /// (`client/mod.rs`'s `parse_stream_sse`), which discards the entire
    /// event including any text/tool-call content it carried. When every
    /// event in a turn is affected this way, the caller sees "no text was
    /// ever delivered" and reports the whole turn as failed, even though the
    /// upstream response was semantically complete and well-formed.
    #[test]
    fn stream_event_deserializes_without_sequence_number() {
        let missing_field_json = serde_json::json!({
            "type": "response.output_text.delta",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
            "logprobs": null,
        });

        let event: OpenAiStreamEvent = serde_json::from_value(missing_field_json)
            .expect("event must deserialize even when sequence_number is absent");

        match event {
            OpenAiStreamEvent::ResponseOutputTextDelta {
                sequence_number,
                delta,
                ..
            } => {
                assert_eq!(sequence_number, None);
                assert_eq!(delta, "Hello");
            }
            other => panic!("expected ResponseOutputTextDelta, got {other:?}"),
        }
    }

    /// The common case (a well-formed, spec-compliant event carrying
    /// `sequence_number`) must keep working identically after widening the
    /// field to `Option<u64>`.
    #[test]
    fn stream_event_deserializes_with_sequence_number_present() {
        let full_json = serde_json::json!({
            "type": "response.output_text.delta",
            "sequence_number": 7,
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
            "logprobs": null,
        });

        let event: OpenAiStreamEvent = serde_json::from_value(full_json)
            .expect("event must deserialize when sequence_number is present");

        match event {
            OpenAiStreamEvent::ResponseOutputTextDelta {
                sequence_number, ..
            } => {
                assert_eq!(sequence_number, Some(7));
            }
            other => panic!("expected ResponseOutputTextDelta, got {other:?}"),
        }
    }

    /// `response.completed` (the event `end_stream()` in `client/mod.rs`
    /// checks for) must also tolerate a missing `sequence_number`, since a
    /// turn that ends on exactly this event type is the case that most
    /// directly determines whether the caller sees a successful completion
    /// or a "stream ended" failure.
    #[test]
    fn response_completed_deserializes_without_sequence_number() {
        let json = serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "agnes-2.0-flash",
            },
        });

        let event: OpenAiStreamEvent = serde_json::from_value(json)
            .expect("response.completed must deserialize even when sequence_number is absent");

        assert!(matches!(
            event,
            OpenAiStreamEvent::ResponseCompleted {
                sequence_number: None,
                ..
            }
        ));
    }
}
