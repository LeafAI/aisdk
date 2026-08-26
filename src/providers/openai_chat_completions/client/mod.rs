//! Client implementation for the OpenAI Chat Completions API.

pub(crate) mod types;

pub(crate) use types::ChatCompletionsOptions;

use crate::core::capabilities::ModelName;
use crate::core::client::{LanguageModelClient, merge_body, merge_headers};
use crate::error::Error;
use crate::providers::openai_chat_completions::OpenAIChatCompletions;
use reqwest::header::CONTENT_TYPE;
use reqwest_eventsource::Event;
use types::*;

impl<M: ModelName> LanguageModelClient for OpenAIChatCompletions<M> {
    type Response = ChatCompletionsResponse;
    type StreamEvent = ChatCompletionsStreamEvent;

    fn path(&self) -> String {
        self.settings
            .path
            .clone()
            .unwrap_or_else(|| "chat/completions".to_string())
    }

    fn method(&self) -> reqwest::Method {
        reqwest::Method::POST
    }

    fn headers(&self) -> crate::error::Result<reqwest::header::HeaderMap> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.settings.api_key).parse().unwrap(),
        );
        merge_headers(
            headers,
            self.settings.headers.as_ref(),
            self.options.extra_headers.as_ref(),
        )
    }

    fn query_params(&self) -> Vec<(&str, &str)> {
        Vec::new()
    }

    fn body(&self) -> crate::error::Result<reqwest::Body> {
        merge_body(
            &self.options,
            self.settings.body.as_ref(),
            self.options.extra_body.as_ref(),
        )
    }

    fn parse_stream_sse(
        event: std::result::Result<Event, Error>,
    ) -> crate::error::Result<Self::StreamEvent> {
        match event {
            Ok(Event::Open) => Ok(ChatCompletionsStreamEvent::Open),
            Ok(Event::Message(msg)) => {
                if msg.data.trim() == "[DONE]" || msg.data.is_empty() {
                    return Ok(ChatCompletionsStreamEvent::Done);
                }

                let chunk: ChatCompletionsStreamChunk =
                    serde_json::from_str(&msg.data).map_err(|e| Error::ApiError {
                        status_code: None,
                        details: format!("Invalid JSON in SSE: {e}"),
                    })?;

                Ok(ChatCompletionsStreamEvent::Chunk(chunk))
            }
            // Already a fully-formed `Error::ApiError` (with its response
            // body, when the rejection carried one) by the time it reaches
            // here -- see `convert_sse_error` in `core/client.rs`.
            Err(e) => Err(e),
        }
    }

    fn end_stream(event: &Self::StreamEvent) -> bool {
        matches!(event, ChatCompletionsStreamEvent::Done)
    }
}
