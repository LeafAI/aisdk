use crate::core::Message;
use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponseContentType, ReasoningEffort, Usage,
};
use crate::providers::anthropic::client::{
    AnthropicAssistantMessageParamContent, AnthropicImageSource, AnthropicMessageDeltaUsage,
    AnthropicMessageParam, AnthropicOptions, AnthropicThinking, AnthropicTool,
    AnthropicToolResultBlock, AnthropicToolResultContent, AnthropicUsage,
    AnthropicUserMessageContent, AnthropicUserMessageContentBlock,
};
use crate::providers::anthropic::extensions;

/// Converts an aisdk [`MediaContent`](crate::core::messages::MediaContent)
/// attachment to an Anthropic `image` content block. Non-image media (audio,
/// etc.) is skipped -- Anthropic's user-message content blocks only support
/// `image` today, not other media kinds -- rather than sent as a malformed
/// request the API would reject.
fn media_to_image_block(
    media: &crate::core::messages::MediaContent,
) -> Option<AnthropicUserMessageContentBlock> {
    media
        .is_image()
        .then(|| AnthropicUserMessageContentBlock::Image {
            source: AnthropicImageSource::base64(media.mime_type.clone(), media.data.clone()),
        })
}

impl From<LanguageModelOptions> for AnthropicOptions {
    fn from(options: LanguageModelOptions) -> Self {
        let extra_body = options.body.clone();
        let extra_headers = options.headers.clone();
        let mut messages = Vec::new();
        let mut request = AnthropicOptions::builder();
        request.model("");

        // TODO: anthropic max_tokens is required. handle compile
        // time checks if not set in core
        let max_tokens = options.max_output_tokens.unwrap_or(10_000);

        // TODO: temperature, top_p, top_k, stop_sequences, and max_tokens are not mapped for Anthropic yet.
        // Add support once provider behavior is confirmed and covered.

        if let Some(system) = options.system
            && !system.is_empty()
        {
            request.system(Some(system));
        } else {
            request.system(None);
        }

        // convert messages to anthropic messages
        for msg in options.messages {
            match msg.message {
                Message::System(s) => {
                    if !s.content.is_empty() {
                        request.system(Some(s.content));
                    }
                }
                Message::User(u) => {
                    // A plain-text message (no media -- the overwhelmingly
                    // common case) keeps the simpler `Text` shape rather
                    // than always emitting a single-element `Blocks` array;
                    // media attachments force the `Blocks` shape, with the
                    // image(s) first and the text last -- matching
                    // Anthropic's documented vision request ordering.
                    let content = if u.media.is_empty() {
                        AnthropicUserMessageContent::Text(u.content)
                    } else {
                        let mut blocks: Vec<AnthropicUserMessageContentBlock> =
                            u.media.iter().filter_map(media_to_image_block).collect();
                        if !u.content.is_empty() {
                            blocks.push(AnthropicUserMessageContentBlock::Text { text: u.content });
                        }
                        AnthropicUserMessageContent::Blocks(blocks)
                    };
                    messages.push(AnthropicMessageParam::User { content });
                }
                Message::Assistant(a) => match a.content {
                    LanguageModelResponseContentType::Text(text) => {
                        messages.push(AnthropicMessageParam::Assistant {
                            content: vec![AnthropicAssistantMessageParamContent::Text { text }],
                        });
                    }
                    LanguageModelResponseContentType::ToolCall(tool) => {
                        messages.push(AnthropicMessageParam::Assistant {
                            content: vec![AnthropicAssistantMessageParamContent::ToolUse {
                                id: tool.tool.id,
                                input: tool.input,
                                name: tool.tool.name,
                            }],
                        });
                    }
                    LanguageModelResponseContentType::Reasoning {
                        content,
                        extensions,
                    } => {
                        // Retrieve Anthropic-specific signature from extensions
                        let signature = extensions
                            .get::<extensions::AnthropicThinkingMetadata>()
                            .signature
                            .clone()
                            .unwrap_or_else(|| content.clone());

                        messages.push(AnthropicMessageParam::Assistant {
                            content: vec![AnthropicAssistantMessageParamContent::Thinking {
                                thinking: content.clone(),
                                signature,
                            }],
                        });
                    }
                    LanguageModelResponseContentType::NotSupported(_) => {}
                    _ => {}
                },
                Message::Tool(tool) => {
                    let text = tool.output.unwrap_or_default().to_string();
                    // A vision-capable tool (e.g. a media-reading tool) sets
                    // `media` on its result so the image is embedded as a
                    // native block inside this `tool_result`, rather than
                    // only described in `output`'s JSON text.
                    let image_blocks: Vec<AnthropicToolResultBlock> = tool
                        .media
                        .iter()
                        .filter(|m| m.is_image())
                        .map(|m| AnthropicToolResultBlock::Image {
                            source: AnthropicImageSource::base64(
                                m.mime_type.clone(),
                                m.data.clone(),
                            ),
                        })
                        .collect();
                    let content = if image_blocks.is_empty() {
                        AnthropicToolResultContent::Text(text)
                    } else {
                        let mut blocks = vec![AnthropicToolResultBlock::Text { text }];
                        blocks.extend(image_blocks);
                        AnthropicToolResultContent::Blocks(blocks)
                    };
                    messages.push(AnthropicMessageParam::User {
                        content: AnthropicUserMessageContent::Blocks(vec![
                            AnthropicUserMessageContentBlock::ToolResult {
                                tool_use_id: tool.tool.id,
                                content,
                            },
                        ]),
                    });
                }
                Message::Developer(dev) => {
                    messages.push(AnthropicMessageParam::User {
                        content:
                            crate::providers::anthropic::client::AnthropicUserMessageContent::Text(
                                format!("<developer>\n{dev}\n</developer>"),
                            ),
                    });
                }
            }
        }
        // update messages
        request.messages(messages);

        // convert tools to anthropic tools
        if let Some(tools) = options.tools {
            request.tools(Some(
                tools
                    .tools
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .iter()
                    .map(|t| {
                        let tool = t.clone();
                        let mut tool_schema = tool.input_schema.to_value();
                        if let Some(schema) = tool_schema.as_object_mut() {
                            schema.remove("$schema");
                        };
                        AnthropicTool {
                            name: tool.name,
                            description: tool.description,
                            input_schema: tool_schema,
                        }
                    })
                    .collect(),
            ));
        }

        // convert reasoning to antropic thinking
        request.thinking(options.reasoning_effort.map(|effort| match effort {
            // None disables thinking entirely
            ReasoningEffort::None => AnthropicThinking::Disable,
            // Low is 25% of the max_tokens
            ReasoningEffort::Low => AnthropicThinking::Enable {
                budget_tokens: (max_tokens / 4) as usize,
            },
            // Medium is 50% of the max_tokens
            ReasoningEffort::Medium => AnthropicThinking::Enable {
                budget_tokens: (max_tokens / 2) as usize,
            },
            // High is 75% of the max_tokens
            ReasoningEffort::High => AnthropicThinking::Enable {
                budget_tokens: (max_tokens - (max_tokens / 4)) as usize,
            },
            // XHigh is 90% of the max_tokens
            ReasoningEffort::XHigh => AnthropicThinking::Enable {
                budget_tokens: ((max_tokens * 9) / 10) as usize,
            },
        }));

        let mut request = request.build().expect("Failed to build AntropicRequest");
        request.extra_body = extra_body;
        request.extra_headers = extra_headers;
        request
    }
}

impl From<AnthropicUsage> for Usage {
    fn from(usage: AnthropicUsage) -> Self {
        Self {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            cached_tokens: Some(usage.cache_creation_input_tokens + usage.cache_read_input_tokens),
            reasoning_tokens: None,
            cache_miss_tokens: None,
        }
    }
}

impl From<AnthropicMessageDeltaUsage> for Usage {
    fn from(usage: AnthropicMessageDeltaUsage) -> Self {
        Self {
            input_tokens: Some(usage.input_tokens.unwrap_or(0)),
            output_tokens: Some(usage.output_tokens),
            cached_tokens: Some(
                usage.cache_creation_input_tokens.unwrap_or(0)
                    + usage.cache_read_input_tokens.unwrap_or(0),
            ),
            reasoning_tokens: None,
            cache_miss_tokens: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Message;
    use crate::core::language_model::{LanguageModelOptions, ReasoningEffort};
    use crate::core::tools::{Tool, ToolExecute, ToolList};
    use schemars::{JsonSchema, schema_for};
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct SumInput {
        a: i32,
        b: i32,
    }

    #[test]
    fn test_scalar_request_options_map_to_anthropic_body() {
        let options = LanguageModelOptions {
            system: Some("You are helpful".to_string()),
            messages: vec![Message::User("Hello".to_string().into()).into()],
            ..Default::default()
        };

        let req: AnthropicOptions = options.into();

        assert_eq!(req.system.as_deref(), Some("You are helpful"));
        assert_eq!(req.messages.len(), 1);

        match &req.messages[0] {
            AnthropicMessageParam::User { content } => match content {
                crate::providers::anthropic::client::AnthropicUserMessageContent::Text(text) => {
                    assert_eq!(text, "Hello")
                }
                _ => panic!("expected user text message"),
            },
            _ => panic!("expected user message"),
        }
    }

    #[test]
    fn test_reasoning_maps_to_thinking_budget() {
        let options = LanguageModelOptions {
            max_output_tokens: Some(200),
            reasoning_effort: Some(ReasoningEffort::High),
            ..Default::default()
        };

        let req: AnthropicOptions = options.into();

        match req.thinking {
            Some(AnthropicThinking::Enable { budget_tokens }) => {
                assert_eq!(budget_tokens, 150);
            }
            _ => panic!("expected anthropic thinking to be enabled"),
        }
    }

    #[test]
    fn test_tools_map_to_anthropic_body() {
        let tool = Tool::builder()
            .name("sum")
            .description("Adds two numbers")
            .input_schema(schema_for!(SumInput))
            .execute(ToolExecute::from_sync(|_, _| Ok("3".to_string())))
            .build()
            .expect("tool should build");

        let options = LanguageModelOptions {
            tools: Some(ToolList::new(vec![tool])),
            ..Default::default()
        };

        let req: AnthropicOptions = options.into();
        let tools = req.tools.expect("tools should be present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "sum");
        assert_eq!(tools[0].description, "Adds two numbers");
        assert_eq!(tools[0].input_schema["type"], json!("object"));
        assert!(tools[0].input_schema["properties"].get("a").is_some());
        assert!(tools[0].input_schema["properties"].get("b").is_some());
        assert!(tools[0].input_schema.get("$schema").is_none());
    }

    #[test]
    fn test_anthropic_usage_to_usage_conversion() {
        let usage = AnthropicUsage {
            cache_creation: crate::providers::anthropic::client::AnthropicCacheCreation {
                ephemeral_1h_input_tokens: 0,
                ephemeral_5m_input_tokens: 0,
            },
            cache_creation_input_tokens: 4,
            cache_read_input_tokens: 6,
            input_tokens: 100,
            output_tokens: 50,
            server_tool_use: crate::providers::anthropic::client::AnthropicServerToolUsage::default(
            ),
            service_tier: "standard".to_string(),
        };

        let sdk_usage: Usage = usage.into();
        assert_eq!(sdk_usage.input_tokens, Some(100));
        assert_eq!(sdk_usage.output_tokens, Some(50));
        assert_eq!(sdk_usage.cached_tokens, Some(10));
        assert_eq!(sdk_usage.reasoning_tokens, None);
    }

    #[test]
    fn test_anthropic_delta_usage_to_usage_conversion() {
        let usage = AnthropicMessageDeltaUsage {
            cache_creation_input_tokens: Some(3),
            cache_read_input_tokens: Some(2),
            input_tokens: Some(10),
            output_tokens: 7,
            server_tool_use: None,
        };

        let sdk_usage: Usage = usage.into();
        assert_eq!(sdk_usage.input_tokens, Some(10));
        assert_eq!(sdk_usage.output_tokens, Some(7));
        assert_eq!(sdk_usage.cached_tokens, Some(5));
        assert_eq!(sdk_usage.reasoning_tokens, None);
    }

    #[test]
    fn test_user_message_with_image_media_becomes_blocks_image_then_text() {
        use crate::core::messages::{MediaContent, UserMessage};
        let options = LanguageModelOptions {
            messages: vec![
                Message::User(UserMessage::with_media(
                    "describe this image",
                    vec![MediaContent::new("aGVsbG8=", "image/png")],
                ))
                .into(),
            ],
            ..Default::default()
        };

        let req: AnthropicOptions = options.into();
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0] {
            AnthropicMessageParam::User { content } => match content {
                AnthropicUserMessageContent::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 2, "image block + text block");
                    match &blocks[0] {
                        AnthropicUserMessageContentBlock::Image { source } => {
                            assert_eq!(source.source_type, "base64");
                            assert_eq!(source.media_type, "image/png");
                            assert_eq!(source.data, "aGVsbG8=");
                        }
                        other => panic!("expected image block first, got: {other:?}"),
                    }
                    match &blocks[1] {
                        AnthropicUserMessageContentBlock::Text { text } => {
                            assert_eq!(text, "describe this image");
                        }
                        other => panic!("expected text block last, got: {other:?}"),
                    }
                }
                other => panic!("expected Blocks content, got: {other:?}"),
            },
            _ => panic!("expected user message"),
        }
    }

    #[test]
    fn test_user_message_without_media_stays_plain_text() {
        let options = LanguageModelOptions {
            messages: vec![Message::User("no image here".to_string().into()).into()],
            ..Default::default()
        };
        let req: AnthropicOptions = options.into();
        match &req.messages[0] {
            AnthropicMessageParam::User { content } => {
                assert!(
                    matches!(content, AnthropicUserMessageContent::Text(t) if t == "no image here"),
                    "a media-less user message must keep the simpler Text shape, got: {content:?}"
                );
            }
            _ => panic!("expected user message"),
        }
    }

    #[test]
    fn test_tool_result_with_image_media_embeds_image_block() {
        use crate::core::messages::MediaContent;
        use crate::core::tools::{ToolDetails, ToolResultInfo};
        let mut result = ToolResultInfo::new("media_read");
        result.tool = ToolDetails {
            name: "media_read".to_string(),
            id: "call-1".to_string(),
        };
        result.output = Ok(json!({"path": "photo.png"}));
        result.media = vec![MediaContent::new("aW1hZ2U=", "image/jpeg")];

        let options = LanguageModelOptions {
            messages: vec![Message::Tool(result).into()],
            ..Default::default()
        };
        let req: AnthropicOptions = options.into();
        match &req.messages[0] {
            AnthropicMessageParam::User { content } => match content {
                AnthropicUserMessageContent::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 1);
                    match &blocks[0] {
                        AnthropicUserMessageContentBlock::ToolResult {
                            tool_use_id,
                            content,
                        } => {
                            assert_eq!(tool_use_id, "call-1");
                            match content {
                                AnthropicToolResultContent::Blocks(inner) => {
                                    assert_eq!(inner.len(), 2, "text block + image block");
                                    assert!(matches!(
                                        &inner[0],
                                        AnthropicToolResultBlock::Text { .. }
                                    ));
                                    match &inner[1] {
                                        AnthropicToolResultBlock::Image { source } => {
                                            assert_eq!(source.media_type, "image/jpeg");
                                            assert_eq!(source.data, "aW1hZ2U=");
                                        }
                                        other => panic!("expected image block, got: {other:?}"),
                                    }
                                }
                                other => {
                                    panic!("expected Blocks tool_result content, got: {other:?}")
                                }
                            }
                        }
                        other => panic!("expected ToolResult block, got: {other:?}"),
                    }
                }
                other => panic!("expected Blocks content, got: {other:?}"),
            },
            _ => panic!("expected user message wrapping the tool result"),
        }
    }

    #[test]
    fn test_tool_result_without_media_stays_plain_text_content() {
        use crate::core::tools::{ToolDetails, ToolResultInfo};
        let mut result = ToolResultInfo::new("file_read");
        result.tool = ToolDetails {
            name: "file_read".to_string(),
            id: "call-2".to_string(),
        };
        result.output = Ok(json!("file contents"));

        let options = LanguageModelOptions {
            messages: vec![Message::Tool(result).into()],
            ..Default::default()
        };
        let req: AnthropicOptions = options.into();
        match &req.messages[0] {
            AnthropicMessageParam::User { content } => match content {
                AnthropicUserMessageContent::Blocks(blocks) => match &blocks[0] {
                    AnthropicUserMessageContentBlock::ToolResult { content, .. } => {
                        assert!(
                            matches!(content, AnthropicToolResultContent::Text(_)),
                            "a media-less tool result must keep the simpler Text shape, got: {content:?}"
                        );
                    }
                    other => panic!("expected ToolResult block, got: {other:?}"),
                },
                other => panic!("expected Blocks content, got: {other:?}"),
            },
            _ => panic!("expected user message wrapping the tool result"),
        }
    }
}
