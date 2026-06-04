//! Helper functions and conversions for the OpenAI Chat Completions provider.

use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponseContentType, ReasoningEffort, Usage,
};
use crate::core::messages::Message;
use crate::core::tools::Tool as SdkTool;
use crate::providers::openai_chat_completions::client::{self, types};

// ============================================================================
// LanguageModelOptions -> ChatCompletionsOptions
// ============================================================================

impl From<LanguageModelOptions> for client::ChatCompletionsOptions {
    fn from(options: LanguageModelOptions) -> Self {
        let extra_body = options.body.clone();
        let extra_headers = options.headers.clone();
        let mut messages: Vec<types::ChatMessage> = Vec::new();

        if let Some(system_prompt) = options.system {
            messages.push(types::ChatMessage {
                role: types::Role::System,
                content: Some(system_prompt),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        messages.extend(
            options
                .messages
                .into_iter()
                .map(|tagged| tagged.message.into()),
        );

        let tools: Option<Vec<types::Tool>> = options.tools.map(|tool_list| {
            tool_list
                .tools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .map(|t| t.clone().into())
                .collect()
        });

        let response_format = options.schema.map(|schema| {
            let mut json_value = serde_json::to_value(schema).unwrap();

            // Ensure required fields for OpenAI Structured Outputs
            if let serde_json::Value::Object(ref mut obj) = json_value {
                obj.insert(
                    "additionalProperties".to_string(),
                    serde_json::Value::Bool(false),
                );
            }

            types::ResponseFormat::JsonSchema {
                json_schema: types::JsonSchemaDefinition {
                    name: json_value
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Response")
                        .to_string(),
                    schema: json_value.clone(),
                    description: json_value
                        .get("description")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    strict: Some(true),
                },
            }
        });

        let reasoning_effort = options.reasoning_effort.map(|effort| {
            match effort {
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High => "high",
            }
            .to_string()
        });

        let tool_choice = if tools.is_some() {
            Some(types::ToolChoice::String("auto".to_string()))
        } else {
            None
        };

        let parallel_tool_calls = if tools.is_some() { Some(true) } else { None };

        client::ChatCompletionsOptions {
            model: "".to_string(),
            messages,
            frequency_penalty: options.frequency_penalty,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            max_completion_tokens: options.max_output_tokens,
            n: None,
            presence_penalty: options.presence_penalty,
            response_format,
            seed: options.seed,
            stop: options.stop_sequences.map(|seqs| {
                if seqs.len() == 1 {
                    types::StopSequences::Single(seqs[0].clone())
                } else {
                    types::StopSequences::Multiple(seqs.into_iter().take(4).collect())
                }
            }),
            stream: None,
            stream_options: None,
            temperature: options.temperature.map(|t| t as f32 / 100.0),
            top_p: options.top_p.map(|t| t as f32 / 100.0),
            tools,
            tool_choice,
            parallel_tool_calls,
            reasoning_effort,
            verbosity: None,
            extra_body,
            extra_headers,
        }
    }
}

// ============================================================================
// SDK Message -> ChatMessage
// ============================================================================

impl From<Message> for types::ChatMessage {
    fn from(msg: Message) -> Self {
        match msg {
            Message::System(s) => types::ChatMessage {
                role: types::Role::System,
                content: Some(s.content),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message::User(u) => types::ChatMessage {
                role: types::Role::User,
                content: Some(u.content),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message::Assistant(a) => match a.content {
                LanguageModelResponseContentType::Text(text) => types::ChatMessage {
                    role: types::Role::Assistant,
                    content: Some(text),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                LanguageModelResponseContentType::ToolCall(tool_info) => types::ChatMessage {
                    role: types::Role::Assistant,
                    content: Some("".to_string()),
                    name: None,
                    tool_calls: Some(vec![types::ToolCall {
                        id: tool_info.tool.id.clone(),
                        type_: "function".to_string(),
                        function: types::FunctionCall {
                            name: tool_info.tool.name.clone(),
                            arguments: tool_info.input.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                    reasoning_content: None,
                },
                LanguageModelResponseContentType::Reasoning { content, .. } => types::ChatMessage {
                    role: types::Role::Assistant,
                    content: Some(String::new()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: Some(content),
                },
                LanguageModelResponseContentType::ReasonedResponse {
                    reasoning,
                    text,
                    tool_calls,
                    ..
                } => {
                    let tc_list = if tool_calls.is_empty() {
                        None
                    } else {
                        Some(
                            tool_calls
                                .into_iter()
                                .map(|tc| types::ToolCall {
                                    id: tc.tool.id,
                                    type_: "function".to_string(),
                                    function: types::FunctionCall {
                                        name: tc.tool.name,
                                        arguments: tc.input.to_string(),
                                    },
                                })
                                .collect(),
                        )
                    };
                    types::ChatMessage {
                        role: types::Role::Assistant,
                        content: Some(text),
                        name: None,
                        tool_calls: tc_list,
                        tool_call_id: None,
                        reasoning_content: Some(reasoning),
                    }
                }
                _ => types::ChatMessage {
                    role: types::Role::Assistant,
                    content: None,
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            },
            Message::Tool(tool_result) => types::ChatMessage {
                role: types::Role::Tool,
                content: Some(
                    tool_result
                        .output
                        .unwrap_or_else(|e| serde_json::Value::String(e.to_string()))
                        .to_string(),
                ),
                name: Some(tool_result.tool.name),
                tool_calls: None,
                tool_call_id: Some(tool_result.tool.id),
                reasoning_content: None,
            },
            Message::Developer(d) => types::ChatMessage {
                role: types::Role::Developer,
                content: Some(d),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        }
    }
}

// ============================================================================
// SDK Tool -> ChatCompletions Tool
// ============================================================================

impl From<SdkTool> for types::Tool {
    fn from(tool: SdkTool) -> Self {
        let mut params = tool.input_schema.to_value();

        // Remove schema metadata fields that may conflict with strict mode
        if let serde_json::Value::Object(ref mut obj) = params {
            obj.remove("$schema");
            obj.remove("title");
        }

        // Ensure required fields for OpenAI Chat Completions
        params["type"] = serde_json::Value::String("object".to_string());
        params["additionalProperties"] = serde_json::Value::Bool(false);

        if !params
            .get("properties")
            .map(|p| p.is_object())
            .unwrap_or(false)
        {
            params["properties"] = serde_json::Value::Object(serde_json::Map::new());
        }

        types::Tool {
            type_: "function".to_string(),
            function: types::FunctionDefinition {
                name: tool.name,
                description: Some(tool.description),
                parameters: params,
                strict: Some(true),
            },
        }
    }
}

// ============================================================================
// ChatCompletions Usage -> SDK Usage
// ============================================================================

impl From<types::Usage> for Usage {
    fn from(usage: types::Usage) -> Self {
        Self {
            input_tokens: Some(usage.prompt_tokens as usize),
            output_tokens: Some(usage.completion_tokens as usize),
            reasoning_tokens: usage
                .completion_tokens_details
                .map(|d| d.reasoning_tokens as usize),
            cached_tokens: usage
                .prompt_tokens_details
                .map(|d| d.cached_tokens as usize),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tools::{Tool, ToolExecute, ToolList, ToolResultInfo};
    use schemars::{JsonSchema, schema_for};
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct StructuredOutput {
        answer: String,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct SumInput {
        a: i32,
        b: i32,
    }

    #[test]
    fn test_message_conversion_system() {
        let msg = Message::System("You are helpful".to_string().into());
        let chat_msg: types::ChatMessage = msg.into();

        assert_eq!(chat_msg.role, types::Role::System);
        assert_eq!(chat_msg.content, Some("You are helpful".to_string()));
        assert!(chat_msg.tool_calls.is_none());
    }

    #[test]
    fn test_message_conversion_user() {
        let msg = Message::User("Hello".to_string().into());
        let chat_msg: types::ChatMessage = msg.into();

        assert_eq!(chat_msg.role, types::Role::User);
        assert_eq!(chat_msg.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_stop_sequences_single() {
        let options = LanguageModelOptions {
            stop_sequences: Some(vec!["STOP".to_string()]),
            ..Default::default()
        };

        let completions_opts: client::ChatCompletionsOptions = options.into();
        assert!(matches!(
            completions_opts.stop,
            Some(types::StopSequences::Single(_))
        ));
    }

    #[test]
    fn test_stop_sequences_multiple_truncated() {
        let options = LanguageModelOptions {
            stop_sequences: Some(vec![
                "S1".to_string(),
                "S2".to_string(),
                "S3".to_string(),
                "S4".to_string(),
                "S5".to_string(),
            ]),
            ..Default::default()
        };

        let completions_opts: client::ChatCompletionsOptions = options.into();
        if let Some(types::StopSequences::Multiple(seqs)) = completions_opts.stop {
            assert_eq!(seqs.len(), 4);
        }
    }

    #[test]
    fn test_scalar_request_options_map_to_chat_completions_body() {
        let options = LanguageModelOptions {
            system: Some("You are helpful".to_string()),
            messages: vec![Message::User("Hello".to_string().into()).into()],
            temperature: Some(70),
            top_p: Some(90),
            seed: Some(42),
            frequency_penalty: Some(0.5),
            stop_sequences: Some(vec!["END".to_string()]),
            ..Default::default()
        };

        let completions_opts: client::ChatCompletionsOptions = options.into();

        assert!(
            completions_opts
                .temperature
                .is_some_and(|value| (value - 0.7).abs() < f32::EPSILON)
        );
        assert!(
            completions_opts
                .top_p
                .is_some_and(|value| (value - 0.9).abs() < f32::EPSILON)
        );
        assert_eq!(completions_opts.seed, Some(42));
        assert_eq!(completions_opts.frequency_penalty, Some(0.5));
        assert_eq!(completions_opts.messages.len(), 2);
        assert_eq!(completions_opts.messages[0].role, types::Role::System);
        assert_eq!(
            completions_opts.messages[0].content.as_deref(),
            Some("You are helpful")
        );
        assert_eq!(completions_opts.messages[1].role, types::Role::User);
        assert_eq!(
            completions_opts.messages[1].content.as_deref(),
            Some("Hello")
        );
        assert!(matches!(
            completions_opts.stop,
            Some(types::StopSequences::Single(sequence)) if sequence == "END"
        ));
    }

    #[test]
    fn test_schema_and_reasoning_options_map_to_chat_completions_body() {
        let options = LanguageModelOptions {
            schema: Some(schema_for!(StructuredOutput)),
            reasoning_effort: Some(ReasoningEffort::High),
            ..Default::default()
        };

        let completions_opts: client::ChatCompletionsOptions = options.into();

        let Some(types::ResponseFormat::JsonSchema { json_schema }) =
            completions_opts.response_format
        else {
            panic!("expected json schema response format");
        };

        assert_eq!(json_schema.name, "StructuredOutput");
        assert_eq!(json_schema.strict, Some(true));
        assert_eq!(json_schema.schema["additionalProperties"], json!(false));
        assert!(json_schema.schema["properties"].get("answer").is_some());
        assert_eq!(completions_opts.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn test_tools_map_to_chat_completions_body() {
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

        let completions_opts: client::ChatCompletionsOptions = options.into();
        let tools = completions_opts.tools.expect("tools should be present");
        let tool = &tools[0];

        assert_eq!(tools.len(), 1);
        assert_eq!(tool.function.name, "sum");
        assert_eq!(
            tool.function.description.as_deref(),
            Some("Adds two numbers")
        );
        assert_eq!(tool.function.strict, Some(true));
        assert_eq!(tool.function.parameters["type"], json!("object"));
        assert_eq!(
            tool.function.parameters["additionalProperties"],
            json!(false)
        );
        assert!(tool.function.parameters["properties"].get("a").is_some());
        assert!(tool.function.parameters["properties"].get("b").is_some());
        assert!(matches!(
            completions_opts.tool_choice,
            Some(types::ToolChoice::String(choice)) if choice == "auto"
        ));
        assert_eq!(completions_opts.parallel_tool_calls, Some(true));
    }

    /// ReasonedResponse variant: one SDK message → one ChatMessage with
    /// reasoning_content + text + tool_calls all in one message.
    #[test]
    fn test_reasoned_response_conversion() {
        let msg = Message::Assistant(crate::core::messages::AssistantMessage {
            content: LanguageModelResponseContentType::ReasonedResponse {
                reasoning: "I need two tools".to_string(),
                text: String::new(),
                tool_calls: vec![
                    {
                        let mut tc = crate::core::tools::ToolCallInfo::new("read_file");
                        tc.id("call_1".to_string());
                        tc.input(serde_json::json!({"path": "a.rs"}));
                        tc
                    },
                    {
                        let mut tc = crate::core::tools::ToolCallInfo::new("search");
                        tc.id("call_2".to_string());
                        tc.input(serde_json::json!({"pattern": "fn main"}));
                        tc
                    },
                ],
                extensions: crate::extensions::Extensions::default(),
            },
            usage: None,
        });

        let chat_msg: types::ChatMessage = msg.into();
        assert_eq!(chat_msg.role, types::Role::Assistant);
        assert_eq!(
            chat_msg.reasoning_content.as_deref(),
            Some("I need two tools")
        );
        let calls = chat_msg
            .tool_calls
            .as_ref()
            .expect("tool_calls must be present");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[1].function.name, "search");
    }

    /// ReasonedResponse without tool calls: reasoning + text only.
    #[test]
    fn test_reasoned_response_no_tool_calls() {
        let msg = Message::Assistant(crate::core::messages::AssistantMessage {
            content: LanguageModelResponseContentType::ReasonedResponse {
                reasoning: "Let me think...".to_string(),
                text: "The answer is 42".to_string(),
                tool_calls: vec![],
                extensions: crate::extensions::Extensions::default(),
            },
            usage: None,
        });

        let chat_msg: types::ChatMessage = msg.into();
        assert_eq!(chat_msg.role, types::Role::Assistant);
        assert_eq!(
            chat_msg.reasoning_content.as_deref(),
            Some("Let me think...")
        );
        assert_eq!(chat_msg.content.as_deref(), Some("The answer is 42"));
        assert!(chat_msg.tool_calls.is_none());
    }

    /// ReasonedResponse with 3 tool calls serialized: all tool_calls must be present
    /// in valid JSON — regression test for truncation in large multi-message requests.
    #[test]
    fn test_reasoned_response_serialization_with_many_messages() {
        let mut messages = Vec::new();
        messages.push(Message::System("You are a helpful assistant".to_string().into()).into());

        // Simulate 6 agent-tool round-trips (12 messages total)
        for i in 0..6 {
            messages.push(Message::User(format!("Request {}", i).into()).into());

            let mut tc1 = crate::core::tools::ToolCallInfo::new("file_read");
            tc1.id(format!("call_{i}_0"));
            tc1.input(serde_json::json!({"path": format!("/tmp/file{}.txt", i), "start_line": 1, "end_line": 50}));
            let mut tc2 = crate::core::tools::ToolCallInfo::new("file_glob");
            tc2.id(format!("call_{i}_1"));
            tc2.input(serde_json::json!({"pattern": format!("**/*.{}.kt", i), "working_dir": "/Users/test"}));
            let mut tc3 = crate::core::tools::ToolCallInfo::new("search");
            tc3.id(format!("call_{i}_2"));
            tc3.input(serde_json::json!({"query": format!("test query {}", i)}));

            messages.push(
                Message::Assistant(crate::core::messages::AssistantMessage {
                    content: LanguageModelResponseContentType::ReasonedResponse {
                        reasoning: format!("Thinking about request {}...", i),
                        text: String::new(),
                        tool_calls: vec![tc1, tc2, tc3],
                        extensions: crate::extensions::Extensions::default(),
                    },
                    usage: None,
                })
                .into(),
            );

            messages.push(
                Message::Tool(ToolResultInfo {
                    tool: crate::core::tools::ToolDetails {
                        name: "file_read".to_string(),
                        id: format!("call_{i}_0"),
                    },
                    output: Ok(serde_json::json!({"content": format!("content of file {}", i)})),
                })
                .into(),
            );
            messages.push(
                Message::Tool(ToolResultInfo {
                    tool: crate::core::tools::ToolDetails {
                        name: "file_glob".to_string(),
                        id: format!("call_{i}_1"),
                    },
                    output: Ok(serde_json::json!({"matches": [format!("file{}.kt", i)]})),
                })
                .into(),
            );
            messages.push(
                Message::Tool(ToolResultInfo {
                    tool: crate::core::tools::ToolDetails {
                        name: "search".to_string(),
                        id: format!("call_{i}_2"),
                    },
                    output: Ok(serde_json::json!({"results": []})),
                })
                .into(),
            );
        }

        let options = LanguageModelOptions {
            messages,
            ..Default::default()
        };

        let completions_opts: client::ChatCompletionsOptions = options.into();
        let json_str =
            serde_json::to_string_pretty(&completions_opts).expect("serialization must succeed");

        // Must be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("output must be valid JSON");

        let msgs = parsed["messages"]
            .as_array()
            .expect("messages must be array");
        // 1 system + 6*(1 user + 1 assistant + 3 tool) = 1 + 6*5 = 31
        assert_eq!(msgs.len(), 31, "expected 31 messages, got {}", msgs.len());

        // Every assistant message with reasoning should have exactly 3 tool_calls
        let mut assistant_count = 0;
        for msg in msgs {
            if msg["role"] == "assistant" && msg["reasoning_content"].is_string() {
                assistant_count += 1;
                let tcs = msg["tool_calls"]
                    .as_array()
                    .expect("tool_calls must be present and an array");
                assert_eq!(
                    tcs.len(),
                    3,
                    "assistant message {} should have 3 tool_calls",
                    assistant_count
                );
                for tc in tcs {
                    assert!(tc["id"].is_string(), "each tool_call must have id");
                    assert_eq!(
                        tc["type"], "function",
                        "each tool_call must have type=function"
                    );
                    assert!(
                        tc["function"]["name"].is_string(),
                        "tool_call must have function name"
                    );
                    assert!(
                        tc["function"]["arguments"].is_string(),
                        "tool_call must have function arguments"
                    );
                }
            }
        }
        assert_eq!(
            assistant_count, 6,
            "expected 6 assistant messages with reasoning"
        );

        // Verify the last assistant message's tool_calls are complete
        let last_assistant = msgs
            .iter()
            .rev()
            .find(|m| m["role"] == "assistant" && m["reasoning_content"].is_string())
            .expect("must have assistant message");
        let last_tcs = last_assistant["tool_calls"].as_array().unwrap();
        assert_eq!(last_tcs.len(), 3);
        assert_eq!(last_tcs[2]["function"]["name"], "search");
        assert_eq!(last_tcs[2]["id"], "call_5_2");
    }

    #[test]
    fn test_usage_conversion() {
        let usage = types::Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: Some(types::PromptTokensDetails {
                cached_tokens: 20,
                audio_tokens: None,
            }),
            completion_tokens_details: Some(types::CompletionTokensDetails {
                reasoning_tokens: 10,
                audio_tokens: None,
                accepted_prediction_tokens: None,
                rejected_prediction_tokens: None,
            }),
        };

        let sdk_usage: Usage = usage.into();
        assert_eq!(sdk_usage.input_tokens, Some(100));
        assert_eq!(sdk_usage.output_tokens, Some(50));
        assert_eq!(sdk_usage.cached_tokens, Some(20));
        assert_eq!(sdk_usage.reasoning_tokens, Some(10));
    }
}
