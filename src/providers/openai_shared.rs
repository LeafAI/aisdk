//! Shared helpers for OpenAI-shaped providers (the Responses API and the
//! Chat Completions API, plus every `openai_chat_completions`-based
//! compatible provider).
//!
//! Both APIs' "strict" structured-output/tool-calling mode requires every
//! key present in a schema's `properties` object to also be listed in its
//! `required` array -- a property that is semantically optional is instead
//! expressed by adding `"null"` to its own `type` (see
//! <https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required>).
//! A caller's `schemars`-derived schema routinely violates this: a field
//! behind `#[serde(default)]`/`Option<T>` is typically *not* listed in
//! `required`, which OpenAI rejects outright with HTTP 400
//! `invalid_function_parameters` before the tool is ever offered to the
//! model -- the caller never even gets a chance to decide whether the field
//! was actually supplied.

use serde_json::Value;

/// Rewrites `schema` in place so every property key is listed in
/// `required`, nullifying the type of any property that wasn't already
/// required (preserving optionality from the model's perspective: it may
/// still omit the field by returning `null` for it). Recurses into nested
/// object schemas under `properties` and into `items` for array schemas,
/// since OpenAI's strict mode enforces this invariant at every nesting
/// level, not just the top one.
///
/// No-op for a non-object `schema` value, or one with no `properties`
/// object -- nothing to make required.
pub(crate) fn make_strict_schema_required(schema: &mut Value) {
    let Value::Object(obj) = schema else { return };

    let Some(Value::Object(properties)) = obj.get("properties").cloned() else {
        return;
    };

    let already_required: Vec<String> = obj
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let mut new_properties = serde_json::Map::new();
    let mut required: Vec<Value> = Vec::with_capacity(properties.len());

    for (key, mut value) in properties {
        let was_required = already_required.iter().any(|r| r == &key);
        if !was_required {
            nullify_type(&mut value);
        }
        // Recurse into nested object/array schemas regardless of whether
        // this property itself was already required -- a nested object can
        // have its own optional sub-fields independent of whether the
        // parent object field itself is required.
        make_strict_schema_required(&mut value);
        if let Some(items) = value.get_mut("items") {
            make_strict_schema_required(items);
        }
        required.push(Value::String(key.clone()));
        new_properties.insert(key, value);
    }

    obj.insert("properties".to_string(), Value::Object(new_properties));
    obj.insert("required".to_string(), Value::Array(required));
}

/// Adds `"null"` to a property schema's `type`, so a field that wasn't
/// already required can still be omitted (as `null`) under strict mode.
/// Handles the three shapes `schemars` can emit for `type`: a bare string
/// (`"string"` -> `["string", "null"]`), an existing array (appends `null`
/// if not already present), or no `type` at all (left untouched -- e.g. a
/// `$ref`, `enum`-only, or `anyOf`/`oneOf` schema, which OpenAI's strict
/// mode handles differently and this helper doesn't attempt to rewrite).
fn nullify_type(value: &mut Value) {
    let Value::Object(obj) = value else { return };
    match obj.get("type").cloned() {
        Some(Value::String(t)) => {
            obj.insert(
                "type".to_string(),
                Value::Array(vec![Value::String(t), Value::String("null".to_string())]),
            );
        }
        Some(Value::Array(mut types)) if !types.iter().any(|t| t.as_str() == Some("null")) => {
            types.push(Value::String("null".to_string()));
            obj.insert("type".to_string(), Value::Array(types));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn leaves_an_already_fully_required_schema_untouched_in_shape() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "a": { "type": "string" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"]
        });
        make_strict_schema_required(&mut schema);
        assert_eq!(schema["required"], json!(["a", "b"]));
        assert_eq!(schema["properties"]["a"]["type"], json!("string"));
        assert_eq!(schema["properties"]["b"]["type"], json!("integer"));
    }

    #[test]
    fn adds_missing_keys_to_required_and_nullifies_their_type() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" },
                "timeout_secs": { "type": "integer", "minimum": 1 }
            },
            "required": ["command"]
        });
        make_strict_schema_required(&mut schema);
        let required = schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 2);
        assert!(required.contains(&json!("command")));
        assert!(required.contains(&json!("timeout_secs")));
        // Previously-required field's type is untouched.
        assert_eq!(schema["properties"]["command"]["type"], json!("string"));
        // Previously-optional field gains "null" in its type.
        assert_eq!(
            schema["properties"]["timeout_secs"]["type"],
            json!(["integer", "null"])
        );
        // Unrelated keywords on the property survive.
        assert_eq!(schema["properties"]["timeout_secs"]["minimum"], json!(1));
    }

    #[test]
    fn handles_a_schema_with_no_required_array_at_all() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "agent": { "type": "string" }
            }
        });
        make_strict_schema_required(&mut schema);
        assert_eq!(schema["required"], json!(["agent"]));
    }

    #[test]
    fn does_not_double_nullify_a_type_array_that_already_has_null() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "note": { "type": ["string", "null"] }
            }
        });
        make_strict_schema_required(&mut schema);
        assert_eq!(
            schema["properties"]["note"]["type"],
            json!(["string", "null"])
        );
    }

    #[test]
    fn recurses_into_nested_object_properties() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": { "type": "string" },
                            "verdict": { "type": "string" }
                        },
                        "required": ["file"]
                    }
                }
            },
            "required": ["findings"]
        });
        make_strict_schema_required(&mut schema);
        let item_required = schema["properties"]["findings"]["items"]["required"]
            .as_array()
            .unwrap();
        assert_eq!(item_required.len(), 2);
        assert!(item_required.contains(&json!("file")));
        assert!(item_required.contains(&json!("verdict")));
        assert_eq!(
            schema["properties"]["findings"]["items"]["properties"]["verdict"]["type"],
            json!(["string", "null"])
        );
    }

    #[test]
    fn no_op_for_a_schema_with_no_properties_object() {
        let mut schema = json!({ "type": "string" });
        let before = schema.clone();
        make_strict_schema_required(&mut schema);
        assert_eq!(schema, before);
    }

    #[test]
    fn leaves_a_ref_or_enum_only_property_untouched_but_still_requires_it() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "level": { "enum": ["low", "high"] }
            }
        });
        make_strict_schema_required(&mut schema);
        assert_eq!(schema["required"], json!(["level"]));
        // No "type" field to nullify -- left exactly as the caller wrote it.
        assert_eq!(
            schema["properties"]["level"],
            json!({ "enum": ["low", "high"] })
        );
    }
}
