//! Shared helpers for OpenAI-shaped providers (the Responses API and the
//! Chat Completions API, plus every `openai_chat_completions`-based
//! compatible provider).
//!
//! Both APIs' "strict" structured-output/tool-calling mode imposes two
//! requirements on every object-shaped (sub-)schema, not just the
//! top-level one:
//!
//! 1. Every key present in `properties` must also be listed in `required`
//!    -- a property that is semantically optional is instead expressed by
//!    adding `"null"` to its own `type` (see
//!    <https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required>).
//! 2. `additionalProperties` must be present and `false`.
//!
//! A caller's `schemars`-derived schema routinely violates both at nested
//! levels: a field behind `#[serde(default)]`/`Option<T>` is typically not
//! listed in `required`, and only the top-level object schema -- not array
//! item schemas or nested object schemas -- gets `additionalProperties:
//! false` set by callers that only patch the top level. OpenAI rejects the
//! request outright with HTTP 400 `invalid_function_parameters` (for
//! either violation, at any nesting depth) before the tool is ever offered
//! to the model -- the caller never even gets a chance to decide whether a
//! field was actually supplied, or to control the shape of an array
//! element.

use serde_json::Value;

/// Rewrites `schema` in place so every object-shaped (sub-)schema -- the
/// top level, every nested object under `properties`, and every array's
/// `items` schema -- satisfies both of OpenAI strict mode's requirements:
/// `required` lists every property key (with non-required ones nullified
/// in their own `type`, preserving optionality from the model's
/// perspective), and `additionalProperties` is `false`.
///
/// No-op for a non-object `schema` value, or one with no `properties`
/// object -- nothing to make required, though `additionalProperties` is
/// still not touched in that case either since there is no properties
/// shape to constrain.
pub(crate) fn make_strict_schema_required(schema: &mut Value) {
    let Value::Object(obj) = schema else { return };

    let Some(Value::Object(properties)) = obj.get("properties").cloned() else {
        return;
    };

    obj.insert("additionalProperties".to_string(), Value::Bool(false));

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
        // have its own optional sub-fields (and its own
        // `additionalProperties` requirement) independent of whether the
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

/// Returns whether `schema` -- and, recursively, every nested object/array
/// schema it contains -- can be expressed under OpenAI's strict mode at
/// all.
///
/// Every object-shaped (sub-)schema under strict mode must declare a
/// concrete `properties` map of its own keys (`additionalProperties:
/// false` is then enforced against exactly that set). A schema that
/// instead accepts arbitrary, un-enumerated keys -- e.g. an MCP tool's
/// free-form passthrough parameter, typically shaped as a bare `{"type":
/// "object"}` with no `properties` at all -- can never be rewritten to
/// satisfy that: forcing `additionalProperties: false` on it would reject
/// every key the caller could ever send, defeating the entire purpose of
/// a free-form object parameter. [`make_strict_schema_required`] must
/// only ever be applied to a schema this function reports as compatible;
/// see [`prepare_openai_tool_schema`] for the caller-facing decision this
/// feeds into.
fn schema_is_strict_compatible(schema: &Value) -> bool {
    // JSON Schema's boolean shorthand: `true` accepts any value at all,
    // including an arbitrary object with arbitrary keys -- exactly what
    // `additionalProperties: false` cannot express. `false` accepts
    // nothing, which is vacuously compatible (if odd) since no key could
    // ever violate a constraint that never applies. `schemars` emits `true`
    // for `serde_json::Value`-typed fields (a deliberately unconstrained
    // "any JSON value" passthrough), which is exactly the free-form shape
    // this whole check exists to catch.
    let Value::Object(obj) = schema else {
        return !matches!(schema, Value::Bool(true));
    };

    let is_object_shaped =
        obj.get("type").is_some_and(type_mentions_object) || obj.contains_key("properties");

    let properties_ok = match obj.get("properties") {
        Some(Value::Object(properties)) => properties.values().all(schema_is_strict_compatible),
        // `properties` present but not an object is malformed, not just
        // "free-form" -- treat it the same as "no declared properties".
        Some(_) => false,
        // An object-shaped schema with no `properties` key at all accepts
        // arbitrary keys -- exactly what `additionalProperties: false`
        // cannot express. A non-object schema (string/array/etc.) with no
        // `properties` key has nothing to enumerate in the first place, so
        // this is trivially satisfied for it.
        None => !is_object_shaped,
    };

    if !properties_ok {
        return false;
    }

    match obj.get("items") {
        Some(items) => schema_is_strict_compatible(items),
        None => true,
    }
}

fn type_mentions_object(type_value: &Value) -> bool {
    match type_value {
        Value::String(s) => s == "object",
        Value::Array(arr) => arr.iter().any(|t| t.as_str() == Some("object")),
        _ => false,
    }
}

/// Prepares a tool's raw JSON schema for an OpenAI-shaped function
/// definition and decides whether strict mode can be used for it at all.
/// Shared by both `openai` (Responses API) and `openai_chat_completions`
/// (and every provider built on it), which otherwise need the exact same
/// pipeline:
///
/// 1. Force the top-level schema to `type: "object"` with a concrete
///    (possibly empty) `properties` map -- both APIs require this of a
///    tool's parameters schema regardless of strict mode.
/// 2. Decide whether the schema can be expressed under strict mode at all
///    (see [`schema_is_strict_compatible`]'s doc for why a free-form
///    passthrough object schema can't be, at any nesting depth).
/// 3. If compatible: recursively rewrite the schema to satisfy strict
///    mode's requirements (see [`make_strict_schema_required`]) and report
///    `true`.
/// 4. If not: leave the schema as given beyond step 1's top-level
///    defaulting (harmless either way) and report `false` -- the tool
///    falls back to ordinary, non-strict tool calling, which has no
///    `required`/`additionalProperties` constraints of its own and so
///    tolerates a free-form object parameter just fine.
///
/// Returns the prepared schema and whether the caller should set
/// `strict: true` on the resulting tool definition.
pub(crate) fn prepare_openai_tool_schema(mut params: Value) -> (Value, bool) {
    if let Value::Object(ref mut obj) = params {
        obj.insert("type".to_string(), Value::String("object".to_string()));
        if !obj.get("properties").is_some_and(Value::is_object) {
            obj.insert(
                "properties".to_string(),
                Value::Object(serde_json::Map::new()),
            );
        }
    }

    if schema_is_strict_compatible(&params) {
        make_strict_schema_required(&mut params);
        (params, true)
    } else {
        (params, false)
    }
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

    /// Regression test reproducing the exact failure shape reported by a
    /// real OpenAI-shaped provider against drift's `report_findings` tool:
    /// `In context=('properties', 'findings', 'items'),
    /// 'additionalProperties' is required to be supplied and to be false.`
    /// Only the top-level schema's `additionalProperties` was being set;
    /// nested object schemas (an array's `items`, here) were left without
    /// one at all.
    #[test]
    fn sets_additional_properties_false_on_every_nested_object_schema_too() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": { "type": "string" },
                            "summary": { "type": "string" }
                        },
                        "required": ["file", "summary"]
                    }
                }
            },
            "required": ["findings"]
        });
        make_strict_schema_required(&mut schema);
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(
            schema["properties"]["findings"]["items"]["additionalProperties"],
            json!(false),
            "nested object schema (array items) must also get additionalProperties: false"
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

    // ------------------------------------------------------------------
    // schema_is_strict_compatible / prepare_openai_tool_schema
    // ------------------------------------------------------------------

    #[test]
    fn strict_compatible_for_a_fully_enumerated_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" }
            },
            "required": ["command"]
        });
        assert!(schema_is_strict_compatible(&schema));
    }

    /// Regression test reproducing the exact failure shape reported by a
    /// real OpenAI-shaped provider against drift's MCP `execute_tool`
    /// bridge: a free-form passthrough parameter shaped as a bare `{"type":
    /// "object"}` with no `properties` of its own (so any key is valid --
    /// exactly what `additionalProperties: false` cannot express).
    #[test]
    fn strict_incompatible_for_a_free_form_object_property_with_no_properties_key() {
        let schema = json!({
            "type": "object",
            "properties": {
                "server": { "type": "string" },
                "tool": { "type": "string" },
                "args": { "type": "object", "description": "Arguments to pass to the tool" }
            },
            "required": ["server", "tool"]
        });
        assert!(
            !schema_is_strict_compatible(&schema),
            "a free-form object property must make the whole schema strict-incompatible"
        );
    }

    #[test]
    fn strict_incompatible_when_the_free_form_object_is_nested_inside_an_array() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": { "type": "object" }
                }
            },
            "required": ["items"]
        });
        assert!(!schema_is_strict_compatible(&schema));
    }

    #[test]
    fn strict_compatible_for_an_object_with_an_explicitly_empty_properties_map() {
        // An object schema that declares `properties: {}` is a concrete
        // (empty) key set, not "accept anything" -- additionalProperties:
        // false against it is meaningful (it just means "no properties at
        // all"), unlike a bare `{"type": "object"}` with no properties key.
        let schema = json!({
            "type": "object",
            "properties": {
                "flags": { "type": "object", "properties": {} }
            },
            "required": ["flags"]
        });
        assert!(schema_is_strict_compatible(&schema));
    }

    #[test]
    fn prepare_tool_schema_rewrites_a_compatible_schema_and_reports_strict_true() {
        let schema = json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" }
            }
        });
        let (prepared, strict) = prepare_openai_tool_schema(schema);
        assert!(strict);
        assert_eq!(prepared["required"], json!(["command"]));
        assert_eq!(prepared["additionalProperties"], json!(false));
    }

    #[test]
    fn prepare_tool_schema_leaves_an_incompatible_schema_unrewritten_and_reports_strict_false() {
        let schema = json!({
            "type": "object",
            "properties": {
                "server": { "type": "string" },
                "args": { "type": "object" }
            },
            "required": ["server"]
        });
        let (prepared, strict) = prepare_openai_tool_schema(schema);
        assert!(
            !strict,
            "a free-form object parameter must fall back to strict: false"
        );
        // Left exactly as the caller's schema declared it -- not rewritten
        // into a shape that would itself violate strict mode if it were
        // ever accidentally sent with strict: true.
        assert_eq!(prepared["required"], json!(["server"]));
        assert_eq!(prepared["properties"]["args"], json!({ "type": "object" }));
    }

    #[test]
    fn prepare_tool_schema_always_defaults_type_and_properties_regardless_of_strictness() {
        let schema = json!({});
        let (prepared, _strict) = prepare_openai_tool_schema(schema);
        assert_eq!(prepared["type"], json!("object"));
        assert_eq!(prepared["properties"], json!({}));
    }
}
