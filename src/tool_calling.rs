//! OpenAI-style tool calling for GGUF models.
//!
//! A local model never sees a `tools` array — it sees text. So tool calling is two string
//! transforms bolted either side of generation:
//!
//! 1. **In:** the `tools` array becomes a block appended to the system prompt, in the format
//!    the model family was fine-tuned on.
//! 2. **Out:** the model's emitted call is parsed back into `tool_calls`.
//!
//! Both directions use the **Hermes** convention (`<tool_call>{...}</tool_call>`), which is
//! what Qwen2/Qwen3/Qwen3.5 and the Hermes-tuned Llama and Mistral variants are trained on.
//! The wording of the injected block is taken from Qwen3's own chat template so the model
//! sees the phrasing it was trained against rather than a paraphrase.
//!
//! ## Why the parser is lenient
//!
//! Models get this wrong constantly, and a strict parser turns a recoverable turn into a
//! dead one. Observed in the wild, all of which parse here:
//!
//! ```text
//! <tool_call>{"name": "x", "arguments": {...}}</tool_call>   the trained format
//! {"name": "x", "arguments": {...}}                          wrapper omitted entirely
//! ```json\n{"name": "x", ...}\n```                           fenced as if for a human
//! {"name": "x", "arguments": "{\"a\": 1}"}                   arguments as a JSON string
//! {"name": "x", "parameters": {...}}                         wrong key
//! ```
//!
//! Leniency stops at correctness: a call is only returned when the name matches a tool the
//! request actually declared and the arguments resolve to an object. A half-understood call
//! handed to a caller is worse than no call, because the caller will run it.

use crate::openai_compat::{FunctionCall, ToolCall, ToolSpec};

/// The instruction block appended to the system prompt.
///
/// Phrasing follows Qwen3's chat template. Do not "improve" it: the model was fine-tuned
/// against these sentences, and rewording them measurably degrades call formatting.
pub fn render_tools_block(tools: &[ToolSpec]) -> String {
    let mut s = String::from(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for t in tools {
        let f = &t.function;
        let entry = serde_json::json!({
            "type": t.tool_type,
            "function": {
                "name": f.name,
                "description": f.description.clone().unwrap_or_default(),
                "parameters": f.parameters.clone().unwrap_or(serde_json::json!({})),
            }
        });
        s.push_str(&entry.to_string());
        s.push('\n');
    }
    s.push_str(
        "</tools>\n\nFor each function call, return a json object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    s
}

/// Everything the model emitted that is a valid call to a declared tool.
///
/// Returns an empty vec when the output is ordinary prose — the caller then treats the turn
/// as a normal completion, which is the common case and must stay cheap.
pub fn parse_tool_calls(output: &str, declared: &[ToolSpec]) -> Vec<ToolCall> {
    let names: Vec<&str> = declared.iter().map(|t| t.function.name.as_str()).collect();
    let mut out = Vec::new();

    for (i, candidate) in candidates(output).into_iter().enumerate() {
        let Some((name, args)) = parse_one(&candidate, &names) else { continue };
        out.push(ToolCall {
            // Stable within a response and unique across it, which is all a tool_call_id has
            // to be for the client to match results back to calls.
            id: format!("call_{i}"),
            call_type: "function".into(),
            function: FunctionCall { name, arguments: args },
        });
    }
    out
}

/// Candidate JSON blobs, most-trusted source first.
fn candidates(output: &str) -> Vec<String> {
    let mut found = Vec::new();

    // 1. Properly wrapped calls. There may be several.
    let mut rest = output;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let end = after.find("</tool_call>").unwrap_or(after.len());
        found.push(after[..end].trim().to_string());
        rest = &after[end.min(after.len())..];
    }
    if !found.is_empty() {
        return found;
    }

    // 2. No wrapper: take the outermost brace span, after stripping any code fence. A model
    //    that forgot the tags usually got the JSON itself right.
    let cleaned = output.replace("```json", " ").replace("```", " ");
    if let (Some(a), Some(b)) = (cleaned.find('{'), cleaned.rfind('}')) {
        if b > a {
            found.push(cleaned[a..=b].to_string());
        }
    }
    found
}

/// One candidate to `(name, arguments-as-json-string)`, or nothing.
fn parse_one(candidate: &str, declared: &[&str]) -> Option<(String, String)> {
    let v: serde_json::Value = serde_json::from_str(candidate).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    if !declared.contains(&name.as_str()) {
        return None;
    }

    let raw = v.get("arguments").or_else(|| v.get("parameters"));
    let args = match raw {
        Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map.clone()),
        // Some models imitate the wire format and send arguments already stringified.
        Some(serde_json::Value::String(s)) => serde_json::from_str(s).ok()?,
        // A call with no arguments at all is legitimate for a zero-arg tool.
        None => serde_json::json!({}),
        _ => return None,
    };
    args.is_object().then(|| (name, args.to_string()))
}

/// Whether the request asked for tools to be suppressed entirely.
pub fn tools_disabled(tool_choice: Option<&serde_json::Value>) -> bool {
    matches!(tool_choice.and_then(|v| v.as_str()), Some("none"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_compat::FunctionSpec;

    fn tools() -> Vec<ToolSpec> {
        vec![ToolSpec {
            tool_type: "function".into(),
            function: FunctionSpec {
                name: "get_weather".into(),
                description: Some("Current weather for a city".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                })),
            },
        }]
    }

    #[test]
    fn the_block_carries_the_signature_the_model_needs() {
        let block = render_tools_block(&tools());
        assert!(block.contains("<tools>") && block.contains("</tools>"));
        assert!(block.contains("get_weather"));
        assert!(block.contains("Current weather for a city"));
        // The schema must survive intact or the model invents argument names.
        assert!(block.contains("\"city\""));
        assert!(block.contains("<tool_call>"));
    }

    #[test]
    fn a_properly_wrapped_call_parses() {
        let out = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Berlin"}}
</tool_call>"#;
        let calls = parse_tool_calls(out, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Berlin"}"#);
        assert_eq!(calls[0].call_type, "function");
    }

    /// The failure that motivated all of this: the model emits the JSON and forgets the
    /// tags, the server sees prose, and the call silently never runs.
    #[test]
    fn a_call_missing_its_wrapper_still_parses() {
        let out = r#"{"name": "get_weather", "arguments": {"city": "Cairo"}}"#;
        let calls = parse_tool_calls(out, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, r#"{"city":"Cairo"}"#);
    }

    #[test]
    fn fenced_and_stringified_and_misnamed_variants_all_parse() {
        let cases = [
            "```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n```",
            r#"{"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}"#,
            r#"{"name": "get_weather", "parameters": {"city": "Oslo"}}"#,
        ];
        for c in cases {
            let calls = parse_tool_calls(c, &tools());
            assert_eq!(calls.len(), 1, "failed on: {c}");
            assert_eq!(calls[0].function.arguments, r#"{"city":"Oslo"}"#);
        }
    }

    #[test]
    fn several_calls_in_one_reply_are_all_returned() {
        let out = r#"<tool_call>{"name":"get_weather","arguments":{"city":"A"}}</tool_call>
<tool_call>{"name":"get_weather","arguments":{"city":"B"}}</tool_call>"#;
        let calls = parse_tool_calls(out, &tools());
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id, "ids must distinguish the calls");
    }

    /// Leniency must not become gullibility: these would be handed to a caller that runs
    /// them, so each has to be refused.
    #[test]
    fn nothing_undeclared_or_malformed_is_returned() {
        for bad in [
            r#"{"name": "rm_rf", "arguments": {"path": "/"}}"#,     // never declared
            r#"{"name": "get_weather", "arguments": "not json"}"#,   // unparseable
            r#"{"name": "get_weather", "arguments": ["Berlin"]}"#,   // not an object
            "The get_weather tool takes a city.",                    // prose about a tool
            "",
        ] {
            assert!(parse_tool_calls(bad, &tools()).is_empty(), "must refuse: {bad}");
        }
    }

    /// A zero-argument tool is a real thing, and `{}` is the right arguments for it.
    #[test]
    fn a_call_with_no_arguments_is_valid() {
        let calls = parse_tool_calls(r#"{"name": "get_weather"}"#, &tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn ordinary_prose_costs_nothing_and_returns_nothing() {
        assert!(parse_tool_calls("The weather in Berlin is mild today.", &tools()).is_empty());
    }

    #[test]
    fn tool_choice_none_suppresses_the_block() {
        assert!(tools_disabled(Some(&serde_json::json!("none"))));
        assert!(!tools_disabled(Some(&serde_json::json!("auto"))));
        assert!(!tools_disabled(None));
    }
}
