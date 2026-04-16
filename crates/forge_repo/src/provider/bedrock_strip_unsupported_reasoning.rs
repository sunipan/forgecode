use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamInput;
use forge_domain::Transformer;

/// Strips Bedrock `thinking` request fields for models that do not support
/// Anthropic reasoning.
///
/// Bedrock reasoning parameters are currently sent via
/// `additional_model_request_fields.thinking`. This field must not be sent to
/// Claude models without thinking support (for example, Claude 3.5 Haiku) and
/// should never be sent to non-Anthropic models.
pub struct StripUnsupportedReasoning {
    model: String,
}

impl StripUnsupportedReasoning {
    /// Creates a new transformer for the target Bedrock model ID.
    ///
    /// # Arguments
    ///
    /// * `model` - Bedrock model identifier used to evaluate thinking support.
    pub fn new(model: impl Into<String>) -> Self {
        Self { model: model.into() }
    }
}

impl Transformer for StripUnsupportedReasoning {
    type Value = ConverseStreamInput;

    fn transform(&mut self, mut request: Self::Value) -> Self::Value {
        if request.additional_model_request_fields.is_some()
            && !model_supports_thinking(&self.model)
        {
            tracing::debug!(
                model = %self.model,
                "Model does not support Bedrock thinking; stripping additional_model_request_fields"
            );
            request.additional_model_request_fields = None;
        }

        request
    }
}

fn model_supports_thinking(model: &str) -> bool {
    let model = model.to_lowercase();

    if !is_anthropic_model(&model) {
        return false;
    }

    if model.contains("claude-opus-4") || model.contains("claude-sonnet-4") {
        return true;
    }

    if model.contains("claude-3-7") || model.contains("claude-3.7") {
        return true;
    }

    if (model.contains("claude-3-5") || model.contains("claude-3.5")) && !model.contains("haiku") {
        return true;
    }

    false
}

fn is_anthropic_model(model: &str) -> bool {
    model.contains("claude") || model.contains("anthropic")
}

#[cfg(test)]
mod tests {
    use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamInput;
    use forge_domain::{Context, ReasoningConfig, Transformer};
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::provider::FromDomain;

    fn reasoning_context_fixture() -> Context {
        Context {
            conversation_id: None,
            initiator: None,
            messages: vec![],
            tools: vec![],
            tool_choice: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            reasoning: Some(ReasoningConfig {
                enabled: Some(true),
                effort: None,
                max_tokens: Some(3000),
                exclude: None,
            }),
            stream: None,
            response_format: None,
        }
    }

    #[test]
    fn test_strip_reasoning_for_claude_3_5_haiku() {
        let fixture =
            ConverseStreamInput::from_domain(reasoning_context_fixture()).expect("valid context");

        assert!(fixture.additional_model_request_fields().is_some());

        let mut transformer =
            StripUnsupportedReasoning::new("us.anthropic.claude-3-5-haiku-20241022-v1:0");
        let actual = transformer.transform(fixture);
        let expected = None;

        assert_eq!(actual.additional_model_request_fields(), expected.as_ref());
    }

    #[test]
    fn test_keep_reasoning_for_claude_3_5_sonnet() {
        let fixture =
            ConverseStreamInput::from_domain(reasoning_context_fixture()).expect("valid context");

        assert!(fixture.additional_model_request_fields().is_some());

        let mut transformer =
            StripUnsupportedReasoning::new("us.anthropic.claude-3-5-sonnet-20241022-v2:0");
        let actual = transformer.transform(fixture);

        assert!(actual.additional_model_request_fields().is_some());
    }

    #[test]
    fn test_strip_reasoning_for_non_anthropic_model() {
        let fixture =
            ConverseStreamInput::from_domain(reasoning_context_fixture()).expect("valid context");

        assert!(fixture.additional_model_request_fields().is_some());

        let mut transformer = StripUnsupportedReasoning::new("amazon.nova-pro-v1:0");
        let actual = transformer.transform(fixture);
        let expected = None;

        assert_eq!(actual.additional_model_request_fields(), expected.as_ref());
    }
}
