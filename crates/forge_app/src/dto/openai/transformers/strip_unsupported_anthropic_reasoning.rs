use forge_domain::Transformer;

use crate::dto::openai::Request;

/// Strips Anthropic reasoning parameters that are unsupported by the selected
/// model when sending requests through OpenAI-compatible providers.
///
/// This transformer protects providers such as OpenRouter, Requesty, and
/// GitHub Copilot when they proxy Anthropic models. Some Claude models (for
/// example, Claude 3.5 Haiku) reject effort-style reasoning parameters.
pub struct StripUnsupportedAnthropicReasoning;

impl Transformer for StripUnsupportedAnthropicReasoning {
    type Value = Request;

    fn transform(&mut self, mut request: Self::Value) -> Self::Value {
        let Some(model) = request
            .model
            .as_ref()
            .map(|model| model.as_str().to_lowercase())
        else {
            return request;
        };

        if !is_anthropic_model(&model) {
            return request;
        }

        if !model_supports_effort(&model) {
            if request.reasoning_effort.is_some() {
                tracing::debug!(
                    model = %model,
                    "Model does not support effort; stripping reasoning_effort"
                );
            }
            request.reasoning_effort = None;

            if let Some(reasoning) = request.reasoning.as_mut() {
                reasoning.effort = None;
            }
        }

        if !model_supports_thinking(&model) {
            if request.reasoning.is_some() {
                tracing::debug!(
                    model = %model,
                    "Model does not support thinking; stripping reasoning"
                );
            }
            request.reasoning = None;
        }

        request
    }
}

fn is_anthropic_model(model: &str) -> bool {
    model.contains("claude") || model.contains("anthropic")
}

fn model_supports_thinking(model: &str) -> bool {
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

fn model_supports_effort(model: &str) -> bool {
    model.contains("claude-opus-4") || model.contains("claude-sonnet-4")
}

#[cfg(test)]
mod tests {
    use forge_domain::{Effort, ModelId, ReasoningConfig, Transformer};
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_non_anthropic_model_preserves_reasoning_fields() {
        let fixture = Request::default()
            .model(ModelId::new("openai/gpt-4o"))
            .reasoning(ReasoningConfig {
                enabled: Some(true),
                effort: Some(Effort::High),
                max_tokens: Some(2048),
                exclude: None,
            })
            .reasoning_effort("high".to_string());

        let mut transformer = StripUnsupportedAnthropicReasoning;
        let actual = transformer.transform(fixture.clone());
        let expected = fixture;

        assert_eq!(actual.reasoning, expected.reasoning);
        assert_eq!(actual.reasoning_effort, expected.reasoning_effort);
    }

    #[test]
    fn test_claude_3_5_haiku_strips_reasoning_and_effort() {
        let fixture = Request::default()
            .model(ModelId::new("anthropic/claude-3-5-haiku-20241022"))
            .reasoning(ReasoningConfig {
                enabled: Some(true),
                effort: Some(Effort::High),
                max_tokens: Some(2048),
                exclude: None,
            })
            .reasoning_effort("high".to_string());

        let mut transformer = StripUnsupportedAnthropicReasoning;
        let actual = transformer.transform(fixture);

        let expected_reasoning: Option<ReasoningConfig> = None;
        let expected_reasoning_effort: Option<String> = None;
        assert_eq!(actual.reasoning, expected_reasoning);
        assert_eq!(actual.reasoning_effort, expected_reasoning_effort);
    }

    #[test]
    fn test_claude_3_7_sonnet_keeps_thinking_budget_but_strips_effort() {
        let fixture = Request::default()
            .model(ModelId::new("anthropic/claude-3-7-sonnet-20250219"))
            .reasoning(ReasoningConfig {
                enabled: Some(true),
                effort: Some(Effort::Medium),
                max_tokens: Some(4096),
                exclude: None,
            })
            .reasoning_effort("medium".to_string());

        let mut transformer = StripUnsupportedAnthropicReasoning;
        let actual = transformer.transform(fixture);

        let expected_reasoning = Some(ReasoningConfig {
            enabled: Some(true),
            effort: None,
            max_tokens: Some(4096),
            exclude: None,
        });
        let expected_reasoning_effort: Option<String> = None;
        assert_eq!(actual.reasoning, expected_reasoning);
        assert_eq!(actual.reasoning_effort, expected_reasoning_effort);
    }

    #[test]
    fn test_claude_sonnet_4_keeps_effort() {
        let fixture = Request::default()
            .model(ModelId::new("anthropic/claude-sonnet-4-20250514"))
            .reasoning(ReasoningConfig {
                enabled: Some(true),
                effort: Some(Effort::High),
                max_tokens: Some(4096),
                exclude: None,
            })
            .reasoning_effort("high".to_string());

        let mut transformer = StripUnsupportedAnthropicReasoning;
        let actual = transformer.transform(fixture.clone());
        let expected = fixture;

        assert_eq!(actual.reasoning, expected.reasoning);
        assert_eq!(actual.reasoning_effort, expected.reasoning_effort);
    }
}
