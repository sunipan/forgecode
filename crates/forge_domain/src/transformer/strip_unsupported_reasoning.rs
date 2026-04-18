use crate::{Context, ModelId, Transformer};

/// Strips unsupported Anthropic-family reasoning configuration from a
/// [`Context`] before provider-specific request serialization.
///
/// Anthropic-family models differ in which reasoning controls they accept:
/// some reject reasoning entirely, some allow thinking but not effort-based
/// configuration, and newer generations allow both. This transformer applies a
/// model-aware normalization step at the context layer so providers do not need
/// to repeat the same stripping logic.
///
/// For non-Anthropic models this transformer is a no-op.
///
/// The normalization rules are:
/// - models without thinking support drop `context.reasoning` entirely
/// - models with thinking support but without effort support keep reasoning and
///   strip only `reasoning.effort`
/// - models that support both keep the reasoning config unchanged
///
/// Supported Anthropic-family behavior:
/// - `claude-3-haiku`, `claude-3-sonnet`, and `claude-3-opus` support neither
///   thinking nor effort
/// - `claude-3-5-sonnet` and `claude-3-7-sonnet` support thinking but not
///   effort
/// - `claude-3-5-haiku` supports neither thinking nor effort
/// - `claude-haiku-4-5` supports thinking but not effort
/// - `claude-sonnet-4*` and `claude-opus-4*` support both thinking and effort
pub struct StripUnsupportedReasoning {
    model_id: ModelId,
}

impl StripUnsupportedReasoning {
    /// Creates a transformer for the given model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The target model identifier used to decide which
    ///   reasoning fields are supported.
    pub fn new(model_id: ModelId) -> Self {
        Self { model_id }
    }
}

impl Transformer for StripUnsupportedReasoning {
    type Value = Context;

    fn transform(&mut self, mut context: Self::Value) -> Self::Value {
        let model = self.model_id.as_str();

        if !is_anthropic_model(model) {
            return context;
        }

        let Some(reasoning) = context.reasoning.as_mut() else {
            return context;
        };

        if !model_supports_thinking(model) {
            tracing::debug!(
                model = %self.model_id,
                "Model does not support Anthropic thinking; stripping reasoning configuration"
            );
            context.reasoning = None;
            return context;
        }

        if reasoning.effort.is_some() && !model_supports_effort(model) {
            tracing::debug!(
                model = %self.model_id,
                "Model does not support Anthropic effort; stripping effort configuration"
            );
            reasoning.effort = None;
        }

        context
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

    if model.contains("claude-haiku-4") {
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
    let model = model.to_lowercase();

    if model.contains("claude-haiku-4-5") {
        return false;
    }

    model.contains("claude-opus-4") || model.contains("claude-sonnet-4")
}

fn is_anthropic_model(model: &str) -> bool {
    let model = model.to_lowercase();

    model.contains("claude") || model.contains("anthropic")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{Effort, ReasoningConfig};

    #[test]
    fn test_strip_reasoning_for_claude_3_5_haiku() {
        let fixture = Context::default().reasoning(
            ReasoningConfig::default()
                .enabled(true)
                .max_tokens(3000)
                .effort(Effort::High),
        );
        let mut transformer =
            StripUnsupportedReasoning::new(ModelId::from("claude-3-5-haiku-20241022"));

        let actual = transformer.transform(fixture);
        let expected = Context::default();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_strip_reasoning_for_bedrock_claude_3_5_haiku() {
        let fixture =
            Context::default().reasoning(ReasoningConfig::default().enabled(true).max_tokens(3000));
        let mut transformer = StripUnsupportedReasoning::new(ModelId::from(
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ));

        let actual = transformer.transform(fixture);
        let expected = Context::default();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_strip_reasoning_for_openai_anthropic_claude_3_5_haiku() {
        let fixture =
            Context::default().reasoning(ReasoningConfig::default().enabled(true).max_tokens(3000));
        let mut transformer =
            StripUnsupportedReasoning::new(ModelId::from("anthropic/claude-3-5-haiku-20241022"));

        let actual = transformer.transform(fixture);
        let expected = Context::default();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_strip_effort_for_openai_anthropic_claude_3_7_sonnet() {
        let fixture = Context::default().reasoning(
            ReasoningConfig::default()
                .enabled(true)
                .max_tokens(3000)
                .effort(Effort::High),
        );
        let mut transformer =
            StripUnsupportedReasoning::new(ModelId::from("anthropic/claude-3-7-sonnet-20250219"));

        let actual = transformer.transform(fixture);
        let expected =
            Context::default().reasoning(ReasoningConfig::default().enabled(true).max_tokens(3000));

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_strip_effort_for_claude_3_7_sonnet() {
        let fixture = Context::default().reasoning(
            ReasoningConfig::default()
                .enabled(true)
                .effort(Effort::High),
        );
        let mut transformer =
            StripUnsupportedReasoning::new(ModelId::from("claude-3-7-sonnet-20250219"));

        let actual = transformer.transform(fixture);
        let expected = Context::default().reasoning(ReasoningConfig::default().enabled(true));

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_keep_effort_for_claude_sonnet_4_5() {
        let fixture = Context::default().reasoning(
            ReasoningConfig::default()
                .enabled(true)
                .effort(Effort::High),
        );
        let mut transformer = StripUnsupportedReasoning::new(ModelId::from("claude-sonnet-4-5"));

        let actual = transformer.transform(fixture.clone());
        let expected = fixture;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_keep_reasoning_for_claude_haiku_4_5() {
        let fixture = Context::default().reasoning(
            ReasoningConfig::default()
                .enabled(true)
                .max_tokens(3000)
                .effort(Effort::High),
        );
        let mut transformer =
            StripUnsupportedReasoning::new(ModelId::from("claude-haiku-4-5-20251001"));

        let actual = transformer.transform(fixture);
        let expected =
            Context::default().reasoning(ReasoningConfig::default().enabled(true).max_tokens(3000));

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_keep_effort_for_claude_sonnet_4() {
        let fixture = Context::default().reasoning(
            ReasoningConfig::default()
                .enabled(true)
                .max_tokens(3000)
                .effort(Effort::High),
        );
        let mut transformer =
            StripUnsupportedReasoning::new(ModelId::from("claude-sonnet-4-20250514"));

        let actual = transformer.transform(fixture.clone());
        let expected = fixture;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_preserve_reasoning_for_non_anthropic_models() {
        let fixture = Context::default().reasoning(
            ReasoningConfig::default()
                .enabled(true)
                .max_tokens(3000)
                .effort(Effort::High),
        );
        let mut transformer = StripUnsupportedReasoning::new(ModelId::from("gpt-5"));

        let actual = transformer.transform(fixture.clone());
        let expected = fixture;

        assert_eq!(actual, expected);
    }
}
