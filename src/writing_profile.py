from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import CLAUDE_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY
from .models import get_model_info
from .pricing import estimate_cost, estimate_token_usage


# ============================================================================
# Pydantic Models for Structured Output - Academic Writing Focus
# ============================================================================


class HedgingPattern(BaseModel):
    """Hedging and epistemic markers that signal certainty levels."""

    modal_verbs: List[str] = Field(
        description="Modal auxiliaries used (may, might, could, should, etc.)",
        max_length=8,
    )
    epistemic_verbs: List[str] = Field(
        description="Verbs that express uncertainty (suggest, indicate, appear, seem, etc.)",
        max_length=8,
    )
    probability_markers: List[str] = Field(
        description="Adjectives/adverbs showing probability (possibly, likely, perhaps, generally, etc.)",
        max_length=8,
    )
    hedging_frequency: str = Field(
        description="How often hedging appears (e.g., '1 per 2-3 sentences', 'frequent in claims, rare in methods')"
    )
    hedging_patterns: List[str] = Field(
        description="Abstract patterns describing HOW hedging is used, not specific content (e.g., 'hedges claims with modals before stating findings', 'uses epistemic verbs to soften interpretations')",
        max_length=3,
    )


class FormalityProfile(BaseModel):
    """Formality level and academic register characteristics."""

    overall_level: str = Field(
        description="Overall formality: 'high academic', 'moderate academic', 'accessible academic', 'conversational academic'"
    )
    nominalization_frequency: str = Field(
        description="Use of noun forms (e.g., 'the analysis of' vs 'analyzing'): high | moderate | low"
    )
    personal_pronoun_usage: str = Field(
        description="First-person usage patterns (e.g., 'never', 'we only', 'I in discussion', 'active first-person throughout')"
    )
    passive_voice_ratio: str = Field(
        description="Estimated passive voice usage (e.g., '20-30% in methods, rare elsewhere', '40-50% throughout')"
    )
    contraction_usage: str = Field(
        description="Use of contractions: 'never' | 'rare' | 'occasional' | 'frequent'"
    )
    formality_patterns: List[str] = Field(
        description="General patterns that characterize formality level (e.g., 'prefers impersonal constructions', 'uses Latinate vocabulary over Anglo-Saxon', 'avoids colloquial expressions')",
        max_length=3,
    )


class VocabularyCharacteristics(BaseModel):
    """Word choice patterns and lexical preferences."""

    technical_density: str = Field(
        description="Frequency of discipline-specific terminology: high | moderate | low"
    )
    lexical_density: str = Field(
        description="Ratio of content words to total words (academic typically 50-60%): e.g., '~55%'"
    )
    word_choice_patterns: List[str] = Field(
        description="General patterns describing vocabulary preferences (e.g., 'prefers action verbs over be-verbs', 'uses process nouns frequently', 'favors precise technical terms over general language')",
        max_length=5,
    )
    register_consistency: str = Field(
        description="Whether vocabulary stays consistently formal or mixes registers"
    )
    vocabulary_range: str = Field(
        description="Breadth of vocabulary: 'narrow and specialized' | 'moderately varied' | 'wide-ranging' | 'deliberately simple'"
    )
    word_length_preference: str = Field(
        description="Typical word complexity: 'short words (1-2 syllables dominant)' | 'mixed' | 'polysyllabic preference'"
    )


class SentenceStructureProfile(BaseModel):
    """Syntactic patterns and sentence construction preferences."""

    average_length: str = Field(
        description="Typical sentence length in words (e.g., '15-20', '25-35', 'highly varied 10-40')"
    )
    length_variation: str = Field(
        description="Pattern of sentence length variation: 'consistent' | 'moderately varied' | 'highly varied'"
    )
    complexity_level: str = Field(
        description="Syntactic complexity: 'simple declaratives' | 'compound structures' | 'complex with subordination' | 'highly complex with multiple clauses'"
    )
    clause_structure_preferences: List[str] = Field(
        description="General patterns describing clause usage (e.g., 'frequent subordinate clauses at sentence start', 'right-branching with multiple relative clauses', 'coordinate structures with parallel elements')",
        max_length=3,
    )
    syntactic_patterns: List[str] = Field(
        description="Characteristic syntactic habits (e.g., 'prefers fronted adverbials', 'uses cleft sentences for emphasis', 'avoids sentence-initial conjunctions')",
        max_length=3,
    )


class PunctuationStyle(BaseModel):
    """Punctuation preferences and stylistic choices."""

    comma_density: str = Field(
        description="Comma usage frequency: 'minimal' | 'moderate' | 'heavy' | 'oxford comma consistent'"
    )
    semicolon_usage: str = Field(
        description="Semicolon frequency and purpose: 'never' | 'rare for complex lists' | 'frequent for clause separation'"
    )
    dash_preferences: str = Field(
        description="Use of em-dashes, en-dashes, hyphens: usage patterns and frequency"
    )
    parenthetical_style: str = Field(
        description="Preference for parentheses, em-dashes, or commas for asides"
    )
    punctuation_patterns: List[str] = Field(
        description="General punctuation habits (e.g., 'uses colons to introduce explanations', 'avoids exclamation marks entirely', 'employs serial commas consistently')",
        max_length=3,
    )


class DiscourseOrganization(BaseModel):
    """Paragraph-level and document-level structural patterns."""

    paragraph_length: str = Field(
        description="Typical paragraph length: 'short (3-5 sentences)' | 'medium (5-8)' | 'long (8-12+)'"
    )
    topic_sentence_style: str = Field(
        description="How paragraphs begin: 'clear topic sentences' | 'bridge transitions' | 'thematic progression'"
    )
    transition_types: List[str] = Field(
        description="Categories of transitions used (e.g., 'contrastive', 'additive', 'causal', 'temporal', 'enumerative')",
        max_length=5,
    )
    signposting_frequency: str = Field(
        description="Use of metadiscourse markers ('first', 'this section', 'as shown above'): high | moderate | minimal"
    )
    argument_structure: str = Field(
        description="How arguments develop: 'claim-evidence-warrant' | 'inductive progression' | 'deductive reasoning' | 'comparative analysis'"
    )
    cohesion_strategies: List[str] = Field(
        description="Methods for creating flow (e.g., 'lexical chains with synonyms', 'demonstrative pronouns referring back', 'parallel grammatical structures')",
        max_length=3,
    )


class CitationIntegrationStyle(BaseModel):
    """How sources are incorporated and referenced."""

    integration_preference: str = Field(
        description="How citations appear: 'integral (Smith argues)' | 'non-integral (...Smith, 2020)' | 'mixed'"
    )
    quotation_frequency: str = Field(
        description="Direct quotes usage: 'frequent' | 'moderate' | 'rare, paraphrasing preferred'"
    )
    source_density: str = Field(
        description="Citations per paragraph: 'heavy (4-6)' | 'moderate (2-3)' | 'light (0-1)'"
    )
    attribution_style: List[str] = Field(
        description="Patterns for introducing sources (e.g., 'names authors as agents in sentence', 'buries citations in parentheticals', 'clusters multiple citations together')",
        max_length=3,
    )


class RhetoricalDevices(BaseModel):
    """Persuasive and stylistic techniques employed."""

    devices_used: List[str] = Field(
        description="Literary/rhetorical devices present (e.g., 'analogy', 'rhetorical questions', 'parallel structure', 'metaphor', 'enumeration')",
        max_length=8,
    )
    device_frequency: str = Field(
        description="How often rhetorical devices appear: 'sparse' | 'moderate' | 'frequent'"
    )
    persuasive_strategy: str = Field(
        description="Overall rhetorical approach (e.g., 'appeals to logic and evidence', 'builds ethos through careful hedging', 'uses concrete examples', 'employs emotional resonance')"
    )
    emphasis_techniques: List[str] = Field(
        description="Methods for creating emphasis (e.g., 'repetition of key terms', 'strategic sentence fragments', 'italics for stress')",
        max_length=3,
    )


class ToneAndVoice(BaseModel):
    """Overall tone, attitude, and authorial presence."""

    primary_tone: str = Field(
        description="Dominant tone: 'objective/neutral' | 'authoritative' | 'analytical' | 'exploratory' | 'critical' | 'enthusiastic'"
    )
    secondary_tones: List[str] = Field(
        description="Secondary tonal qualities that appear",
        max_length=3,
    )
    authorial_presence: str = Field(
        description="How visible the author is: 'invisible/objective' | 'present but restrained' | 'clearly present' | 'personal and engaged'"
    )
    stance_indicators: List[str] = Field(
        description="Ways the author reveals attitude (e.g., 'evaluative adjectives', 'stance adverbs', 'attitudinal verbs', 'rhetorical questions')",
        max_length=3,
    )
    confidence_level: str = Field(
        description="Overall certainty in claims: 'highly confident' | 'appropriately hedged' | 'tentative' | 'varies by section'"
    )
    emotional_register: str = Field(
        description="Emotional quality: 'detached and clinical' | 'measured and professional' | 'engaged but controlled' | 'passionate and expressive'"
    )


class DisciplineConventions(BaseModel):
    """Field-specific norms and expectations."""

    primary_discipline: str = Field(description="The academic field this writing belongs to (e.g., 'psychology', 'literary studies', 'computer science')")
    structure_preference: str = Field(
        description="Dominant organization: 'IMRAD' | 'thesis-driven' | 'problem-solution' | 'chronological' | 'thematic'"
    )
    evidence_types: List[str] = Field(
        description="What counts as evidence (e.g., 'empirical data', 'textual analysis', 'case studies', 'theoretical frameworks')",
        max_length=5,
    )
    methodology_visibility: str = Field(
        description="How methods are discussed: 'detailed methods section' | 'integrated throughout' | 'minimal methods description'"
    )
    field_specific_conventions: List[str] = Field(
        description="Discipline-specific patterns (e.g., 'passive voice in methods', 'extensive literature review', 'theoretical framework first')",
        max_length=5,
    )


class AcademicWritingProfile(BaseModel):
    """Complete academic writing style profile for LLM-based imitation."""

    hedging: HedgingPattern
    formality: FormalityProfile
    vocabulary: VocabularyCharacteristics
    sentence_structure: SentenceStructureProfile
    punctuation: PunctuationStyle
    discourse: DiscourseOrganization
    citations: CitationIntegrationStyle
    rhetoric: RhetoricalDevices
    tone_voice: ToneAndVoice
    discipline: DisciplineConventions
    profile_summary: str = Field(description="2-3 sentence summary capturing the essence of this writing style")
    distinguishing_features: List[str] = Field(
        description="Most distinctive characteristics that set this style apart",
        max_length=5,
    )


# ============================================================================
# System Prompt for Academic Writing Profile Extraction
# ============================================================================

ACADEMIC_WRITING_PROFILE_SYSTEM_PROMPT = """
You are an expert linguistic analyst specializing in academic writing style extraction. Your task is to analyze the provided writing sample(s) and extract a comprehensive style profile that can be used to guide LLMs in imitating this author's writing style for academic tasks.

# CRITICAL INSTRUCTIONS

## 1. Extract Style, NOT Content
- **DO NOT** describe what the text is about (subject matter, topics, arguments)
- **DO** describe HOW the text is written (linguistic patterns, structural choices, stylistic habits)
- Focus on patterns that would remain consistent if the author wrote about a completely different topic
- Example: ❌ "Uses machine learning terminology" → ✅ "Uses 15-20 technical terms per paragraph with disciplinary precision"

## 2. Be Specific and Evidence-Based
- Every observation must be grounded in actual patterns from the text
- Provide VERBATIM examples wherever requested
- Quantify patterns where possible ("1 hedge per 2-3 sentences" not "frequent hedging")
- Include both positive patterns (what the author DOES) and negative patterns (what they AVOID)

## 3. Focus on Features LLMs Can Replicate
Prioritize extracting:
- **Hedging patterns** - modals, epistemic verbs, probability markers (highest impact for academic writing)
- **Formality markers** - nominalization, passive voice, pronoun usage, register consistency
- **Vocabulary patterns** - preferred terms, avoided terms, technical vocabulary, lexical patterns
- **Sentence structure** - length distributions, complexity patterns, syntactic preferences
- **Punctuation habits** - comma density, semicolon usage, parenthetical preferences
- **Discourse organization** - paragraph structure, transitions, signposting, argument flow
- **Citation integration** - how sources are introduced, quoted vs. paraphrased, attribution patterns
- **Rhetorical strategies** - persuasive devices, metadiscourse, stance-taking
- **Tone and voice** - authorial presence, confidence levels, attitude markers

## 4. Academic Writing Specificity
Consider discipline-specific conventions:
- STEM fields: IMRAD structure, passive voice in methods, high hedge density in discussion, non-integral citations
- Humanities: thesis-driven, integral citations, direct quotations, active voice, personal engagement
- Social sciences: mixed approaches, methodological transparency, theory-driven frameworks

Capture variation by section if present:
- Methods sections: procedural, passive voice, technical precision
- Discussion sections: high hedging density, interpretive language, future-oriented
- Literature reviews: synthesis patterns, critical stance, citation density

## 5. Multi-Level Analysis
Extract patterns at each linguistic level:
- **Word level**: vocabulary choices, technical terminology, hedging devices, avoided words
- **Sentence level**: length, complexity, voice, clause patterns, punctuation
- **Paragraph level**: length, topic sentences, transitions, cohesion
- **Document level**: overall structure, argument progression, metadiscourse, citation patterns

## 6. Avoid Common Pitfalls
- ❌ Don't list generic academic features (e.g., "uses formal language") without specific evidence
- ❌ Don't conflate the author's field with their style (field = context, style = how they write within it)
- ❌ Don't include observations based on just one sentence (patterns require multiple examples)
- ❌ Don't describe the argument/thesis/findings—only describe stylistic patterns
- ❌ Don't be vague ("sometimes," "often") when you can quantify ("~30% of sentences," "2-3 per paragraph")

## 7. Output Format
Provide ONLY the completed JSON structure following the AcademicWritingProfile schema. No additional commentary, no explanations, no markdown formatting—just the raw JSON object that matches the schema exactly.

List fields should contain items based on what is actually present in the text. Return as many items as are relevant, up to the maximum specified:
- hedging.modal_verbs: up to 8 items
- hedging.epistemic_verbs: up to 8 items
- hedging.probability_markers: up to 8 items
- hedging.hedging_patterns: up to 3 items
- formality.formality_patterns: up to 3 items
- vocabulary.word_choice_patterns: up to 5 items
- sentence_structure.clause_structure_preferences: up to 3 items
- sentence_structure.syntactic_patterns: up to 3 items
- punctuation.punctuation_patterns: up to 3 items
- discourse.transition_types: up to 5 items
- discourse.cohesion_strategies: up to 3 items
- citations.attribution_style: up to 3 items
- rhetoric.devices_used: up to 8 items
- rhetoric.emphasis_techniques: up to 3 items
- tone_voice.secondary_tones: up to 3 items
- tone_voice.stance_indicators: up to 3 items
- discipline.evidence_types: up to 5 items
- discipline.field_specific_conventions: up to 5 items
- distinguishing_features: up to 5 items

# ANALYSIS WORKFLOW

1. **Initial Read**: Understand the overall writing style without focusing on content
2. **Hedging Analysis**: Identify all hedging devices (modals, epistemic verbs, probability markers) and their frequency
3. **Formality Assessment**: Evaluate nominalization, passive voice, personal pronouns, contractions
4. **Vocabulary Profiling**: Extract preferred terms, avoided terms, technical vocabulary, lexical density
5. **Syntactic Patterns**: Analyze sentence length, complexity, clause structures, syntactic preferences
6. **Punctuation Habits**: Document comma usage, semicolons, dashes, parentheticals with examples
7. **Discourse Structure**: Examine paragraph organization, transitions, signposting, cohesion
8. **Citation Practices**: Note integration style, quotation frequency, attribution patterns
9. **Rhetorical Analysis**: Identify persuasive devices, metadiscourse, rhetorical strategies
10. **Tone/Voice Extraction**: Determine authorial presence, confidence, stance markers
11. **Discipline Conventions**: Infer field-specific norms and structural expectations
12. **Profile Synthesis**: Summarize distinguishing features and overall style essence

Remember: You are creating a REUSABLE style specification that will guide an LLM to write NEW content in THIS style. Every element you extract should be actionable for generation, not just descriptive of the sample.
""".strip()


# ============================================================================
# Prompt Building
# ============================================================================

def _format_reasoning_directive(options: Dict[str, Any]) -> str:
    segments: List[str] = []
    effort = options.get("reasoning_effort")
    if effort and effort != "none":
        segments.append(f"- Target reasoning effort: {effort}")
    thinking = options.get("thinking_mode")
    if thinking:
        segments.append(f"- Thinking time preference: {thinking}")
    if options.get("deep_think"):
        segments.append("- Enable extended Deep Think style reasoning.")
    budget = options.get("thinking_budget")
    if budget:
        segments.append(f"- Allocate roughly {budget} tokens of thinking budget.")
    if not segments:
        return ""
    joined = "\n".join(segments)
    return f"Reasoning preferences:\n{joined}\n"


def build_academic_profile_user_prompt(
    sample_count: int = 1,
    total_words: int = 0,
    reasoning_enabled: bool = False,
    reasoning_directive: str = "",
) -> str:
    """Build user prompt for academic writing profile extraction."""

    word_info = f" (approximately {total_words:,} words)" if total_words > 0 else ""
    sample_plural = "samples" if sample_count > 1 else "sample"

    base_prompt = (
        f"Analyze the attached writing {sample_plural}{word_info} and extract a comprehensive academic writing style profile.\n\n"
        "Focus on identifying patterns that characterize HOW this author writes, not WHAT they write about. "
        "Extract specific, quantifiable patterns at word, sentence, paragraph, and document levels.\n\n"
        "Your analysis should enable an LLM to generate new academic content that matches this author's distinctive style across different topics."
    )

    if reasoning_enabled:
        base_prompt += (
            "\n\nTake your time to thoroughly analyze the text. Consider:\n"
            "- Reading through the sample(s) multiple times to identify consistent patterns\n"
            "- Comparing different sections to distinguish style from content\n"
            "- Quantifying patterns wherever possible rather than using vague descriptors\n"
            "- Gathering specific verbatim examples for every observation\n"
            "- Ensuring all extracted features are replicable and topic-independent"
        )

    if reasoning_directive:
        base_prompt += f"\n\n{reasoning_directive.strip()}"

    return base_prompt.strip()


# ============================================================================
# Profile to Prompt Converter (for using the profile in generation)
# ============================================================================


def convert_profile_to_generation_prompt(profile: AcademicWritingProfile) -> str:
    """Convert an extracted writing profile into a system prompt for content generation."""

    prompt_parts = [
        "# WRITING STYLE INSTRUCTIONS",
        "",
        profile.profile_summary,
        "",
        "## Core Style Requirements",
        "",
    ]

    prompt_parts.extend(
        [
            "**Hedging and Certainty:**",
            f"- Use modal verbs: {', '.join(profile.hedging.modal_verbs[:5])}",
            f"- Use epistemic verbs: {', '.join(profile.hedging.epistemic_verbs[:5])}",
            f"- Include probability markers: {', '.join(profile.hedging.probability_markers[:5])}",
            f"- Hedging frequency: {profile.hedging.hedging_frequency}",
            f"- Hedging patterns: {'; '.join(profile.hedging.hedging_patterns)}",
            "",
        ]
    )

    prompt_parts.extend(
        [
            "**Formality and Register:**",
            f"- Overall level: {profile.formality.overall_level}",
            f"- Nominalization: {profile.formality.nominalization_frequency}",
            f"- Personal pronouns: {profile.formality.personal_pronoun_usage}",
            f"- Passive voice: {profile.formality.passive_voice_ratio}",
            f"- Contractions: {profile.formality.contraction_usage}",
            f"- Formality patterns: {'; '.join(profile.formality.formality_patterns)}",
            "",
        ]
    )

    prompt_parts.extend(
        [
            "**Vocabulary Preferences:**",
            f"- Technical density: {profile.vocabulary.technical_density}",
            f"- Lexical density target: {profile.vocabulary.lexical_density}",
            f"- Word-choice patterns: {'; '.join(profile.vocabulary.word_choice_patterns)}",
            f"- Register consistency: {profile.vocabulary.register_consistency}",
            f"- Vocabulary range: {profile.vocabulary.vocabulary_range}",
            f"- Word length preference: {profile.vocabulary.word_length_preference}",
            "",
        ]
    )

    prompt_parts.extend(
        [
            "**Sentence Structure:**",
            f"- Target length: {profile.sentence_structure.average_length} words",
            f"- Variation pattern: {profile.sentence_structure.length_variation}",
            f"- Complexity level: {profile.sentence_structure.complexity_level}",
            f"- Clause tendencies: {'; '.join(profile.sentence_structure.clause_structure_preferences)}",
            f"- Syntactic patterns: {'; '.join(profile.sentence_structure.syntactic_patterns)}",
            "",
        ]
    )

    prompt_parts.extend(
        [
            "**Punctuation Style:**",
            f"- Comma usage: {profile.punctuation.comma_density}",
            f"- Semicolons: {profile.punctuation.semicolon_usage}",
            f"- Dashes: {profile.punctuation.dash_preferences}",
            f"- Parentheticals: {profile.punctuation.parenthetical_style}",
            f"- Punctuation patterns: {'; '.join(profile.punctuation.punctuation_patterns)}",
            "",
        ]
    )

    prompt_parts.extend(
        [
            "**Paragraph Organization:**",
            f"- Length: {profile.discourse.paragraph_length}",
            f"- Opening style: {profile.discourse.topic_sentence_style}",
            f"- Signposting: {profile.discourse.signposting_frequency}",
            f"- Argument structure: {profile.discourse.argument_structure}",
            f"- Transition types: {', '.join(profile.discourse.transition_types)}",
            f"- Cohesion strategies: {'; '.join(profile.discourse.cohesion_strategies)}",
            "",
        ]
    )

    prompt_parts.extend(
        [
            "**Citation Integration:**",
            f"- Integration: {profile.citations.integration_preference}",
            f"- Quotation frequency: {profile.citations.quotation_frequency}",
            f"- Source density: {profile.citations.source_density}",
            f"- Attribution style: {'; '.join(profile.citations.attribution_style)}",
            "",
        ]
    )

    prompt_parts.extend(
        [
            "**Rhetorical Moves:**",
            f"- Devices to include: {', '.join(profile.rhetoric.devices_used)}",
            f"- Device frequency: {profile.rhetoric.device_frequency}",
            f"- Persuasive strategy: {profile.rhetoric.persuasive_strategy}",
            f"- Emphasis techniques: {'; '.join(profile.rhetoric.emphasis_techniques)}",
            "",
        ]
    )

    prompt_parts.extend(
        [
            "**Tone and Voice:**",
            f"- Primary tone: {profile.tone_voice.primary_tone}",
            f"- Supplemental tones: {', '.join(profile.tone_voice.secondary_tones)}",
            f"- Authorial presence: {profile.tone_voice.authorial_presence}",
            f"- Stance indicators: {'; '.join(profile.tone_voice.stance_indicators)}",
            f"- Confidence level: {profile.tone_voice.confidence_level}",
            f"- Emotional register: {profile.tone_voice.emotional_register}",
            "",
        ]
    )

    prompt_parts.extend(
        [
            "## Most Distinctive Features:",
            "",
        ]
    )
    for i, feature in enumerate(profile.distinguishing_features, 1):
        prompt_parts.append(f"{i}. {feature}")

    prompt_parts.extend(
        [
            "",
            "---",
            "",
            "Write your response matching ALL of these stylistic patterns. The content should be new, but the style should be indistinguishable from the source author.",
        ]
    )

    return "\n".join(prompt_parts)


def _format_markdown_preview(profile: AcademicWritingProfile) -> str:
    """Convert the AcademicWritingProfile into readable markdown."""

    parts: List[str] = ["# 🎓 Academic Writing Profile\n"]

    parts.append("## Executive Summary\n")
    parts.append(f"{profile.profile_summary}\n")
    parts.append("\n### Distinguishing Features\n")
    for idx, feature in enumerate(profile.distinguishing_features, 1):
        parts.append(f"{idx}. {feature}\n")

    parts.append("\n## Hedging & Certainty\n")
    parts.append(f"- Modal verbs: {', '.join(profile.hedging.modal_verbs)}\n")
    parts.append(f"- Epistemic verbs: {', '.join(profile.hedging.epistemic_verbs)}\n")
    parts.append(f"- Probability markers: {', '.join(profile.hedging.probability_markers)}\n")
    parts.append(f"- Frequency: {profile.hedging.hedging_frequency}\n")
    parts.append("\n**Hedging patterns**\n")
    for pattern in profile.hedging.hedging_patterns:
        parts.append(f"- {pattern}\n")

    parts.append("\n## Formality & Register\n")
    parts.append(f"- Overall level: {profile.formality.overall_level}\n")
    parts.append(f"- Nominalization: {profile.formality.nominalization_frequency}\n")
    parts.append(f"- Pronoun usage: {profile.formality.personal_pronoun_usage}\n")
    parts.append(f"- Passive voice: {profile.formality.passive_voice_ratio}\n")
    parts.append(f"- Contractions: {profile.formality.contraction_usage}\n")
    parts.append("\n**Formality patterns**\n")
    for pattern in profile.formality.formality_patterns:
        parts.append(f"- {pattern}\n")

    parts.append("\n## Vocabulary Patterns\n")
    parts.append(f"- Technical density: {profile.vocabulary.technical_density}\n")
    parts.append(f"- Lexical density: {profile.vocabulary.lexical_density}\n")
    parts.append(f"- Register consistency: {profile.vocabulary.register_consistency}\n")
    parts.append(f"- Vocabulary range: {profile.vocabulary.vocabulary_range}\n")
    parts.append(f"- Word length preference: {profile.vocabulary.word_length_preference}\n")
    parts.append("\n**Word-choice patterns**\n")
    for pattern in profile.vocabulary.word_choice_patterns:
        parts.append(f"- {pattern}\n")

    parts.append("\n## Sentence Structure\n")
    parts.append(f"- Average length: {profile.sentence_structure.average_length}\n")
    parts.append(f"- Variation: {profile.sentence_structure.length_variation}\n")
    parts.append(f"- Complexity: {profile.sentence_structure.complexity_level}\n")
    parts.append("\n**Clause structure preferences**\n")
    for clause in profile.sentence_structure.clause_structure_preferences:
        parts.append(f"- {clause}\n")
    parts.append("\n**Syntactic patterns**\n")
    for pattern in profile.sentence_structure.syntactic_patterns:
        parts.append(f"- {pattern}\n")

    parts.append("\n## Punctuation & Mechanics\n")
    parts.append(f"- Comma usage: {profile.punctuation.comma_density}\n")
    parts.append(f"- Semicolons: {profile.punctuation.semicolon_usage}\n")
    parts.append(f"- Dashes: {profile.punctuation.dash_preferences}\n")
    parts.append(f"- Parentheticals: {profile.punctuation.parenthetical_style}\n")
    parts.append("\n**Punctuation patterns**\n")
    for pattern in profile.punctuation.punctuation_patterns:
        parts.append(f"- {pattern}\n")

    parts.append("\n## Discourse Organization\n")
    parts.append(f"- Paragraph length: {profile.discourse.paragraph_length}\n")
    parts.append(f"- Topic sentence style: {profile.discourse.topic_sentence_style}\n")
    parts.append(f"- Signposting: {profile.discourse.signposting_frequency}\n")
    parts.append(f"- Argument structure: {profile.discourse.argument_structure}\n")
    parts.append("\n**Transition types**\n")
    for transition in profile.discourse.transition_types:
        parts.append(f"- {transition}\n")
    parts.append("\n**Cohesion strategies**\n")
    for strategy in profile.discourse.cohesion_strategies:
        parts.append(f"- {strategy}\n")

    parts.append("\n## Citation Integration\n")
    parts.append(f"- Integration preference: {profile.citations.integration_preference}\n")
    parts.append(f"- Quotation frequency: {profile.citations.quotation_frequency}\n")
    parts.append(f"- Source density: {profile.citations.source_density}\n")
    parts.append("\n**Attribution style**\n")
    for style in profile.citations.attribution_style:
        parts.append(f"- {style}\n")

    parts.append("\n## Rhetorical Devices\n")
    parts.append(f"- Strategy: {profile.rhetoric.persuasive_strategy}\n")
    parts.append(f"- Devices: {', '.join(profile.rhetoric.devices_used)}\n")
    parts.append(f"- Device frequency: {profile.rhetoric.device_frequency}\n")
    parts.append("\n**Emphasis techniques**\n")
    for technique in profile.rhetoric.emphasis_techniques:
        parts.append(f"- {technique}\n")

    parts.append("\n## Tone & Voice\n")
    parts.append(f"- Primary tone: {profile.tone_voice.primary_tone}\n")
    parts.append(f"- Secondary tones: {', '.join(profile.tone_voice.secondary_tones)}\n")
    parts.append(f"- Authorial presence: {profile.tone_voice.authorial_presence}\n")
    parts.append(f"- Confidence level: {profile.tone_voice.confidence_level}\n")
    parts.append(f"- Emotional register: {profile.tone_voice.emotional_register}\n")
    parts.append("\n**Stance indicators**\n")
    for indicator in profile.tone_voice.stance_indicators:
        parts.append(f"- {indicator}\n")

    parts.append("\n## Discipline Conventions\n")
    parts.append(f"- Primary discipline: {profile.discipline.primary_discipline}\n")
    parts.append(f"- Structure preference: {profile.discipline.structure_preference}\n")
    parts.append(f"- Methodology visibility: {profile.discipline.methodology_visibility}\n")
    parts.append("\n**Evidence types**\n")
    for evidence in profile.discipline.evidence_types:
        parts.append(f"- {evidence}\n")
    parts.append("\n**Field conventions**\n")
    for convention in profile.discipline.field_specific_conventions:
        parts.append(f"- {convention}\n")

    return "".join(parts)


# ============================================================================
# Provider-Specific Calls using LangChain
# ============================================================================

def _call_with_langchain(
    model_id: str,
    provider: str,
    system_prompt: str,
    user_prompt: str,
    sample_text: str,
    reasoning: Dict[str, Any],
) -> AcademicWritingProfile:
    """Use LangChain's structured output for all providers."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Import LangChain models based on provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        
        # Reasoning models need max_completion_tokens (excludes reasoning tokens)
        is_reasoning_model = model_id.startswith(("gpt-5", "o1", "o3"))
        
        kwargs: Dict[str, Any] = {
            "model": model_id,
            "api_key": OPENAI_API_KEY,
        }
        
        # Use max_completion_tokens for reasoning models, max_tokens for others
        if is_reasoning_model:
            kwargs["max_completion_tokens"] = 16384  # Very large structured output needs high limit
        else:
            kwargs["max_tokens"] = 16384  # Very large structured output needs high limit
            kwargs["temperature"] = 0.15
        
        # Add reasoning effort for supported models (pass directly, not in model_kwargs)
        effort = reasoning.get("reasoning_effort")
        if effort and effort != "none" and is_reasoning_model:
            kwargs["reasoning_effort"] = effort
        
        llm = ChatOpenAI(**kwargs)
        
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        kwargs: Dict[str, Any] = {
            "model": model_id,
            "google_api_key": GEMINI_API_KEY,
            "max_output_tokens": 16384,  # Very large structured output needs high limit
            "temperature": 0.2,
        }
        
        # Add thinking config if supported (Gemini uses model_kwargs for this)
        budget = reasoning.get("thinking_budget")
        if budget:
            kwargs["model_kwargs"] = {"thinking_config": {"budget_tokens": int(budget)}}
        
        llm = ChatGoogleGenerativeAI(**kwargs)
        
    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic
        
        # Claude Haiku has a lower token limit (8192), Sonnet supports 16384
        max_tokens_claude = 8192 if "haiku" in model_id.lower() else 16384
        
        # Check if thinking is enabled
        budget = reasoning.get("thinking_budget")
        has_thinking = budget and int(budget) > 0
        
        # IMPORTANT: Claude's structured output uses forced tool calling, which is
        # incompatible with extended thinking mode. We must disable thinking for
        # structured output to work. See: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
        if has_thinking:
            print(f"[WARNING] Claude thinking mode is incompatible with structured output. Disabling thinking for {model_id}.")
            has_thinking = False
        
        kwargs: Dict[str, Any] = {
            "model": model_id,
            "anthropic_api_key": CLAUDE_API_KEY,
            "max_tokens": max_tokens_claude,
            "temperature": 0.15,  # Use standard temperature since thinking is disabled
            }
        
        llm = ChatAnthropic(**kwargs)
        
    elif provider == "groq":
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(
            model=model_id,
            groq_api_key=GROQ_API_KEY,
            max_tokens=16384,  # Very large structured output needs high limit
            temperature=0.15,
        )
        
    else:
        raise ValueError(f"Provider '{provider}' is not supported")
    
    # Build messages with instructions and sample sent separately to mimic file attachment
    instruction_message = HumanMessage(content=[{"type": "text", "text": user_prompt}])
    sample_message = HumanMessage(
        content=[
            {"type": "text", "text": "[attachment:writing_sample.txt]"},
            {"type": "text", "text": sample_text},
        ],
        additional_kwargs={
            "attachment": {
                "name": "writing_sample.txt",
                "mime_type": "text/plain",
            }
        },
    )

    messages = [
        SystemMessage(content=system_prompt),
        instruction_message,
        sample_message,
    ]

    # Use structured output with appropriate method per provider
    if provider == "openai":
        # OpenAI supports native function calling for structured output
        structured_llm = llm.with_structured_output(AcademicWritingProfile, method="function_calling")
    elif provider == "claude":
        # Claude uses tool calling for structured output
        structured_llm = llm.with_structured_output(AcademicWritingProfile, method="function_calling")
    elif provider == "gemini":
        # Gemini supports JSON schema
        structured_llm = llm.with_structured_output(AcademicWritingProfile, method="json_mode")
    elif provider == "groq":
        # Groq has limited structured output support - use JSON mode
        structured_llm = llm.with_structured_output(AcademicWritingProfile, method="json_mode")
    else:
        # Default fallback
        structured_llm = llm.with_structured_output(AcademicWritingProfile)
    
    # Invoke and return
    try:
        print(f"[DEBUG] Calling {provider} with structured output...")
        result = structured_llm.invoke(messages)
        print(f"[DEBUG] Got result from {provider}")
        return result
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Structured output failed for {provider}: {error_msg[:200]}")
        import traceback
        print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        
        # Try plain invocation and manual parsing
        try:
            print(f"[DEBUG] Attempting fallback for {provider}...")
            plain_result = llm.invoke(messages)
            
            # Extract content, handling both string and list cases
            if hasattr(plain_result, 'content'):
                content = plain_result.content
                # Handle case where content might be a list (e.g., from some providers)
                if isinstance(content, list):
                    # Join list elements if they're strings, otherwise convert to string
                    json_str = " ".join(str(item) for item in content)
                elif isinstance(content, str):
                    json_str = content
                else:
                    json_str = str(content)
            else:
                json_str = str(plain_result)
            
            # Ensure json_str is a string before calling string methods
            if not isinstance(json_str, str):
                json_str = str(json_str)
            
            # Try to extract JSON from the response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            else:
                json_str = json_str.strip()
            
            print(f"[DEBUG] Parsing JSON manually...")
            import json
            parsed = json.loads(json_str)
            
            print(f"[DEBUG] Creating AcademicWritingProfile from parsed data...")
            return AcademicWritingProfile(**parsed)
            
        except Exception as e2:
            print(f"[ERROR] Fallback also failed: {str(e2)[:200]}")
            raise ValueError(f"Failed to get structured output from {provider}. Original error: {error_msg[:200]}, Fallback error: {str(e2)[:200]}")


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_writing_profile(
    model_id: str,
    sample_text: str,
    *,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the writing profile analysis with the selected model."""
    if not sample_text or not sample_text.strip():
        raise ValueError("Writing sample is empty.")

    meta = get_model_info(model_id)
    provider = meta["provider"]
    actual_model_id = meta["model"]
    reasoning = reasoning or {}

    system_prompt = ACADEMIC_WRITING_PROFILE_SYSTEM_PROMPT

    sample_count = max(sample_text.count("\n\n---\n\n") + 1, 1)
    total_words = len(sample_text.split())
    reasoning_directive = _format_reasoning_directive(reasoning)
    reasoning_enabled = any(
        value not in (None, "", False, 0) for value in reasoning.values()
    )
    user_prompt = build_academic_profile_user_prompt(
        sample_count=sample_count,
        total_words=total_words,
        reasoning_enabled=reasoning_enabled,
        reasoning_directive=reasoning_directive,
    )
    thinking_budget = reasoning.get("thinking_budget") or 0
    token_usage = estimate_token_usage(sample_text, thinking_tokens=int(thinking_budget))
    pricing = estimate_cost(model_id, **token_usage)

    # Use LangChain for all providers (sample_text is attached in _call_with_langchain)
    profile_obj = _call_with_langchain(
        actual_model_id,
        provider,
        system_prompt,
        user_prompt,
        sample_text,
        reasoning,
    )
    
    # Convert Pydantic model to dict
    profile_dict = profile_obj.model_dump()
    
    # For raw output, convert back to JSON string for display
    import json
    raw_output = json.dumps(profile_dict, indent=2)
    
    # Generate beautiful markdown preview
    markdown_preview = _format_markdown_preview(profile_obj)
    
    return {
        "profile": profile_dict,
        "raw_output": raw_output,
        "markdown_preview": markdown_preview,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "pricing": pricing,
        "token_usage": token_usage,
    }
