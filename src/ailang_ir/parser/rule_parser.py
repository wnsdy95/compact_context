"""
Rule-based semantic parser.

Converts a natural language sentence into a SemanticFrame by applying
normalization vocabulary rules and simple structural heuristics.

This is the MVP parser — deterministic, no ML dependency.
It will produce partial frames when confidence is low, leaving fields
at their default/unknown values rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ailang_ir.models.domain import (
    Certainty,
    Entity,
    Predicate,
    Priority,
    SemanticAct,
    SemanticFrame,
    SemanticMode,
    Sentiment,
    SpeakerRole,
    TimeReference,
)
from ailang_ir.normalize.vocabulary import NormalizationVocabulary


@dataclass
class RuleBasedParser:
    """
    Deterministic rule-based parser that converts raw text into a SemanticFrame.

    Uses NormalizationVocabulary for surface → canonical mapping,
    then applies structural heuristics to extract the semantic object.
    """
    vocab: NormalizationVocabulary = field(default_factory=NormalizationVocabulary)

    def parse(
        self,
        text: str,
        speaker: SpeakerRole = SpeakerRole.USER,
    ) -> SemanticFrame:
        """
        Parse a single sentence into a SemanticFrame.

        The parser extracts:
        - mode (epistemic/speech-act type)
        - act (core predicate)
        - object (what the statement is about)
        - target (optional secondary entity, e.g. for comparisons)
        - time reference
        - certainty estimate
        - sentiment

        Fields that cannot be determined are left at default/unknown values.
        """
        text = text.strip()
        if not text:
            return SemanticFrame(speaker=speaker, source_text=text)

        mode = self.vocab.match_mode(text)
        act = self.vocab.match_act(text)
        time = self.vocab.match_time(text)
        sentiment = self.vocab.match_sentiment(text)
        certainty_val = self.vocab.estimate_certainty(text)

        obj_entity, target_entity = self._extract_entities(text, act)

        return SemanticFrame(
            speaker=speaker,
            mode=mode,
            predicate=Predicate(act),
            object=obj_entity,
            target=target_entity,
            time=time,
            certainty=Certainty(certainty_val),
            sentiment=sentiment,
            priority=Priority.UNSET,
            source_text=text,
        )

    def _extract_entities(
        self, text: str, act: SemanticAct
    ) -> tuple[Entity | None, Entity | None]:
        """
        Extract the primary object and optional target entity from text.

        Strategy:
        1. For comparisons (COMPARE, PREFER), look for "X over Y" / "X than Y" patterns
        2. For other acts, extract the main noun phrase after the verb
        3. Normalize the extracted text into a canonical key
        """
        obj_text = None
        target_text = None

        # Comparison patterns — try regardless of detected act,
        # since act detection may miss comparisons.
        comparison = re.search(
            r'(?:prefer|like|favor|choose)\s+(.+?)\s+(?:over|than|versus|vs\.?)\s+(.+)',
            text, re.IGNORECASE,
        )
        if not comparison:
            # "X seems more appropriate than Y"
            comparison = re.search(
                r'(.+?)\s+(?:seems?|is|are)\s+(?:more|better|worse|less)\s+\w+\s+than\s+(.+)',
                text, re.IGNORECASE,
            )
        if comparison:
            obj_text = comparison.group(1).strip()
            target_text = comparison.group(2).strip()

        # General object extraction if not already found
        if obj_text is None:
            obj_text = self._extract_main_object(text, act)

        # Build entities
        obj_entity = None
        if obj_text:
            key = self.vocab.normalize_object_key(obj_text)
            if key:
                obj_entity = Entity(canonical=key, surface_forms=(obj_text,))

        target_entity = None
        if target_text:
            key = self.vocab.normalize_object_key(target_text)
            if key:
                target_entity = Entity(canonical=key, surface_forms=(target_text,))

        return obj_entity, target_entity

    def _extract_main_object(self, text: str, act: SemanticAct) -> str | None:
        """
        Extract the main object/topic from a sentence.

        Strategy:
        1. Strip speaker/mode prefixes
        2. Strip act-related verbs
        3. Strip auxiliary/copula verbs and filler clauses
        4. Extract the core noun phrase(s) that remain
        """
        cleaned = text.strip()

        # Step 1: Remove speaker/mode prefixes
        prefix_patterns = [
            r'^(?:I think|I believe|I feel|I guess|In my opinion|Maybe|Perhaps|Probably)\s+(?:that\s+)?',
            r'^(?:It seems like|It appears that|It looks like)\s+',
            r'^(?:We should|We need to|We could|You should|The system should)\s+',
        ]
        for pat in prefix_patterns:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)

        # Step 2: Remove act-related verb phrases
        act_verbs = {
            SemanticAct.BELIEVE: r'(?:think|believe|feel|suppose)\s+(?:that\s+)?',
            SemanticAct.SUGGEST: r'(?:suggest|propose|recommend)\s+(?:that\s+)?',
            SemanticAct.NEED: r'(?:need|require|must have|should have)\s+',
            SemanticAct.CREATE: r'(?:create|build|make|implement|develop)\s+',
            SemanticAct.MODIFY: r'(?:modify|change|update|edit|revise)\s+',
            SemanticAct.DELETE: r'(?:delete|remove|drop|discard)\s+',
            SemanticAct.OBSERVE: r'(?:notice|see|observe|found)\s+(?:that\s+)?',
            SemanticAct.EXPLAIN: r'(?:explain|because|the reason is)\s+(?:that\s+)?',
            SemanticAct.QUERY: r'(?:what is|what are|how does|how do|why|where|when)\s+',
        }
        if act in act_verbs:
            cleaned = re.sub(f'^{act_verbs[act]}', '', cleaned, flags=re.IGNORECASE)

        # Step 3: Reduce clausal complexity — extract core topic
        # Remove "be able to X" → keep X
        cleaned = re.sub(r'\bbe\s+able\s+to\s+', '', cleaned, flags=re.IGNORECASE)
        # Remove auxiliary chains: "should be", "will be", "can be"
        cleaned = re.sub(r'^(?:should|will|can|could|would|must|may|might)\s+(?:be\s+)?', '', cleaned, flags=re.IGNORECASE)
        # Remove "be segmented/stored/..." → keep the core concept
        cleaned = re.sub(r'^be\s+(\w+ed)\s+', r'\1 ', cleaned, flags=re.IGNORECASE)

        # Step 4: Extract the main noun phrase
        # Try to find "NOUN_PHRASE VERB_TAIL" and keep just the noun phrase
        # Heuristic: if the remaining text has a clear subject-predicate structure,
        # split at the main verb
        verb_split = re.search(
            r'^(.+?)\s+(?:will|should|can|could|would|must|is|are|was|were|has|have|had)\s+(?:be\s+)?(\w+)',
            cleaned, re.IGNORECASE,
        )
        if verb_split:
            subject = verb_split.group(1).strip()
            complement = verb_split.group(2).strip()
            # Use subject if it looks like a real noun phrase (>= 2 chars, not just a pronoun)
            if len(subject) >= 3 and subject.lower() not in ('it', 'he', 'she', 'they', 'we', 'you', 'this', 'that'):
                # Combine subject with key complement for richer meaning
                tail = cleaned[verb_split.end():].strip().rstrip('.!,;:')
                if complement and tail:
                    cleaned = f"{subject} {complement} {tail}"
                elif complement:
                    cleaned = f"{subject} {complement}"
                else:
                    cleaned = subject

        # Final cleanup
        cleaned = cleaned.rstrip('.!,;:').strip()

        if not cleaned or len(cleaned) < 2:
            return None

        return cleaned

    def parse_multi(
        self,
        text: str,
        speaker: SpeakerRole = SpeakerRole.USER,
    ) -> list[SemanticFrame]:
        """
        Parse text that may contain multiple sentences.

        Splits on sentence boundaries and parses each independently.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [self.parse(s, speaker) for s in sentences if s.strip()]
