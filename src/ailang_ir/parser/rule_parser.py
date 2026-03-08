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
        if not comparison:
            # "X verb better/more than Y" (no copula)
            comparison = re.search(
                r'(.+?)\s+\w+\s+(?:better|more|worse|less|faster|slower)\s+than\s+(.+)',
                text, re.IGNORECASE,
            )
        if comparison:
            obj_text = self._strip_prefix(comparison.group(1).strip())
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
        1. Strip speaker/mode prefixes (with adverb tolerance)
        2. Strip act-related verbs (with comma-clause handling)
        3. Strip auxiliary/copula verbs and filler clauses
        4. Strip residual pronouns and non-content verbs
        5. Extract the core noun phrase — prefer compact, meaningful keys
        """
        cleaned = text.strip()

        # Step 1: Remove speaker/mode prefixes (tolerant of adverbs like "also")
        prefix_patterns = [
            r'^(?:I think|I believe|I feel|I guess|In my opinion|Maybe|Perhaps|Probably)\s+(?:that\s+)?',
            r'^(?:It seems like|It appears that|It looks like)\s+',
            r'^(?:We|You|The system)\s+(?:also\s+)?(?:should|need to|need|could|will|can|must)\s+',
            r'^(?:Let\'s|Let us)\s+',
            # Question-form prefixes
            r'^(?:Have you|Did you|Do you|Can you|Could you)\s+(?:ever\s+)?(?:considered|thought about|tried|looked at|used)\s+',
            r'^(?:What about|How about)\s+',
        ]
        for pat in prefix_patterns:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)

        # Step 2: Remove act-related verb phrases (including "I verb" patterns)
        act_verbs = {
            SemanticAct.BELIEVE: r'(?:I\s+)?(?:think|believe|feel|suppose)\s+(?:that\s+)?',
            SemanticAct.PREFER: r'(?:I\s+)?(?:prefer|favor|like better|would rather)\s+',
            SemanticAct.SUGGEST: r'(?:I\s+)?(?:suggest|propose|recommend)\s+(?:that\s+)?(?:we\s+)?(?:use\s+|try\s+)?',
            SemanticAct.NEED: r'(?:I\s+|we\s+)?(?:also\s+)?(?:need|require|must have|should have)\s+',
            SemanticAct.DECIDE: r'(?:I\s+)?(?:decide|chose|settled on)\s+(?:on\s+|to\s+)?',
            SemanticAct.REJECT: r'(?:I\s+)?(?:reject|refuse|decline)\s+',
            SemanticAct.AGREE: r'(?:I\s+)?(?:agree|concur)\s+(?:with\s+|that\s+)?',
            SemanticAct.DISAGREE: r'(?:I\s+)?(?:disagree)\s+(?:with\s+|that\s+)?',
            SemanticAct.CREATE: r'(?:I\s+|we\s+)?(?:create[ds]?|build|built|make|implement(?:ed)?|develop(?:ed)?|writ(?:e|ten)|generate[ds]?|add(?:ed)?)\s+',
            SemanticAct.MODIFY: r'(?:I\s+|we\s+)?(?:modif(?:y|ied)|change[ds]?|update[ds]?|edit(?:ed)?|revis(?:e[ds]?|ing)|adjust(?:ed)?|fix(?:ed)?|improv(?:e[ds]?|ing))\s+',
            SemanticAct.DELETE: r'(?:I\s+|we\s+)?(?:delete[ds]?|remov(?:e[ds]?|ing)|drop(?:ped)?|discard(?:ed)?)\s+',
            SemanticAct.OBSERVE: r'(?:I\s+)?(?:notice[ds]?|see|observe[ds]?|found|show[ns]?|reveal[ns]?|indicat(?:e[ds]?|ing)|confirm[ns]?)\s+(?:that\s+)?',
            SemanticAct.EXPLAIN: r'(?:explain|because|the reason is)\s+(?:that\s+)?',
            SemanticAct.QUERY: r'(?:what is|what are|how does|how do|how is|how are|why|where|when)\s+',
            SemanticAct.WARN: r'(?:warn|be careful|watch out)\s+(?:about\s+)?',
            SemanticAct.COMPARE: r'(?:I\s+)?(?:compare|comparing)\s+',
            SemanticAct.PLAN: r'(?:I\s+|we\s+)?(?:plan to|going to|intend to|aim to)\s+',
        }
        if act in act_verbs:
            cleaned = re.sub(f'^{act_verbs[act]}', '', cleaned, flags=re.IGNORECASE)

        # Step 2b: Handle comma-separated stance clauses
        # "I disagree, a monolith is better" → "a monolith is better"
        # Only strip if the stance word is at the START (not mid-sentence)
        # and what's after the comma has enough content
        comma_stance = re.match(
            r'^(?:I\s+)?(agree|disagree|concur|think so|don\'t think so)[,;]\s*(.+)',
            cleaned, re.IGNORECASE,
        )
        if comma_stance and len(comma_stance.group(2).split()) >= 3:
            cleaned = comma_stance.group(2)

        # Step 3: Reduce clausal complexity — extract core topic
        # Strip punctuation early so trailing-word patterns work
        cleaned = cleaned.rstrip('.!,;:').strip()
        # Remove "be able to X" → keep X
        cleaned = re.sub(r'\bbe\s+able\s+to\s+', '', cleaned, flags=re.IGNORECASE)
        # Remove auxiliary chains
        cleaned = re.sub(r'^(?:should|will|can|could|would|must|may|might)\s+(?:be\s+)?', '', cleaned, flags=re.IGNORECASE)
        # Remove passive "be Xed" → keep as active concept
        cleaned = re.sub(r'^be\s+(\w+ed)\b', r'\1', cleaned, flags=re.IGNORECASE)
        # Remove observation prefix if still present
        cleaned = re.sub(r'^(?:I notice|I see|I observe|I found)\s+(?:that\s+)?', '', cleaned, flags=re.IGNORECASE)
        # Remove trailing filler
        cleaned = re.sub(r'\s+(?:instead|as well|too|also|anyway)$', '', cleaned, flags=re.IGNORECASE)

        # Step 3b: Strip residual leading verbs and pronouns
        # "use PostgreSQL" → "PostgreSQL", "deploy on AWS" → "AWS"
        cleaned = re.sub(
            r'^(?:use|using|deploy|deploying|design|designing|consider|considered|'
            r'handle|handling|require|requires|suggest|suggesting|'
            r'added|implemented|built|created|wrote|generated|'
            r'updated|changed|fixed|improved|modified|revised|'
            r'removed|deleted|dropped|'
            r'shows?|reveals?|indicates?|confirms?|'
            r'completed|passed|finished|done)\s+',
            '', cleaned, flags=re.IGNORECASE,
        )

        # Step 4: Extract noun-phrase core
        cleaned = cleaned.strip()
        cleaned = self._trim_to_noun_phrase(cleaned)

        if not cleaned or len(cleaned) < 2:
            return None

        return cleaned

    @staticmethod
    def _trim_to_noun_phrase(text: str) -> str:
        """
        Trim a clause down to its core noun phrase(s).

        Heuristics:
        - If text contains "SUBJECT is/are/will PREDICATE", keep SUBJECT + key PREDICATE words
        - Remove subordinate clauses introduced by "that", "which", "because"
        - Limit to ~4 content words to keep keys compact
        """
        # Remove subordinate clause tails (but NOT "into" — it often precedes the key object)
        text = re.split(r'\s+(?:that|which|because|since|although|where|when)\s+', text, flags=re.IGNORECASE)[0]

        # Split at copula/auxiliary to separate subject from predicate
        verb_split = re.search(
            r'^(.+?)\s+(?:will|should|can|could|would|must|is|are|was|were|has|have|had)\s+(?:be\s+)?(.+)',
            text, re.IGNORECASE,
        )
        if verb_split:
            subject = verb_split.group(1).strip()
            predicate = verb_split.group(2).strip()
            pronouns = {'it', 'he', 'she', 'they', 'we', 'you', 'this', 'that', 'there'}
            if subject.lower() in pronouns:
                # Use predicate as the object
                text = predicate
            else:
                # Check if predicate is just adjectives/adverbs (no nouns)
                # e.g. "Redis is fast and well supported" → subject "Redis" is the concept
                pred_words = [w for w in predicate.split()
                              if w.lower() not in {'a', 'an', 'the', 'and', 'or', 'but',
                                                   'very', 'really', 'well', 'quite'}
                              and len(w) > 2]
                # If predicate has a preposition + noun, keep that structure
                prep_match = re.search(
                    r'(\w+)\s+(into|from|for|with|about|over|under|between)\s+(.+)',
                    predicate, re.IGNORECASE,
                )
                if prep_match:
                    text = f"{subject} {prep_match.group(1)} {prep_match.group(2)} {prep_match.group(3)}"
                elif pred_words and any(w[0].isupper() for w in pred_words):
                    # Predicate has proper nouns — include them
                    text = f"{subject} {' '.join(pred_words[:2])}"
                elif pred_words:
                    text = f"{subject} {' '.join(pred_words[:2])}"
                else:
                    text = subject

        # Compact: limit to ~6 content words
        words = text.split()
        if len(words) > 6:
            text = ' '.join(words[:6])

        return text.strip()

    @staticmethod
    def _strip_prefix(text: str) -> str:
        """Remove speaker/mode prefixes from extracted text fragments."""
        patterns = [
            r'^(?:I think|I believe|I feel|I guess|In my opinion|Maybe|Perhaps|Probably)\s+(?:that\s+)?',
            r'^(?:It seems like|It appears that|It looks like)\s+',
            r'^(?:We should|We need to|We need|We could|You should|The system should)\s+',
            r'^(?:Let\'s|Let us)\s+',
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.IGNORECASE)
        return text.strip()

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
