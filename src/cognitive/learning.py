from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

@dataclass
class LearningExample:
    """Represents a learning example from past experiences."""
    id: str
    context: Dict[str, Any]
    decision: Dict[str, Any]
    outcome: Dict[str, Any]
    feedback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class Rule:
    """Represents a learned rule for decision making."""
    id: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Any]
    confidence: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    description: str = ""

class LearningSystem:
    """System for learning from past experiences and improving decision making."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.examples: List[LearningExample] = []
        self.rules: Dict[str, Rule] = {}
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)
        
        # Load existing data if storage path is provided
        if storage_path:
            self.load_examples()
    
    def add_example(
        self,
        example_id: str,
        context: Dict[str, Any],
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        feedback: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> LearningExample:
        """Add a new learning example."""
        example = LearningExample(
            id=example_id,
            context=context,
            decision=decision,
            outcome=outcome,
            feedback=feedback,
            tags=tags or []
        )
        self.examples.append(example)
        
        # Save to storage if path is configured
        if self.storage_path:
            self.save_examples()
        
        # Try to learn new rules from this example
        self._learn_from_example(example)
        
        return example
    
    def add_rule(
        self,
        rule_id: str,
        condition: Callable[[Dict[str, Any]], bool],
        action: Callable[[Dict[str, Any]], Any],
        description: str = ""
    ) -> Rule:
        """Add a new rule to the system."""
        rule = Rule(
            id=rule_id,
            condition=condition,
            action=action,
            description=description
        )
        self.rules[rule_id] = rule
        return rule
    
    def get_applicable_rules(
        self,
        context: Dict[str, Any]
    ) -> List[Rule]:
        """Get all rules that apply to the given context."""
        applicable_rules = []
        for rule in self.rules.values():
            try:
                if rule.condition(context):
                    applicable_rules.append(rule)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.id}: {str(e)}")
        
        # Sort by confidence and usage
        return sorted(
            applicable_rules,
            key=lambda r: (r.confidence, r.usage_count),
            reverse=True
        )
    
    def apply_rules(
        self,
        context: Dict[str, Any]
    ) -> List[tuple[Rule, Any]]:
        """Apply all applicable rules to the context."""
        results = []
        for rule in self.get_applicable_rules(context):
            try:
                result = rule.action(context)
                rule.usage_count += 1
                rule.last_used = datetime.now()
                results.append((rule, result))
            except Exception as e:
                self.logger.error(f"Error applying rule {rule.id}: {str(e)}")
        
        return results
    
    def update_rule_confidence(
        self,
        rule_id: str,
        success: bool
    ) -> None:
        """Update a rule's confidence based on its success."""
        rule = self.rules.get(rule_id)
        if not rule:
            return
        
        rule.usage_count += 1
        if success:
            rule.success_count += 1
        
        # Update confidence using a simple success rate
        rule.confidence = rule.success_count / rule.usage_count
    
    def get_similar_examples(
        self,
        context: Dict[str, Any],
        limit: int = 5
    ) -> List[LearningExample]:
        """Get examples similar to the given context."""
        scored_examples = []
        
        for example in self.examples:
            score = self._calculate_similarity(context, example.context)
            if score > 0:
                scored_examples.append((score, example))
        
        # Sort by similarity score
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_examples[:limit]]
    
    def _calculate_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two contexts."""
        # This is a simple implementation that could be enhanced
        # with more sophisticated similarity metrics
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarity = 0.0
        for key in common_keys:
            if context1[key] == context2[key]:
                similarity += 1.0
        
        return similarity / len(common_keys)
    
    def _learn_from_example(self, example: LearningExample) -> None:
        """Try to learn new rules from an example."""
        # This is where you would implement more sophisticated
        # learning algorithms. For now, we'll just log the example.
        self.logger.info(f"Learning from example {example.id}")
    
    def save_examples(self) -> None:
        """Save learning examples to storage."""
        if not self.storage_path:
            return
        
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        serialized_examples = [
            {
                "id": ex.id,
                "context": ex.context,
                "decision": ex.decision,
                "outcome": ex.outcome,
                "feedback": ex.feedback,
                "timestamp": ex.timestamp.isoformat(),
                "tags": ex.tags
            }
            for ex in self.examples
        ]
        
        with open(path, "w") as f:
            json.dump(serialized_examples, f, indent=2)
    
    def load_examples(self) -> None:
        """Load learning examples from storage."""
        if not self.storage_path or not Path(self.storage_path).exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self.examples = [
                LearningExample(
                    id=ex["id"],
                    context=ex["context"],
                    decision=ex["decision"],
                    outcome=ex["outcome"],
                    feedback=ex["feedback"],
                    timestamp=datetime.fromisoformat(ex["timestamp"]),
                    tags=ex["tags"]
                )
                for ex in data
            ]
        except Exception as e:
            self.logger.error(f"Error loading examples: {str(e)}")
    
    def get_examples_by_tag(self, tag: str) -> List[LearningExample]:
        """Get all examples with a specific tag."""
        return [ex for ex in self.examples if tag in ex.tags]
    
    def get_example_by_id(self, example_id: str) -> Optional[LearningExample]:
        """Get a specific example by ID."""
        for example in self.examples:
            if example.id == example_id:
                return example
        return None
    
    def add_feedback(
        self,
        example_id: str,
        feedback: str
    ) -> Optional[LearningExample]:
        """Add feedback to an existing example."""
        example = self.get_example_by_id(example_id)
        if example:
            example.feedback = feedback
            if self.storage_path:
                self.save_examples()
        return example
    
    def clear_examples(self) -> None:
        """Clear all learning examples."""
        self.examples.clear()
        if self.storage_path:
            self.save_examples()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning system."""
        return {
            "total_examples": len(self.examples),
            "total_rules": len(self.rules),
            "rules_by_confidence": {
                rule.id: rule.confidence
                for rule in sorted(
                    self.rules.values(),
                    key=lambda r: r.confidence,
                    reverse=True
                )
            },
            "most_common_tags": self._get_tag_frequencies(),
            "examples_by_month": self._get_examples_by_month()
        }
    
    def _get_tag_frequencies(self) -> Dict[str, int]:
        """Get frequency count of tags."""
        frequencies: Dict[str, int] = {}
        for example in self.examples:
            for tag in example.tags:
                frequencies[tag] = frequencies.get(tag, 0) + 1
        return dict(sorted(
            frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def _get_examples_by_month(self) -> Dict[str, int]:
        """Get count of examples by month."""
        by_month: Dict[str, int] = {}
        for example in self.examples:
            month_key = example.timestamp.strftime("%Y-%m")
            by_month[month_key] = by_month.get(month_key, 0) + 1
        return dict(sorted(by_month.items())) 