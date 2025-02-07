import pytest
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any

from src.cognitive.learning import (
    LearningExample,
    Rule,
    LearningSystem
)

def test_learning_example_creation():
    """Test learning example creation."""
    example = LearningExample(
        id="test_example",
        context={"type": "test"},
        decision={"option": "A"},
        outcome={"success": True}
    )
    assert example.id == "test_example"
    assert example.context == {"type": "test"}
    assert example.decision == {"option": "A"}
    assert example.outcome == {"success": True}
    assert isinstance(example.timestamp, datetime)

def test_rule_creation():
    """Test rule creation."""
    def condition(context: Dict[str, Any]) -> bool:
        return context.get("type") == "test"
    
    def action(context: Dict[str, Any]) -> str:
        return "test_action"
    
    rule = Rule(
        id="test_rule",
        condition=condition,
        action=action,
        description="Test rule"
    )
    
    assert rule.id == "test_rule"
    assert rule.description == "Test rule"
    assert rule.confidence == 0.0
    assert rule.usage_count == 0
    assert rule.success_count == 0
    assert isinstance(rule.created_at, datetime)
    assert rule.last_used is None

class TestLearningSystem:
    """Test suite for LearningSystem."""
    
    @pytest.fixture
    def learning_system(self, temp_storage_path):
        """Create a learning system instance."""
        return LearningSystem(str(temp_storage_path / "learning.json"))
    
    def test_add_example(self, learning_system, sample_learning_example):
        """Test adding a learning example."""
        example = learning_system.add_example(
            example_id=sample_learning_example["id"],
            context=sample_learning_example["context"],
            decision=sample_learning_example["decision"],
            outcome=sample_learning_example["outcome"],
            tags=sample_learning_example["tags"]
        )
        
        assert example.id == sample_learning_example["id"]
        assert example in learning_system.examples
    
    def test_add_rule(self, learning_system):
        """Test adding a rule."""
        def condition(context: Dict[str, Any]) -> bool:
            return True
        
        def action(context: Dict[str, Any]) -> str:
            return "action"
        
        rule = learning_system.add_rule(
            rule_id="test_rule",
            condition=condition,
            action=action,
            description="Test rule"
        )
        
        assert rule.id == "test_rule"
        assert rule in learning_system.rules.values()
    
    def test_get_applicable_rules(self, learning_system):
        """Test getting applicable rules."""
        def condition1(context: Dict[str, Any]) -> bool:
            return context.get("type") == "test1"
        
        def condition2(context: Dict[str, Any]) -> bool:
            return context.get("type") == "test2"
        
        def action(context: Dict[str, Any]) -> str:
            return "action"
        
        learning_system.add_rule(
            rule_id="rule1",
            condition=condition1,
            action=action
        )
        learning_system.add_rule(
            rule_id="rule2",
            condition=condition2,
            action=action
        )
        
        rules = learning_system.get_applicable_rules({"type": "test1"})
        assert len(rules) == 1
        assert rules[0].id == "rule1"
    
    def test_apply_rules(self, learning_system):
        """Test applying rules."""
        def condition(context: Dict[str, Any]) -> bool:
            return True
        
        def action(context: Dict[str, Any]) -> str:
            return context.get("input", "") + "_processed"
        
        learning_system.add_rule(
            rule_id="test_rule",
            condition=condition,
            action=action
        )
        
        results = learning_system.apply_rules({"input": "test"})
        assert len(results) == 1
        rule, result = results[0]
        assert result == "test_processed"
        assert rule.usage_count == 1
        assert rule.last_used is not None
    
    def test_update_rule_confidence(self, learning_system):
        """Test updating rule confidence."""
        def condition(context: Dict[str, Any]) -> bool:
            return True
        
        def action(context: Dict[str, Any]) -> str:
            return "action"
        
        rule = learning_system.add_rule(
            rule_id="test_rule",
            condition=condition,
            action=action
        )
        
        learning_system.update_rule_confidence("test_rule", True)
        assert rule.usage_count == 1
        assert rule.success_count == 1
        assert rule.confidence == 1.0
        
        learning_system.update_rule_confidence("test_rule", False)
        assert rule.usage_count == 2
        assert rule.success_count == 1
        assert rule.confidence == 0.5
    
    def test_get_similar_examples(self, learning_system):
        """Test getting similar examples."""
        example1 = learning_system.add_example(
            example_id="test1",
            context={"type": "handler", "path": "/users"},
            decision={"option": "A"},
            outcome={"success": True}
        )
        
        example2 = learning_system.add_example(
            example_id="test2",
            context={"type": "handler", "path": "/products"},
            decision={"option": "B"},
            outcome={"success": True}
        )
        
        example3 = learning_system.add_example(
            example_id="test3",
            context={"type": "model"},
            decision={"option": "C"},
            outcome={"success": True}
        )
        
        similar = learning_system.get_similar_examples(
            {"type": "handler", "path": "/orders"}
        )
        
        assert len(similar) == 2
        assert all(ex.context["type"] == "handler" for ex in similar)
    
    def test_save_and_load(self, learning_system, sample_learning_example):
        """Test saving and loading examples."""
        # Add an example
        learning_system.add_example(
            example_id=sample_learning_example["id"],
            context=sample_learning_example["context"],
            decision=sample_learning_example["decision"],
            outcome=sample_learning_example["outcome"],
            tags=sample_learning_example["tags"]
        )
        
        # Save examples
        learning_system.save_examples()
        
        # Create new system and load examples
        new_system = LearningSystem(learning_system.storage_path)
        
        # Verify examples were loaded
        assert len(new_system.examples) == 1
        example = new_system.examples[0]
        assert example.id == sample_learning_example["id"]
        assert example.context == sample_learning_example["context"]
        assert example.decision == sample_learning_example["decision"]
        assert example.outcome == sample_learning_example["outcome"]
        assert example.tags == sample_learning_example["tags"]
    
    def test_get_examples_by_tag(self, learning_system):
        """Test getting examples by tag."""
        learning_system.add_example(
            example_id="test1",
            context={},
            decision={},
            outcome={},
            tags=["tag1", "tag2"]
        )
        
        learning_system.add_example(
            example_id="test2",
            context={},
            decision={},
            outcome={},
            tags=["tag2", "tag3"]
        )
        
        examples = learning_system.get_examples_by_tag("tag2")
        assert len(examples) == 2
        
        examples = learning_system.get_examples_by_tag("tag1")
        assert len(examples) == 1
        assert examples[0].id == "test1"
    
    def test_add_feedback(self, learning_system):
        """Test adding feedback to examples."""
        example = learning_system.add_example(
            example_id="test",
            context={},
            decision={},
            outcome={}
        )
        
        updated = learning_system.add_feedback(
            example_id="test",
            feedback="Test feedback"
        )
        
        assert updated == example
        assert updated.feedback == "Test feedback"
    
    def test_get_statistics(self, learning_system):
        """Test getting system statistics."""
        # Add examples
        learning_system.add_example(
            example_id="test1",
            context={},
            decision={},
            outcome={},
            tags=["tag1"]
        )
        learning_system.add_example(
            example_id="test2",
            context={},
            decision={},
            outcome={},
            tags=["tag1", "tag2"]
        )
        
        # Add rules
        def condition(context: Dict[str, Any]) -> bool:
            return True
        
        def action(context: Dict[str, Any]) -> str:
            return "action"
        
        learning_system.add_rule(
            rule_id="rule1",
            condition=condition,
            action=action
        )
        
        stats = learning_system.get_statistics()
        assert stats["total_examples"] == 2
        assert stats["total_rules"] == 1
        assert stats["most_common_tags"]["tag1"] == 2
        assert stats["most_common_tags"]["tag2"] == 1 