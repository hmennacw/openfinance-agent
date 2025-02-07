import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import tempfile
import os
import logging
import shutil
from unittest.mock import MagicMock

from src.cognitive.learning import (
    LearningExample,
    Rule,
    LearningSystem
)

def test_learning_example_creation():
    """Test basic learning example creation."""
    example = LearningExample(
        id="test_example",
        context={"input": "test"},
        decision={"choice": "A"},
        outcome={"success": True}
    )
    assert example.id == "test_example"
    assert example.context == {"input": "test"}
    assert example.decision == {"choice": "A"}
    assert example.outcome == {"success": True}
    assert example.feedback is None
    assert isinstance(example.timestamp, datetime)
    assert example.tags == []

def test_learning_example_with_optional_fields():
    """Test learning example creation with optional fields."""
    timestamp = datetime.now()
    example = LearningExample(
        id="test_example",
        context={"input": "test"},
        decision={"choice": "A"},
        outcome={"success": True},
        feedback="Good choice",
        timestamp=timestamp,
        tags=["test", "example"]
    )
    assert example.feedback == "Good choice"
    assert example.timestamp == timestamp
    assert example.tags == ["test", "example"]

def test_rule_creation():
    """Test basic rule creation."""
    def condition(context: Dict[str, Any]) -> bool:
        return context.get("value", 0) > 10

    def action(context: Dict[str, Any]) -> str:
        return "high" if context.get("value", 0) > 10 else "low"

    rule = Rule(
        id="test_rule",
        condition=condition,
        action=action,
        description="Test value threshold"
    )
    assert rule.id == "test_rule"
    assert rule.condition({"value": 15}) is True
    assert rule.condition({"value": 5}) is False
    assert rule.action({"value": 15}) == "high"
    assert rule.confidence == 0.0
    assert rule.usage_count == 0
    assert rule.success_count == 0
    assert isinstance(rule.created_at, datetime)
    assert rule.last_used is None
    assert rule.description == "Test value threshold"

class TestLearningSystem:
    """Test suite for LearningSystem."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary file for storage."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def learning_system(self, temp_storage_path):
        """Create a learning system instance."""
        return LearningSystem(storage_path=temp_storage_path)
    
    def test_add_example(self, learning_system):
        """Test adding a learning example."""
        example = learning_system.add_example(
            example_id="test1",
            context={"input": "test"},
            decision={"choice": "A"},
            outcome={"success": True},
            feedback="Good",
            tags=["test"]
        )
        assert example.id == "test1"
        assert len(learning_system.examples) == 1
        assert learning_system.examples[0] == example
    
    def test_add_rule(self, learning_system):
        """Test adding a rule."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return True
        
        def action(ctx: Dict[str, Any]) -> str:
            return "test"
        
        rule = learning_system.add_rule(
            rule_id="rule1",
            condition=condition,
            action=action,
            description="Test rule"
        )
        assert rule.id == "rule1"
        assert rule in learning_system.rules.values()
    
    def test_get_applicable_rules(self, learning_system):
        """Test getting applicable rules."""
        def condition1(ctx: Dict[str, Any]) -> bool:
            return ctx.get("value", 0) > 10

        def condition2(ctx: Dict[str, Any]) -> bool:
            return ctx.get("value", 0) < 5

        def action(ctx: Dict[str, Any]) -> str:
            return "test"

        learning_system.add_rule("rule1", condition1, action)
        learning_system.add_rule("rule2", condition2, action)
        
        rules = learning_system.get_applicable_rules({"value": 15})
        assert len(rules) == 1
        assert rules[0].id == "rule1"
    
    def test_apply_rules(self, learning_system):
        """Test applying rules."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return True

        def action(ctx: Dict[str, Any]) -> str:
            return ctx.get("input", "") + "_processed"

        rule = learning_system.add_rule("rule1", condition, action)
        
        results = learning_system.apply_rules({"input": "test"})
        assert len(results) == 1
        assert results[0][0] == rule
        assert results[0][1] == "test_processed"
        assert rule.usage_count == 1
        assert rule.last_used is not None
    
    def test_update_rule_confidence(self, learning_system):
        """Test updating rule confidence."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return True

        def action(ctx: Dict[str, Any]) -> str:
            return "test"

        rule = learning_system.add_rule("test_rule", condition, action)
        
        # Initial state
        assert rule.usage_count == 0
        assert rule.success_count == 0
        assert rule.confidence == 0.0
        
        # First update - success
        learning_system.update_rule_confidence("test_rule", True)
        assert rule.usage_count == 1
        assert rule.success_count == 1
        assert rule.confidence == 1.0  # 1/1
        
        # Second update - failure
        learning_system.update_rule_confidence("test_rule", False)
        assert rule.usage_count == 2
        assert rule.success_count == 1
        assert rule.confidence == 0.5  # 1/2
        
        # Third update - success
        learning_system.update_rule_confidence("test_rule", True)
        assert rule.usage_count == 3
        assert rule.success_count == 2
        assert rule.confidence == 2/3  # 2/3
        
        # Test with non-existent rule
        learning_system.update_rule_confidence("non_existent", True)  # Should not raise error
    
    def test_update_rule_confidence_with_multiple_updates(self, learning_system):
        """Test updating rule confidence with multiple updates."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return True

        def action(ctx: Dict[str, Any]) -> str:
            return "test"

        rule = learning_system.add_rule("rule1", condition, action)
        
        # First update - success
        learning_system.update_rule_confidence("rule1", True)
        assert rule.usage_count == 1
        assert rule.success_count == 1
        assert rule.confidence == 1.0  # 1/1
        
        # Second update - failure
        learning_system.update_rule_confidence("rule1", False)
        assert rule.usage_count == 2
        assert rule.success_count == 1
        assert rule.confidence == 0.5  # 1/2
        
        # Third update - success
        learning_system.update_rule_confidence("rule1", True)
        assert rule.usage_count == 3
        assert rule.success_count == 2
        assert rule.confidence == 2/3  # 2/3
    
    def test_get_similar_examples(self, learning_system):
        """Test getting similar examples."""
        example1 = learning_system.add_example(
            "ex1",
            context={"type": "A", "value": 1},
            decision={"choice": "X"},
            outcome={"success": True}
        )
        example2 = learning_system.add_example(
            "ex2",
            context={"type": "A", "value": 2},
            decision={"choice": "Y"},
            outcome={"success": False}
        )
        example3 = learning_system.add_example(
            "ex3",
            context={"type": "B", "value": 3},
            decision={"choice": "Z"},
            outcome={"success": True}
        )
        
        similar = learning_system.get_similar_examples({"type": "A", "value": 1})
        assert len(similar) == 2
        assert similar[0] == example1  # Most similar
        assert similar[1] == example2  # Less similar
    
    def test_save_and_load_examples(self, learning_system, temp_storage_path):
        """Test saving and loading examples."""
        example = learning_system.add_example(
            "test1",
            context={"input": "test"},
            decision={"choice": "A"},
            outcome={"success": True},
            feedback="Good",
            tags=["test"]
        )
        
        # Create new instance to load saved examples
        new_system = LearningSystem(storage_path=temp_storage_path)
        loaded_example = new_system.examples[0]
        
        assert len(new_system.examples) == 1
        assert loaded_example.id == example.id
        assert loaded_example.context == example.context
        assert loaded_example.decision == example.decision
        assert loaded_example.outcome == example.outcome
        assert loaded_example.feedback == example.feedback
        assert loaded_example.tags == example.tags
    
    def test_get_examples_by_tag(self, learning_system):
        """Test getting examples by tag."""
        example1 = learning_system.add_example(
            "ex1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True},
            tags=["tag1", "tag2"]
        )
        example2 = learning_system.add_example(
            "ex2",
            context={"test": True},
            decision={"choice": "B"},
            outcome={"success": False},
            tags=["tag2"]
        )
        
        tag1_examples = learning_system.get_examples_by_tag("tag1")
        assert len(tag1_examples) == 1
        assert tag1_examples[0] == example1
        
        tag2_examples = learning_system.get_examples_by_tag("tag2")
        assert len(tag2_examples) == 2
        assert example1 in tag2_examples
        assert example2 in tag2_examples
    
    def test_get_example_by_id(self, learning_system):
        """Test getting example by ID."""
        example = learning_system.add_example(
            "test1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True}
        )
        
        found = learning_system.get_example_by_id("test1")
        assert found == example
        assert learning_system.get_example_by_id("nonexistent") is None
    
    def test_add_feedback(self, learning_system):
        """Test adding feedback to an example."""
        example = learning_system.add_example(
            "test1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True}
        )
        
        updated = learning_system.add_feedback("test1", "Good choice")
        assert updated == example
        assert updated.feedback == "Good choice"
        
        assert learning_system.add_feedback("nonexistent", "feedback") is None
    
    def test_clear_examples(self, learning_system):
        """Test clearing all examples."""
        learning_system.add_example(
            "test1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True}
        )
        
        assert len(learning_system.examples) == 1
        learning_system.clear_examples()
        assert len(learning_system.examples) == 0
    
    def test_get_statistics(self, learning_system):
        """Test getting system statistics."""
        learning_system.add_example(
            "ex1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True},
            tags=["tag1"]
        )
        learning_system.add_example(
            "ex2",
            context={"test": True},
            decision={"choice": "B"},
            outcome={"success": False},
            tags=["tag1", "tag2"]
        )
        
        stats = learning_system.get_statistics()
        assert stats["total_examples"] == 2
        assert stats["total_rules"] == 0
        assert stats["most_common_tags"]["tag1"] == 2
        assert stats["most_common_tags"]["tag2"] == 1
        assert len(stats["examples_by_month"]) == 1  # All examples are from same month
        assert len(stats["rules_by_confidence"]) == 0  # No rules added
    
    def test_error_handling(self, learning_system):
        """Test error handling in rule evaluation."""
        def failing_condition(ctx: Dict[str, Any]) -> bool:
            raise ValueError("Test error")

        def failing_action(ctx: Dict[str, Any]) -> str:
            raise ValueError("Test error")

        learning_system.add_rule("rule1", failing_condition, failing_action)
        
        # Should handle condition error gracefully
        rules = learning_system.get_applicable_rules({"test": True})
        assert len(rules) == 0
        
        # Should handle action error gracefully
        learning_system.add_rule("rule2", lambda x: True, failing_action)
        results = learning_system.apply_rules({"test": True})
        assert len(results) == 0
    
    def test_invalid_storage_path(self):
        """Test handling of invalid storage path."""
        system = LearningSystem(storage_path="/nonexistent/path/examples.json")
        assert len(system.examples) == 0  # Should not raise error
    
    def test_corrupted_storage_file(self, temp_storage_path):
        """Test handling of corrupted storage file."""
        # Write invalid JSON
        with open(temp_storage_path, "w") as f:
            f.write("invalid json content")
        
        system = LearningSystem(storage_path=temp_storage_path)
        assert len(system.examples) == 0  # Should not raise error
    
    def test_save_examples_no_storage_path(self):
        """Test save_examples when no storage path is set."""
        system = LearningSystem()  # No storage path
        system.add_example(
            "test1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True}
        )
        system.save_examples()  # Should not raise error
    
    def test_learn_from_example_logging(self, learning_system, caplog):
        """Test that _learn_from_example logs the example ID."""
        with caplog.at_level(logging.INFO):
            example = learning_system.add_example(
                "test1",
                context={"test": True},
                decision={"choice": "A"},
                outcome={"success": True}
            )
            assert f"Learning from example {example.id}" in caplog.text 

    def test_save_examples_with_directory_creation(self, tmp_path):
        """Test save_examples creates directory if it doesn't exist."""
        storage_dir = tmp_path / "subdir" / "nested"
        storage_path = storage_dir / "examples.json"
        
        # Remove directory if it exists
        if storage_dir.exists():
            shutil.rmtree(storage_dir)
        
        # Create system without storage path first
        system = LearningSystem()
        system.add_example(
            "test1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True}
        )
        
        # Now set storage path and save
        system.storage_path = str(storage_path)
        assert not storage_dir.exists()
        system.save_examples()
        assert storage_path.exists()
        
        # Cleanup
        shutil.rmtree(storage_dir) 

    def test_save_examples_with_directory_creation_error(self, tmp_path, monkeypatch, caplog):
        """Test save_examples handles directory creation error."""
        storage_dir = tmp_path / "subdir" / "nested"
        storage_path = storage_dir / "examples.json"
        
        # Create system without storage path first
        system = LearningSystem()
        system.add_example(
            "test1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True}
        )
        
        # Mock mkdir to raise an error
        def mock_mkdir(*args, **kwargs):
            raise OSError("Test error")
        
        monkeypatch.setattr(Path, "mkdir", mock_mkdir)
        
        # Now set storage path and try to save
        with caplog.at_level(logging.ERROR):
            system.storage_path = str(storage_path)
            system.save_examples()
            assert "Error saving examples" in caplog.text 

    def test_save_examples_with_file_write_error(self, tmp_path, monkeypatch, caplog):
        """Test save_examples handles file write error."""
        storage_path = tmp_path / "examples.json"
        
        # Create system without storage path first
        system = LearningSystem()
        system.add_example(
            "test1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True}
        )
        
        # Mock open to raise an error
        def mock_open(*args, **kwargs):
            raise OSError("Test error")
        
        monkeypatch.setattr("builtins.open", mock_open)
        
        # Now set storage path and try to save
        with caplog.at_level(logging.ERROR):
            system.storage_path = str(storage_path)
            system.save_examples()
            assert "Error saving examples" in caplog.text 

    def test_calculate_similarity_no_common_keys(self, learning_system):
        """Test similarity calculation with no common keys."""
        context1 = {"a": 1, "b": 2}
        context2 = {"c": 3, "d": 4}
        
        similarity = learning_system._calculate_similarity(context1, context2)
        assert similarity == 0.0
    
    def test_save_examples_with_json_error(self, tmp_path, monkeypatch, caplog):
        """Test save_examples handles JSON serialization error."""
        storage_path = tmp_path / "examples.json"
        
        # Create system without storage path first
        system = LearningSystem()
        system.add_example(
            "test1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True}
        )
        
        # Create an object that can't be JSON serialized
        class UnserializableObject:
            pass
        
        system.examples[0].context["unserializable"] = UnserializableObject()
        
        # Now set storage path and try to save
        with caplog.at_level(logging.ERROR):
            system.storage_path = str(storage_path)
            system.save_examples()
            assert "Error saving examples" in caplog.text 

    def test_learn_from_example_direct(self, learning_system):
        """Test _learn_from_example method directly."""
        # Create a mock logger
        mock_logger = MagicMock()
        learning_system.logger = mock_logger
        
        example = LearningExample(
            id="test1",
            context={"test": True},
            decision={"choice": "A"},
            outcome={"success": True}
        )
        
        # Call the method directly to ensure coverage
        learning_system._learn_from_example(example)
        
        # Verify the logger was called with the correct arguments
        mock_logger.info.assert_called_once_with("Learning from example %s", example.id) 

    def test_learn_from_example_with_real_logger(self):
        # Create a real logger
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)

        # Create a handler that captures log messages
        log_messages = []
        class TestHandler(logging.Handler):
            def emit(self, record):
                log_messages.append(record.getMessage())
        
        handler = TestHandler()
        logger.addHandler(handler)

        # Create a learning system and set its logger
        system = LearningSystem(storage_path=None)
        system.logger = logger

        example = LearningExample(id="test1", context={"key": "value"}, decision={"choice": "A"}, outcome={"result": "success"})

        # Test with INFO level (should log)
        logger.setLevel(logging.INFO)
        system._learn_from_example(example)
        assert len(log_messages) == 1
        assert "Learning from example test1" in log_messages[0]

        # Clear log messages
        log_messages.clear()

        # Test with WARNING level (should not log)
        logger.setLevel(logging.WARNING)
        system._learn_from_example(example)
        assert len(log_messages) == 0

        # Clean up
        logger.removeHandler(handler) 