import pytest
from datetime import datetime
from typing import Dict, Any, List
from src.cognitive.decision import (
    DecisionOutcome,
    Decision,
    DecisionCriteria,
    DecisionMaker,
    CodeGenerationDecisionMaker
)

def test_decision_creation():
    """Test basic decision creation."""
    decision = Decision(
        id="test_decision",
        context="test context",
        options=["option1", "option2"]
    )
    assert decision.id == "test_decision"
    assert decision.context == "test context"
    assert decision.options == ["option1", "option2"]
    assert decision.outcome == DecisionOutcome.UNCERTAIN
    assert isinstance(decision.timestamp, datetime)

def test_decision_criteria_creation():
    """Test decision criteria creation."""
    def eval_fn(x: Any) -> float:
        return 1.0
    
    criteria = DecisionCriteria(
        name="test_criteria",
        weight=0.5,
        evaluation_fn=eval_fn,
        description="Test criteria"
    )
    assert criteria.name == "test_criteria"
    assert criteria.weight == 0.5
    assert criteria.description == "Test criteria"
    assert criteria.evaluation_fn(None) == 1.0

class TestDecisionOutcome:
    """Test suite for DecisionOutcome enum."""
    
    def test_decision_outcome_values(self):
        """Test that all expected outcomes are defined."""
        assert DecisionOutcome.SUCCESSFUL.value == "successful"
        assert DecisionOutcome.FAILED.value == "failed"
        assert DecisionOutcome.UNCERTAIN.value == "uncertain"
        assert DecisionOutcome.DEFERRED.value == "deferred"

class TestDecision:
    """Test suite for Decision dataclass."""
    
    def test_decision_initialization(self):
        """Test Decision initialization with minimal parameters."""
        decision = Decision(
            id="test_id",
            context="test context",
            options=["option1", "option2"]
        )
        assert decision.id == "test_id"
        assert decision.context == "test context"
        assert decision.options == ["option1", "option2"]
        assert decision.chosen_option is None
        assert decision.outcome == DecisionOutcome.UNCERTAIN
        assert decision.confidence == 0.0
        assert decision.reasoning == ""
        assert isinstance(decision.metadata, dict)
        assert isinstance(decision.timestamp, datetime)
    
    def test_decision_full_initialization(self):
        """Test Decision initialization with all parameters."""
        metadata = {"key": "value"}
        timestamp = datetime.now()
        decision = Decision(
            id="test_id",
            context="test context",
            options=["option1", "option2"],
            chosen_option="option1",
            outcome=DecisionOutcome.SUCCESSFUL,
            confidence=0.9,
            reasoning="test reasoning",
            metadata=metadata,
            timestamp=timestamp
        )
        assert decision.chosen_option == "option1"
        assert decision.outcome == DecisionOutcome.SUCCESSFUL
        assert decision.confidence == 0.9
        assert decision.reasoning == "test reasoning"
        assert decision.metadata == metadata
        assert decision.timestamp == timestamp

class TestDecisionCriteria:
    """Test suite for DecisionCriteria dataclass."""
    
    def test_criteria_initialization(self):
        """Test DecisionCriteria initialization."""
        def dummy_eval(x: Any) -> float:
            return 1.0
        
        criteria = DecisionCriteria(
            name="test",
            weight=0.5,
            evaluation_fn=dummy_eval,
            description="test description"
        )
        assert criteria.name == "test"
        assert criteria.weight == 0.5
        assert criteria.evaluation_fn == dummy_eval
        assert criteria.description == "test description"
    
    def test_criteria_evaluation(self):
        """Test that evaluation function works."""
        def test_eval(x: Any) -> float:
            return float(x)
        
        criteria = DecisionCriteria(
            name="test",
            weight=1.0,
            evaluation_fn=test_eval
        )
        assert criteria.evaluation_fn(5) == 5.0

class TestDecisionMaker:
    """Test suite for DecisionMaker class."""
    
    @pytest.fixture
    def decision_maker(self):
        """Create a DecisionMaker instance."""
        return DecisionMaker()
    
    def test_initialization(self, decision_maker):
        """Test DecisionMaker initialization."""
        assert decision_maker.criteria == {}
        assert decision_maker.decisions == []
        assert decision_maker.confidence_threshold == 0.7
    
    def test_add_criterion(self, decision_maker):
        """Test adding a criterion."""
        def dummy_eval(x: Any) -> float:
            return 1.0
        
        decision_maker.add_criterion(
            name="test",
            weight=0.5,
            evaluation_fn=dummy_eval,
            description="test description"
        )
        
        assert "test" in decision_maker.criteria
        criterion = decision_maker.criteria["test"]
        assert criterion.name == "test"
        assert criterion.weight == 0.5
        assert criterion.evaluation_fn == dummy_eval
        assert criterion.description == "test description"
    
    def test_add_criterion_invalid_weight(self, decision_maker):
        """Test adding a criterion with invalid weight."""
        def dummy_eval(x: Any) -> float:
            return 1.0
        
        with pytest.raises(ValueError):
            decision_maker.add_criterion(
                name="test",
                weight=1.5,  # Invalid weight > 1
                evaluation_fn=dummy_eval
            )
        
        with pytest.raises(ValueError):
            decision_maker.add_criterion(
                name="test",
                weight=-0.5,  # Invalid weight < 0
                evaluation_fn=dummy_eval
            )
    
    def test_evaluate_option(self, decision_maker):
        """Test option evaluation."""
        def eval_fn1(x: Any) -> float:
            return 0.8
        
        def eval_fn2(x: Any) -> float:
            return 0.6
        
        decision_maker.add_criterion("c1", 0.6, eval_fn1)
        decision_maker.add_criterion("c2", 0.4, eval_fn2)
        
        score = decision_maker.evaluate_option("test_option")
        expected_score = (0.8 * 0.6 + 0.6 * 0.4)
        assert score == pytest.approx(expected_score)
    
    def test_evaluate_option_no_criteria(self, decision_maker):
        """Test evaluation with no criteria."""
        with pytest.raises(ValueError):
            decision_maker.evaluate_option("test_option")
    
    def test_evaluate_option_with_error(self, decision_maker):
        """Test evaluation when criterion raises error."""
        def error_fn(x: Any) -> float:
            raise ValueError("Test error")
        
        decision_maker.add_criterion("error", 1.0, error_fn)
        score = decision_maker.evaluate_option("test_option")
        assert score == 0.0
    
    def test_make_decision(self, decision_maker):
        """Test making a decision."""
        def eval_fn(x: Any) -> float:
            return 0.8 if x == "option1" else 0.6
        
        decision_maker.add_criterion("test", 1.0, eval_fn)
        options = ["option1", "option2"]
        
        decision = decision_maker.make_decision(
            "test_id",
            "test context",
            options
        )
        
        assert decision.id == "test_id"
        assert decision.context == "test context"
        assert decision.options == options
        assert decision.chosen_option == "option1"
        assert decision.outcome == DecisionOutcome.SUCCESSFUL
        assert decision.confidence == pytest.approx(0.8)
        assert "Selected option scored 0.80" in decision.reasoning
    
    def test_make_decision_no_options(self, decision_maker):
        """Test making a decision with no options."""
        with pytest.raises(ValueError):
            decision_maker.make_decision("test_id", "test context", [])
    
    def test_make_decision_uncertain(self, decision_maker):
        """Test making a decision with uncertain outcome."""
        def eval_fn(x: Any) -> float:
            return 0.65 if x == "option1" else 0.62
        
        decision_maker.add_criterion("test", 1.0, eval_fn)
        options = ["option1", "option2"]
        
        decision = decision_maker.make_decision(
            "test_id",
            "test context",
            options
        )
        
        assert decision.outcome == DecisionOutcome.UNCERTAIN
    
    def test_make_decision_deferred(self, decision_maker):
        """Test making a decision with deferred outcome."""
        def eval_fn(x: Any) -> float:
            return 0.65 if x == "option1" else 0.5
        
        decision_maker.add_criterion("test", 1.0, eval_fn)
        options = ["option1", "option2"]
        
        decision = decision_maker.make_decision(
            "test_id",
            "test context",
            options
        )
        
        assert decision.outcome == DecisionOutcome.DEFERRED
    
    def test_get_decision_history(self, decision_maker):
        """Test getting decision history."""
        def eval_fn(x: Any) -> float:
            return 0.8
        
        decision_maker.add_criterion("test", 1.0, eval_fn)
        
        decision1 = decision_maker.make_decision(
            "id1",
            "context1",
            ["option1"]
        )
        decision2 = decision_maker.make_decision(
            "id2",
            "context2",
            ["option2"]
        )
        
        history = decision_maker.get_decision_history()
        assert len(history) == 2
        assert history[0] == decision1
        assert history[1] == decision2
        
        filtered = decision_maker.get_decision_history("context1")
        assert len(filtered) == 1
        assert filtered[0] == decision1
    
    def test_get_decision(self, decision_maker):
        """Test getting a specific decision."""
        def eval_fn(x: Any) -> float:
            return 0.8
        
        decision_maker.add_criterion("test", 1.0, eval_fn)
        decision = decision_maker.make_decision(
            "test_id",
            "test context",
            ["option1"]
        )
        
        retrieved = decision_maker.get_decision("test_id")
        assert retrieved == decision
        
        assert decision_maker.get_decision("nonexistent") is None
    
    def test_revise_decision(self, decision_maker):
        """Test revising a decision."""
        def eval_fn(x: Any) -> float:
            return 0.8
        
        decision_maker.add_criterion("test", 1.0, eval_fn)
        decision = decision_maker.make_decision(
            "test_id",
            "test context",
            ["option1"]
        )
        
        revised = decision_maker.revise_decision(
            "test_id",
            DecisionOutcome.FAILED,
            "New information"
        )
        
        assert revised == decision
        assert revised.outcome == DecisionOutcome.FAILED
        assert "New information" in revised.reasoning
        
        assert decision_maker.revise_decision(
            "nonexistent",
            DecisionOutcome.FAILED,
            ""
        ) is None
    
    def test_clear_history(self, decision_maker):
        """Test clearing decision history."""
        def eval_fn(x: Any) -> float:
            return 0.8
        
        decision_maker.add_criterion("test", 1.0, eval_fn)
        decision_maker.make_decision("id1", "context", ["option1"])
        decision_maker.make_decision("id2", "context", ["option2"])
        
        assert len(decision_maker.decisions) == 2
        decision_maker.clear_history()
        assert len(decision_maker.decisions) == 0
    
    def test_set_confidence_threshold(self, decision_maker):
        """Test setting confidence threshold."""
        decision_maker.set_confidence_threshold(0.8)
        assert decision_maker.confidence_threshold == 0.8
        
        with pytest.raises(ValueError):
            decision_maker.set_confidence_threshold(1.5)
        
        with pytest.raises(ValueError):
            decision_maker.set_confidence_threshold(-0.5)

    def test_make_decision_with_evaluation_error(self, decision_maker):
        """Test making a decision when criterion evaluation fails."""
        def error_fn(x: Any) -> float:
            raise ValueError("Test error")
        
        decision_maker.add_criterion("error", 1.0, error_fn)
        
        decision = decision_maker.make_decision(
            "test_id",
            "test context",
            ["option1"]
        )
        
        assert decision.id == "test_id"
        assert decision.context == "test context"
        assert decision.chosen_option == "option1"
        assert decision.outcome == DecisionOutcome.DEFERRED
        assert "Error in evaluation: Test error" in decision.reasoning

class TestCodeGenerationDecisionMaker:
    """Test suite for CodeGenerationDecisionMaker class."""
    
    @pytest.fixture
    def code_decision_maker(self):
        """Create a CodeGenerationDecisionMaker instance."""
        return CodeGenerationDecisionMaker()
    
    def test_initialization(self, code_decision_maker):
        """Test initialization and criteria setup."""
        criteria_names = {
            "complexity",
            "maintainability",
            "performance",
            "best_practices"
        }
        assert set(code_decision_maker.criteria.keys()) == criteria_names
    
    def test_evaluate_complexity(self, code_decision_maker):
        """Test complexity evaluation."""
        option = {"complexity_score": 5}
        score = code_decision_maker.criteria["complexity"].evaluation_fn(option)
        assert score == 0.5  # (10 - 5) / 10
    
    def test_evaluate_maintainability(self, code_decision_maker):
        """Test maintainability evaluation."""
        option = {
            "readability": 8,
            "modularity": 7,
            "testability": 9
        }
        score = code_decision_maker.criteria["maintainability"].evaluation_fn(option)
        assert score == pytest.approx(0.8)  # (8 + 7 + 9) / (3 * 10)
    
    def test_evaluate_performance(self, code_decision_maker):
        """Test performance evaluation."""
        option = {"performance_score": 7}
        score = code_decision_maker.criteria["performance"].evaluation_fn(option)
        assert score == 0.7  # 7 / 10
    
    def test_evaluate_best_practices(self, code_decision_maker):
        """Test best practices evaluation."""
        option = {"best_practices_score": 9}
        score = code_decision_maker.criteria["best_practices"].evaluation_fn(option)
        assert score == 0.9  # 9 / 10
    
    def test_make_code_decision(self, code_decision_maker):
        """Test making a code generation decision."""
        options = [
            {
                "complexity_score": 3,
                "readability": 8,
                "modularity": 9,
                "testability": 8,
                "performance_score": 7,
                "best_practices_score": 9
            },
            {
                "complexity_score": 7,
                "readability": 6,
                "modularity": 5,
                "testability": 6,
                "performance_score": 8,
                "best_practices_score": 6
            }
        ]
        
        decision = code_decision_maker.make_decision(
            "test_id",
            "Generate API handler",
            options
        )
        
        assert decision.chosen_option == options[0]
        assert decision.outcome == DecisionOutcome.SUCCESSFUL
        assert decision.confidence > 0.7 