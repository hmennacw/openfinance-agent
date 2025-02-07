from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

T = TypeVar('T')

class DecisionOutcome(Enum):
    """Possible outcomes of a decision."""
    SUCCESSFUL = "successful"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    DEFERRED = "deferred"

@dataclass
class Decision(Generic[T]):
    """Represents a decision made by the system."""
    id: str
    context: str
    options: List[T]
    chosen_option: Optional[T] = None
    outcome: DecisionOutcome = DecisionOutcome.UNCERTAIN
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DecisionCriteria:
    """Criteria for evaluating options."""
    name: str
    weight: float
    evaluation_fn: Callable[[Any], float]
    description: str = ""

class DecisionMaker:
    """Makes decisions based on defined criteria and context."""
    
    def __init__(self):
        self.criteria: Dict[str, DecisionCriteria] = {}
        self.decisions: List[Decision] = []
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = 0.7
    
    def add_criterion(
        self,
        name: str,
        weight: float,
        evaluation_fn: Callable[[Any], float],
        description: str = ""
    ) -> None:
        """Add a criterion for decision making."""
        if weight < 0 or weight > 1:
            raise ValueError("Weight must be between 0 and 1")
        
        self.criteria[name] = DecisionCriteria(
            name=name,
            weight=weight,
            evaluation_fn=evaluation_fn,
            description=description
        )
    
    def evaluate_option(
        self,
        option: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Evaluate a single option against all criteria."""
        if not self.criteria:
            raise ValueError("No criteria defined for decision making")
        
        total_score = 0.0
        total_weight = 0.0
        
        for criterion in self.criteria.values():
            try:
                score = criterion.evaluation_fn(option)
                total_score += score * criterion.weight
                total_weight += criterion.weight
            except Exception as e:
                self.logger.error(
                    f"Error evaluating criterion {criterion.name}: {str(e)}"
                )
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def make_decision(
        self,
        decision_id: str,
        context: str,
        options: List[T],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Decision[T]:
        """Make a decision by evaluating all options."""
        if not options:
            raise ValueError("No options provided for decision making")
        
        scores: List[tuple[T, float]] = []
        for option in options:
            score = self.evaluate_option(option, additional_context)
            scores.append((option, score))
        
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        best_option, best_score = scores[0]
        
        # Determine the outcome based on confidence
        outcome = DecisionOutcome.SUCCESSFUL
        if best_score < self.confidence_threshold:
            if len(scores) > 1 and scores[0][1] - scores[1][1] < 0.1:
                outcome = DecisionOutcome.UNCERTAIN
            else:
                outcome = DecisionOutcome.DEFERRED
        
        # Create the reasoning string
        reasoning_parts = [
            f"Selected option scored {best_score:.2f}",
            "Evaluation breakdown:"
        ]
        for criterion in self.criteria.values():
            try:
                score = criterion.evaluation_fn(best_option)
                reasoning_parts.append(
                    f"- {criterion.name}: {score:.2f} (weight: {criterion.weight})"
                )
            except Exception as e:
                reasoning_parts.append(
                    f"- {criterion.name}: Error in evaluation: {str(e)}"
                )
        
        decision = Decision(
            id=decision_id,
            context=context,
            options=options,
            chosen_option=best_option,
            outcome=outcome,
            confidence=best_score,
            reasoning="\n".join(reasoning_parts),
            metadata=additional_context or {}
        )
        
        self.decisions.append(decision)
        return decision
    
    def get_decision_history(
        self,
        context_filter: Optional[str] = None
    ) -> List[Decision]:
        """Get the history of decisions, optionally filtered by context."""
        if context_filter:
            return [d for d in self.decisions if context_filter in d.context]
        return self.decisions
    
    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Get a specific decision by ID."""
        for decision in self.decisions:
            if decision.id == decision_id:
                return decision
        return None
    
    def revise_decision(
        self,
        decision_id: str,
        new_outcome: DecisionOutcome,
        reasoning: str
    ) -> Optional[Decision]:
        """Revise a previous decision with new information."""
        decision = self.get_decision(decision_id)
        if not decision:
            return None
        
        decision.outcome = new_outcome
        decision.reasoning += f"\n\nRevision ({datetime.now()}):\n{reasoning}"
        return decision
    
    def clear_history(self) -> None:
        """Clear the decision history."""
        self.decisions.clear()
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for decisions."""
        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        self.confidence_threshold = threshold

class CodeGenerationDecisionMaker(DecisionMaker):
    """Specialized decision maker for code generation tasks."""
    
    def __init__(self):
        super().__init__()
        self._setup_criteria()
    
    def _setup_criteria(self) -> None:
        """Set up the default criteria for code generation decisions."""
        
        def evaluate_complexity(option: Dict[str, Any]) -> float:
            """Evaluate the complexity of a code generation approach."""
            complexity_score = option.get("complexity_score", 0)
            return 1.0 - (complexity_score / 10)  # Normalize to 0-1
        
        def evaluate_maintainability(option: Dict[str, Any]) -> float:
            """Evaluate the maintainability of generated code."""
            maintainability_factors = [
                option.get("readability", 0),
                option.get("modularity", 0),
                option.get("testability", 0)
            ]
            return sum(maintainability_factors) / (len(maintainability_factors) * 10)
        
        def evaluate_performance(option: Dict[str, Any]) -> float:
            """Evaluate the performance characteristics."""
            performance_score = option.get("performance_score", 0)
            return performance_score / 10
        
        def evaluate_best_practices(option: Dict[str, Any]) -> float:
            """Evaluate adherence to Go best practices."""
            practices_score = option.get("best_practices_score", 0)
            return practices_score / 10
        
        # Add the criteria with appropriate weights
        self.add_criterion(
            name="complexity",
            weight=0.25,
            evaluation_fn=evaluate_complexity,
            description="Evaluates the complexity of the solution"
        )
        
        self.add_criterion(
            name="maintainability",
            weight=0.3,
            evaluation_fn=evaluate_maintainability,
            description="Evaluates code maintainability"
        )
        
        self.add_criterion(
            name="performance",
            weight=0.2,
            evaluation_fn=evaluate_performance,
            description="Evaluates runtime performance"
        )
        
        self.add_criterion(
            name="best_practices",
            weight=0.25,
            evaluation_fn=evaluate_best_practices,
            description="Evaluates adherence to Go best practices"
        ) 