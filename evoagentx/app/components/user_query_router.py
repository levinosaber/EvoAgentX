"""
User Query Router Component

This component is responsible for processing user queries related to AI Agent workflow modifications.
It classifies user intents, handles composite queries, and routes them to appropriate downstream processors.

The component supports the following query types:
- Addition: Add new elements (nodes, agents, functionality)
- Deletion: Remove existing elements (nodes, agents, functionality)  
- Modification: Update existing elements or their properties
- Restructure: Structural adjustments involving multiple elements
- Query: Information requests about current workflow
- Optimization: Performance improvement suggestions
"""
from evoagentx.core import BaseModule
from evoagentx.core.logging import logger

from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json

from pydantic import BaseModel


class QueryType(Enum):
    """Enumeration of supported query types for workflow modifications."""
    
    # Addition types
    ADD_FUNCTIONALITY = "add_functionality"
    ADD_NODE = "add_node"
    ADD_AGENT = "add_agent"
    
    # Deletion types
    DELETE_FUNCTIONALITY = "delete_functionality"
    DELETE_NODE = "delete_node"
    DELETE_AGENT = "delete_agent"
    
    # Modification types
    MODIFY_FUNCTIONALITY = "modify_functionality"
    MODIFY_NODE = "modify_node"
    MODIFY_AGENT = "modify_agent"
    
    # Structural adjustment
    RESTRUCTURE = "restructure"
    
    # Information and optimization
    QUERY = "query"
    OPTIMIZATION = "optimization"


class ParsedIntent(BaseModel):
    """Represents a single parsed intent from user query."""
    
    intent_type: QueryType
    target_object: Optional[str] = None  # ID or name of target node/agent
    description: str = ""  # Detailed description of the intent
    parameters: Dict[str, Any] = None  # Additional parameters for the operation
    confidence: float = 0.0  # Confidence score from LLM parsing
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class QueryParseResult(BaseModel):
    """Result of query parsing containing all identified intents."""
    
    original_query: str
    parsed_intents: List[ParsedIntent]
    is_composite: bool = False  # Whether query contains multiple intents
    rewritten_queries: List[str] = None  # Rewritten atomic queries if applicable
    workflow_context: Dict[str, Any] = None  # Current workflow state for context
    
    def __post_init__(self):
        if self.rewritten_queries is None:
            self.rewritten_queries = []
        if self.workflow_context is None:
            self.workflow_context = {}


class UserQueryRouter(BaseModule):
    """
    Main component for routing and processing user queries related to workflow modifications.
    
    This component uses LLM-based intent recognition to classify user requests and
    prepare them for downstream processing. It handles both simple and composite
    queries, performing necessary query rewriting and intent extraction.
    """
    
    def __init__(
        self,
        llm_model_name: str = "gpt-4o-mini",
        max_retries: int = 3,
        temperature: float = 0.1,
        enable_traditional_ml: bool = False,
        logger: Optional[Any] = logger
    ):
        """
        Initialize the UserQueryRouter with configuration parameters.
        
        Args:
            llm_model_name: Name of the LLM model to use for intent recognition
            max_retries: Maximum number of retries for LLM API calls
            temperature: Temperature setting for LLM to control randomness
            enable_traditional_ml: Whether to enable traditional ML fallback methods
            logger: Logger instance for debugging and monitoring
        """
        self.llm_model_name = llm_model_name
        self.max_retries = max_retries
        self.temperature = temperature
        self.enable_traditional_ml = enable_traditional_ml
        self.logger = logger
        
        # Initialize LLM client and other components
        self._llm_client = None
        self._traditional_ml_classifier = None
        
        # Prompt templates for different scenarios
        self._intent_classification_prompt = ""
        self._composite_query_decomposition_prompt = ""
        self._query_rewrite_prompt = ""
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize LLM client and other necessary components."""
        # TODO: Initialize LLM client based on model name
        # TODO: Load traditional ML models if enabled
        # TODO: Load and prepare prompt templates
        pass
    
    def route_query(
        self,
        user_query: str,
        workflow_context: Dict[str, Any]
    ) -> QueryParseResult:
        """
        Main entry point for processing user queries.
        
        This method orchestrates the entire query processing pipeline:
        1. Classify the query type and detect composite intents
        2. Decompose composite queries into atomic operations
        3. Rewrite queries if necessary for clarity
        4. Extract structured intent information
        
        Args:
            user_query: Raw user input describing desired modifications(both front end and workflow side are available)
            workflow_context: Current workflow state including nodes, agents, and structure, use json file generated by Workflow.to_dict() or Workflow.save_module(), two methods inherited from BaseModule.
            
        Returns:
            QueryParseResult containing parsed intents and processing metadata
        """
        # TODO: Implement main routing logic
        pass
    
    def classify_intent(
        self,
        query: str,
        workflow_context: Dict[str, Any]
    ) -> List[QueryType]:
        """
        Classify the intent type(s) present in the user query.
        
        Uses LLM-based classification to identify which types of operations
        the user is requesting. Handles both single and multiple intents.
        
        Args:
            query: User query to classify
            workflow_context: Current workflow state for context
            
        Returns:
            List of identified query types
        """
        # TODO: Implement LLM-based intent classification
        pass
    
    def decompose_composite_query(
        self,
        query: str,
        workflow_context: Dict[str, Any]
    ) -> List[str]:
        """
        Decompose a composite query into multiple atomic operations.
        
        When a user query contains multiple modification requests,
        this method breaks it down into individual, actionable operations.
        
        Args:
            query: Composite user query
            workflow_context: Current workflow state for context
            
        Returns:
            List of atomic query strings
        """
        # TODO: Implement composite query decomposition using LLM
        pass
    
    def rewrite_query(
        self,
        query: str,
        intent_type: QueryType,
        workflow_context: Dict[str, Any]
    ) -> str:
        """
        Rewrite user query for better clarity and processing.
        
        Sometimes user queries are ambiguous or lack specific details.
        This method generates clearer, more actionable versions of the query.
        
        Args:
            query: Original user query
            intent_type: Identified intent type
            workflow_context: Current workflow state for context
            
        Returns:
            Rewritten query with improved clarity
        """
        # TODO: Implement query rewriting logic
        pass
    
    def extract_intent_details(
        self,
        query: str,
        intent_type: QueryType,
        workflow_context: Dict[str, Any]
    ) -> ParsedIntent:
        """
        Extract detailed information from a classified query.
        
        After classification, this method extracts specific details like
        target objects, parameters, and operation specifics.
        
        Args:
            query: User query (potentially rewritten)
            intent_type: Classified intent type
            workflow_context: Current workflow state for context
            
        Returns:
            ParsedIntent object with extracted details
        """
        # TODO: Implement intent detail extraction
        pass
    
    def validate_intent(
        self,
        intent: ParsedIntent,
        workflow_context: Dict[str, Any]
    ) -> bool:
        """
        Validate that the parsed intent is feasible given current workflow state.
        
        Checks if the requested operation can be performed on the current
        workflow structure (e.g., target objects exist, operations are valid).
        
        Args:
            intent: Parsed intent to validate
            workflow_context: Current workflow state
            
        Returns:
            True if intent is valid and feasible, False otherwise
        """
        # TODO: Implement intent validation logic
        pass
    
    def prepare_downstream_request(
        self,
        parse_result: QueryParseResult
    ) -> Dict[str, Any]:
        """
        Prepare standardized request format for downstream processors.
        
        Converts the parsed query result into a standardized format
        that downstream workflow modification components can process.
        
        Args:
            parse_result: Complete query parse result
            
        Returns:
            Standardized request dictionary for downstream processing
        """
        # TODO: Implement downstream request preparation
        pass
    
    def _call_llm(
        self,
        prompt: str,
        expected_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Make a call to the LLM with error handling and retries.
        
        Args:
            prompt: Formatted prompt for the LLM
            expected_format: Expected response format (json, text, etc.)
            
        Returns:
            Parsed LLM response
        """
        # TODO: Implement LLM API call with retry logic
        pass
    
    def _traditional_ml_strategy(
        self,
        query: str
    ) -> QueryType:
        """
        Used when hybrid mode or pure ML mode is enabled.
        
        Args:
            query: User query to classify
            
        Returns:
            Classified query type using traditional ML methods
        """
        # TODO: Implement traditional ML fallback classification
        pass
    
    def _build_intent_classification_prompt(
        self,
        query: str,
        workflow_context: Dict[str, Any]
    ) -> str:
        """
        Build prompt for LLM-based intent classification.
        
        Args:
            query: User query to classify
            workflow_context: Current workflow context
            
        Returns:
            Formatted prompt for intent classification
        """
        # TODO: Implement prompt building for intent classification
        pass
    
    def _build_decomposition_prompt(
        self,
        query: str,
        workflow_context: Dict[str, Any]
    ) -> str:
        """
        Build prompt for composite query decomposition.
        
        Args:
            query: Composite query to decompose
            workflow_context: Current workflow context
            
        Returns:
            Formatted prompt for query decomposition
        """
        # TODO: Implement prompt building for query decomposition
        pass
    
    def get_supported_query_types(self) -> List[str]:
        """
        Get list of all supported query types.
        
        Returns:
            List of supported query type names
        """
        return [query_type.value for query_type in QueryType]
    
    def get_component_status(self) -> Dict[str, Any]:
        """
        Get current status and configuration of the router component.
        
        Returns:
            Dictionary containing component status and configuration
        """
        return {
            "llm_model": self.llm_model_name,
            "traditional_ml_enabled": self.enable_traditional_ml,
            "supported_types": self.get_supported_query_types(),
            "component_ready": self._llm_client is not None
        }
